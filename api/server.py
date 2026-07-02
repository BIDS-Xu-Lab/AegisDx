"""
API server for clinical case management.

/api/chat runs DiagnosisWorkflow (LangGraph) in-process and streams its
per-node progress + final dict as Server-Sent Events.
"""
import os
import sys
import json
import asyncio
import uuid
from typing import Optional, AsyncGenerator

from fastapi import FastAPI, HTTPException, status, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlmodel import Session
from sse_starlette import EventSourceResponse
from dotenv import load_dotenv

# Load .env BEFORE importing workflow — OPENAI_API_KEY / PUBMED_* must be set
# when DiagnosisWorkflow modules are first imported. Use absolute path so the
# load works regardless of the cwd uvicorn is launched from.
load_dotenv(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))

# AEGISDX_* is canonical; OPENDX_* kept as fallback so a half-migrated .env
# keeps working through the rollout.
def _env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(f"AEGISDX_{name}") or os.environ.get(f"OPENDX_{name}") or default

_ENGINE_DEFAULT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "engine")
AEGISDX_ENGINE_PATH = _env("ENGINE_PATH", _ENGINE_DEFAULT)
if AEGISDX_ENGINE_PATH and AEGISDX_ENGINE_PATH not in sys.path:
    sys.path.insert(0, AEGISDX_ENGINE_PATH)

import database
from auth import get_user_id, get_optional_user_id
from workflow import DiagnosisWorkflow

# For azure_openai, AEGISDX_BASE_MODEL is the Azure *deployment name*, not the model name.
BASE_MODEL = _env("BASE_MODEL", "gpt-4.1")
NUM_INFERENCE = int(_env("NUM_INFERENCE", "10"))
LLM_PROVIDER = _env("LLM_PROVIDER", "openai")
ADD_REASONING = (_env("ADD_REASONING", "false") or "").lower() == "true"
ADD_REFERENCES = (_env("ADD_REFERENCES", "false") or "").lower() == "true"

# Provider → required env var(s) for the precheck.
_PROVIDER_KEY_ENV = {
    "azure_openai": ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION"),
    "openai": ("OPENAI_API_KEY",),
    "anthropic": ("ANTHROPIC_API_KEY",),
}

app = FastAPI(title="AegisDx API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def get_db():
    with Session(database.engine) as session:
        yield session


class ChatRequest(BaseModel):
    case_text: str = Field(..., min_length=1, description="Clinical case text")


class HistoryResponse(BaseModel):
    cases: list[dict]


# Map distinctive print_func strings (workflow.py nodes) → (stage_id, label).
NODE_LABELS = {
    "start initial diagnosis":           ("initial_diagnosis",   "Generating initial differentials"),
    "generating warning diagnosis":      ("warning_diagnosis",   "Generating critical warning diagnoses"),
    "aggregating predictions":           ("cluster_predictions", "Clustering similar predictions"),
    "checking predictions":              ("check_predictions",   "Checking prediction coverage"),
    "generating additional predictions": ("generate_additional", "Expanding differential list"),
    "generating verification":           ("verify_diagnosis",    "Verifying and re-ranking diagnoses"),
    "generating reasoning":              ("generate_reasoning",  "Generating per-diagnosis reasoning"),
    "generating overall reasoning":      ("overall_reasoning",   "Synthesizing overall reasoning"),
    "generating actions":                ("generate_actions",    "Generating action plans"),
    "generating management plan":        ("generate_management", "Generating management plan"),
}


def _classify_progress(msg: str) -> dict:
    low = msg.lower()
    for key, (stage, label) in NODE_LABELS.items():
        if key in low:
            return {"stage": stage, "message": label}
    return {"stage": "log", "message": msg.strip()}


async def workflow_event_generator(
    case_id: str,
    case_text: str,
    user_id: Optional[str],
    db: Session,
) -> AsyncGenerator[str, None]:
    """
    SSE event sequence:
      case_created           {case_id}
      progress (× N nodes)   {stage, message}
      result                 {data: diagnose() dict}
      error                  {message}
    """
    message_id_user = str(uuid.uuid4())
    progress_queue: asyncio.Queue[str] = asyncio.Queue()

    def print_hook(msg: str) -> None:
        # Workflow nodes are async coroutines on this loop; put_nowait is safe.
        try:
            progress_queue.put_nowait(msg)
        except Exception:
            pass

    try:
        if user_id:
            database.create_case(db, case_id=case_id, user_id=user_id, title=case_text[:100])
            database.update_case_status(db, case_id=case_id, status="PROCESSING")
            database.add_message(
                db,
                case_id=case_id,
                user_id=user_id,
                message_id=message_id_user,
                message_data={
                    "from_id": user_id,
                    "message_type": "USER",
                    "text": case_text,
                    "stage": "final",
                },
            )

        yield json.dumps({"type": "case_created", "case_id": case_id})

        # Langchain provider (AzureChatOpenAI / ChatOpenAI / etc) reads its own
        # credentials from env; api_key here is only consumed by legacy code paths
        # that aren't on the workflow's runtime call chain.
        required_envs = _PROVIDER_KEY_ENV.get(LLM_PROVIDER, ())
        missing = [k for k in required_envs if not os.environ.get(k)]
        if missing:
            yield json.dumps({
                "type": "error",
                "message": f"Missing env var(s) for provider '{LLM_PROVIDER}': {', '.join(missing)}",
            })
            if user_id:
                database.update_case_status(db, case_id=case_id, status="ERROR")
            return

        workflow = DiagnosisWorkflow(
            base_model=BASE_MODEL,
            api_key=os.environ.get("OPENAI_API_KEY") or os.environ.get("AZURE_OPENAI_API_KEY", ""),
            num_inference=NUM_INFERENCE,
            add_reasoning=ADD_REASONING,
            add_references=ADD_REFERENCES,
            llm_provider=LLM_PROVIDER,
            print_func=print_hook,
        )

        diagnose_task = asyncio.create_task(workflow.diagnose(case_text))

        while True:
            try:
                msg = await asyncio.wait_for(progress_queue.get(), timeout=0.5)
                yield json.dumps({"type": "progress", **_classify_progress(msg)})
            except asyncio.TimeoutError:
                if diagnose_task.done():
                    break

        while not progress_queue.empty():
            try:
                msg = progress_queue.get_nowait()
                yield json.dumps({"type": "progress", **_classify_progress(msg)})
            except asyncio.QueueEmpty:
                break

        if diagnose_task.exception() is not None:
            err = diagnose_task.exception()
            yield json.dumps({"type": "error", "message": f"Workflow failure: {err}"})
            if user_id:
                database.update_case_status(db, case_id=case_id, status="ERROR")
            return

        result = diagnose_task.result()
        # `result` shape — matches workflow.DiagnosisWorkflow.diagnose():
        #   case_description, predictions, warning_diagnosis,
        #   reasoning (List[str]), overall_reasoning (References footer included),
        #   management, actions

        if user_id:
            message_id_agent = str(uuid.uuid4())
            database.add_message(
                db,
                case_id=case_id,
                user_id=user_id,
                message_id=message_id_agent,
                message_data={
                    "from_id": "agent",
                    "message_type": "AGENT",
                    "text": result.get("overall_reasoning", ""),
                    "payload_json": result,
                    "stage": "final",
                },
            )
            database.update_case_status(db, case_id=case_id, status="COMPLETED")

        yield json.dumps({"type": "result", "data": result})

    except Exception as e:
        yield json.dumps({"type": "error", "message": f"Internal error: {e}"})
        if user_id:
            try:
                database.update_case_status(db, case_id=case_id, status="ERROR")
            except Exception:
                pass


@app.get("/api/health")
async def health():
    return {"status": "ok"}


@app.post("/api/chat")
async def chat(
    request: ChatRequest,
    user_id: Optional[str] = Depends(get_optional_user_id),
    db: Session = Depends(get_db),
):
    case_id = str(uuid.uuid4())
    return EventSourceResponse(workflow_event_generator(case_id, request.case_text, user_id, db))


@app.get("/api/history")
async def get_history(
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    cases = database.get_cases(db, user_id=user_id, limit=100)
    return HistoryResponse(cases=[case.to_dict() for case in cases])


@app.get("/api/cases/{case_id}/full")
async def get_case_full(
    case_id: str,
    user_id: str = Depends(get_user_id),
    db: Session = Depends(get_db),
):
    case_data = database.get_case_full(db, case_id=case_id, user_id=user_id)
    if not case_data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Case not found or you don't have access to this case",
        )
    return case_data


def main():
    import uvicorn
    database.init_db()
    print("Database initialized")
    port = int(os.getenv("PORT", "9627"))
    print(f"Starting server on http://0.0.0.0:{port}")
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)


if __name__ == "__main__":
    main()
