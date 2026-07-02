"""FastAPI server that exposes the diagnosis pipeline to OpenDX.

OpenDX (`api/server.py`) forwards user case text here as:

    POST /chat  {"messages": [{"role": "user", "content": "<case text>"}]}

and streams our `text/event-stream` response, expecting `progress`, `result`,
and `error` events. We use `sse-starlette` to format `data: ...\\n\\n` frames
and keep the connection alive with periodic pings.
"""
from __future__ import annotations

import json
import logging
import os
from typing import Any, AsyncIterator

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field, field_validator
from sse_starlette.sse import EventSourceResponse

from .pipeline import run_pipeline

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s")
log = logging.getLogger("aegisdx")

app = FastAPI(title="AegisDx", version="0.1.0")


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1)

    @field_validator("messages")
    @classmethod
    def _must_have_user(cls, v: list[ChatMessage]) -> list[ChatMessage]:
        if not any(m.role == "user" and m.content.strip() for m in v):
            raise ValueError("messages must contain at least one non-empty user message")
        return v

    def case_text(self) -> str:
        # OpenDX sends one user message; concatenate any extras for safety.
        return "\n\n".join(m.content for m in self.messages if m.role == "user").strip()


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    try:
        return int(raw) if raw is not None else default
    except ValueError:
        log.warning("Invalid integer for %s=%r, falling back to %d", name, raw, default)
        return default


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat")
async def chat(request: ChatRequest) -> EventSourceResponse:
    case_text = request.case_text()

    async def event_stream() -> AsyncIterator[dict[str, Any]]:
        try:
            async for event in run_pipeline(
                case_text,
                llm_model=os.getenv("LLM_MODEL", "gpt-4.1"),
                aggregate_model=os.getenv("AGGREGATE_MODEL", "gpt-4o-mini"),
                llm_provider=os.getenv("LLM_PROVIDER", "openai"),
                num_inference=_env_int("NUM_INFERENCE", 10),
                add_references=_env_bool("ADD_REFERENCES", True),
            ):
                # sse-starlette serialises {"data": str} → `data: <str>\n\n`,
                # matching what OpenAPI's forwarder parses with `json.loads`.
                yield {"data": json.dumps(event)}
        except Exception as e:  # noqa: BLE001
            log.exception("Pipeline crashed")
            yield {"data": json.dumps({"type": "error", "message": f"Internal error: {e!s}"})}

    return EventSourceResponse(event_stream())


def main() -> None:
    """`uv run aegisdx-server` entry point."""
    import uvicorn

    uvicorn.run(
        "aegisdx.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=_env_int("PORT", 8000),
        log_level="info",
    )


if __name__ == "__main__":
    main()
