# AegisDx

Agent-based diagnostic reasoning service. AegisDx is the inference backend that
[OpenDX](https://github.com/BIDS-Xu-Lab/opendx) calls through its `AGENT_SERVICE_URL`.

It exposes a single streaming endpoint that takes a clinical case description,
runs a multi-agent reasoning pipeline, and emits Server-Sent Events containing
progress updates and a final structured diagnosis payload.

## Pipeline

For each case the service runs the following agents:

1. **DiagnosisAgent** — sample N candidate diagnoses with chain-of-thought reasoning.
2. **AggregateAgent** — cluster candidates and rank by frequency.
3. **WarningAgent** — select must-not-miss differentials from a curated registry.
4. **ReasoningAgent** — generate per-diagnosis physician-style rationale,
   optionally grounded with PubMed references.
5. **ActionAgent** — propose diagnostic / clinical actions and an overall
   management plan.

The final SSE `result` event matches the schema OpenDX persists in its database.

## API

### `POST /chat`

Request:

```json
{ "messages": [{"role": "user", "content": "<clinical case description>"}] }
```

Response: `text/event-stream`, with events of the form `data: <json>\n\n`.

Event types:

- `{"type": "progress", "message": "<stage>"}`
- `{"type": "result",   "data":    {...payload...}}`
- `{"type": "error",    "message": "<reason>"}`

Result payload:

```jsonc
{
  "case_description":   "string",
  "predictions":        ["string", ...],
  "warning_diagnosis":  ["string", ...],
  "reasoning": [
    { "reasoning": "string", "references": ["string", ...] }
  ],
  "overall_reasoning":  "string",
  "management":         "string",
  "actions":            ["string", ...]
}
```

### `GET /health`

Returns `{"status": "ok"}`.

## Setup

```bash
cp dotenv.tpl .env
# fill in OPENAI_API_KEY (and PUBMED_* if ADD_REFERENCES=true)

uv sync
uv run aegisdx-server
```

Point OpenDX at this service:

```bash
# in opendx/api/.env
AGENT_SERVICE_URL=http://localhost:8000
```

## Configuration

| Var               | Default        | Purpose                                                    |
| ----------------- | -------------- | ---------------------------------------------------------- |
| `LLM_PROVIDER`    | `openai`       | LangChain provider name (`openai`, `anthropic`, ...).      |
| `LLM_MODEL`       | `gpt-4.1`      | Model for reasoning, warning, action.                       |
| `AGGREGATE_MODEL` | `gpt-4o-mini`  | Cheaper model for clustering predictions.                  |
| `NUM_INFERENCE`   | `10`           | Number of candidate diagnoses to sample.                   |
| `ADD_REFERENCES`  | `true`         | If true, ReasoningAgent retrieves PubMed references.       |
| `PUBMED_EMAIL`    | —              | Required by Entrez when `ADD_REFERENCES=true`.             |
| `PUBMED_API_KEY`  | —              | Optional, raises Entrez rate limits.                       |
| `HOST` / `PORT`   | `0.0.0.0:8000` | Bind address.                                              |
