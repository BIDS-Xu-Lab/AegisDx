# AegisDx

Multi-agent clinical case diagnosis with LangGraph. Frontend hosts a case-entry
UI; backend orchestrates a diagnosis workflow (initial differentials → warning
diagnoses → clustering → verification → per-diagnosis reasoning + PubMed
citations → overall reasoning → management plan) and streams progress + final
result over SSE.

## Layout

```
api/       FastAPI server (server.py) — /api/chat streams DiagnosisWorkflow
web/       Vue 3 + Vite SPA (deploys to GitHub Pages)
engine/    DiagnosisWorkflow (LangGraph) + agents/utils/retrievers
conf/      Docker + nginx for self-hosted prod
legacy/    Earlier standalone agent-service prototype (kept for reference)
```

The API imports the engine via a relative `sys.path` (see `api/server.py`), so
`api/` and `engine/` must ship together (both live in the same image).

## Local dev

```bash
# backend
cd api && cp dotenv.tpl .env    # fill in OPENAI_API_KEY, PUBMED_*, etc.
uv sync && uv run server.py     # binds 0.0.0.0:9627

# frontend (separate terminal)
cd web && cp dotenv.tpl .env
npm install && npx vite         # binds :8968, proxies /api → :9627
```

## Env vars

| Var | Purpose |
|---|---|
| `OPENAI_API_KEY` | Any OpenAI-compatible endpoint's key (incl. Azure Foundry) |
| `OPENAI_BASE_URL` | Point at Azure Foundry / self-hosted gateway; omit for api.openai.com |
| `AEGISDX_LLM_PROVIDER` | `openai` / `azure_openai` / `anthropic` / ... (default `openai`) |
| `AEGISDX_BASE_MODEL` | Primary model / Azure deployment name (default `gpt-4.1`) |
| `AEGISDX_FAST_MODEL` | Small model for aggregation/parsing (default `gpt-4o-mini`) |
| `AEGISDX_NUM_INFERENCE` | Initial diagnoses to sample (default 10) |
| `AEGISDX_ADD_REFERENCES` | Enable PubMed retrieval + citations (default `false`) |
| `PUBMED_EMAIL`, `PUBMED_API_KEY` | NCBI E-utilities creds (needed if ADD_REFERENCES=true) |
| `DATABASE_URL` | Postgres URL (Supabase) |
| `SUPABASE_JWT_SECRET` | HS256 secret for authenticated routes |

## Deploy

**Frontend (GitHub Pages)** — `.github/workflows/pages.yml` builds `web/` and
publishes to Pages on every push to `main`. To wire up:

1. Repo Settings → **Pages** → Source: *GitHub Actions*
2. Repo Settings → Secrets and variables → **Actions**
   - Variable: `VITE_API_URL` = public backend URL (e.g. `https://aegisdx-api.onrender.com`)
   - Secrets: `VITE_SUPABASE_URL`, `VITE_SUPABASE_PUBLISHABLE_KEY`

Site lands at `https://bids-xu-lab.github.io/AegisDx/`.

**Backend (Render)** — `render.yaml` is a Blueprint.

1. render.com → **New** → **Blueprint** → connect this repo
2. Fill in the secrets flagged `sync: false` (OpenAI key, PubMed creds, DB URL, JWT secret)
3. Deploy; the service URL is what `VITE_API_URL` should point at.

Fly.io or any Docker-capable host works too — the Dockerfile is at
`conf/prod/dockerfile.api` and builds from the repo root.

**Self-hosted** — `docker compose -f conf/prod/docker-compose.yml up --build`.
