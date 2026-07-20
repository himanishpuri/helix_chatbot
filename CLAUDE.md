# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Helix Chatbot — a local-only RAG chatbot for a college knowledge base. FastAPI gateway + async worker pool + llama.cpp (host) + ChromaDB (retrieval) + Redis (job queue, pub/sub streaming, conversation memory, semantic cache), fronted by Nginx. No external API calls; everything runs on local hardware. Full architecture diagrams and design rationale live in `README.md` — read it before making non-trivial changes, don't re-derive what's already documented there.

All application code lives under `chatbot/`: `gateway/` (FastAPI app + workers), `ingestion/` (KB chunk/embed/load script), `nginx/`, `frontend/` (single static HTML file, vanilla JS, no build step).

## Commands

Gateway and ingestion are separate `uv` projects (separate `pyproject.toml`/`uv.lock`) — `cd` into the right one before running `uv`.

```bash
# Run tests (pure-logic, no services needed — this is what CI runs)
cd chatbot/gateway && uv run python test_cache.py       # semantic cache: best-match + cosine guard
cd chatbot/gateway && uv run python test_templates.py   # model presets: render + stop tokens
cd chatbot/ingestion && uv run python ingest.py --selfcheck  # chunking: overlap + edge cases

# Format (dev dependency in gateway only)
cd chatbot/gateway && uv run black .

# Full stack
cd chatbot && docker-compose up            # redis, gateway, worker, nginx
cd chatbot && docker compose --profile ingest run --rm ingest  # one-shot ingestion in a container

# Run without Docker (gateway uses flat imports — `from cache import ...` — so it
# MUST be run with cwd = gateway/, not the repo root)
cd chatbot/gateway && uv run uvicorn main:app --port 8000
cd chatbot/gateway && uv run python worker.py

# Ingestion on the host (writes to chatbot/data/chroma_db/, mounted into gateway+worker containers)
cd chatbot/ingestion && uv sync && uv run python ingest.py

# Benchmark the semantic cache (vector KNN vs. linear-scan fallback)
cd chatbot/gateway && uv run python bench_cache.py
```

There is no single top-level test runner — CI (`.github/workflows/ci.yml`) invokes the three self-check scripts above directly with just `numpy` + `redis` installed (no torch/chromadb), so they must keep working without those heavier deps. The streaming/RAG end-to-end path needs a real Redis + GGUF model and is verified manually, not in CI.

## Architecture

Request flow (see `README.md` for the full sequence diagram): Browser → Nginx (rate limit 20r/m) → Gateway `/chat` → embed query → semantic cache lookup (Redis) → on miss: retrieve top-3 chunks from ChromaDB → build prompt → push job onto Redis list `inference_queue` → one of 8 async workers (`worker.py`) BRPOPs it, streams from llama.cpp, publishes tokens to a Redis pub/sub channel `response:{session_id}` → Gateway forwards as SSE to the browser → on `[DONE]`, gateway caches the full answer and appends the turn to conversation history.

Key cross-file invariants to preserve when touching this path:

- **Subscribe before enqueue** (`gateway/main.py`): the gateway subscribes to the pub/sub channel *before* pushing the job, because Redis pub/sub has no backlog — reversing the order silently drops leading tokens.
- **Session id is server-generated**, never client-supplied — it names the pub/sub channel, so a client-controlled id would let one client subscribe to another's stream.
- **Cancellation**: on client disconnect the gateway sets `cancel:{session_id}` in Redis; `worker.py` checks it between tokens (every `CANCEL_CHECK_EVERY` tokens) and aborts the llama.cpp stream.
- **Query rewrite**: follow-up turns are condensed into a standalone question via a short, deterministic (temperature=0) llama.cpp call (`rewrite_standalone` in `main.py`) before embedding — this is what makes retrieval and the semantic cache key off intent rather than a bare pronoun. Skipped entirely when there's no history (no model call on the first turn).
- **`EMBED_MODEL` must match between gateway and ingestion** — changing it changes vector dimensions, so ingestion must be re-run. `MODEL_PRESET` (chat template + stop tokens, `gateway/templates.py`) is independent and hot-swappable via env var alone.
- **Semantic cache** (`gateway/cache.py`): Redis vector `KNN 1` search (HNSW/cosine) when the Redis Query Engine is available (bundled in `redis:8`), else a linear cosine-scan fallback — the gateway boots against either backend. Threshold `SIMILARITY_THRESHOLD = 0.92`.
- Tunable constants live at the top of their owning file, not in a central config — see the table in `README.md` (`gateway/main.py`, `gateway/cache.py`, `gateway/worker.py`, `ingestion/ingest.py`) before adding new env vars.

## Docs

- `README.md` — architecture diagrams, design decisions and their rationale, full config reference, model-choice guidance per hardware.
- `docs/INTERVIEW.md` — interview-prep notes on this project's design tradeoffs.
