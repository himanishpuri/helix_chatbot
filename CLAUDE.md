# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Helix Chatbot — a local-only RAG chatbot. FastAPI SSE gateway + an async worker pool talk
through Redis; ChromaDB is the vector store; llama.cpp (on the host) does inference. All code
lives under `chatbot/` (~360 LoC across 4 Python files).

## Two-process split

The gateway and the workers are **separate processes** that never call each other directly —
they communicate only through Redis:

- **`gateway/main.py`** — the web tier. Embeds the query, checks the semantic cache, does RAG
  retrieval, `LPUSH`es a job onto the `inference_queue` list, and streams tokens back to the
  browser over SSE. Stateless; serves `/chat` and `/health`.
- **`gateway/worker.py`** — `NUM_WORKERS` (8) async tasks in one process, each `BRPOP`ing the
  queue and streaming from llama.cpp.

### The streaming contract (don't break this)

1. Gateway **subscribes to `response:{session_id}` BEFORE `LPUSH`**. Redis pub/sub has no
   backlog — subscribing after enqueue drops leading tokens. Keep this order.
2. Worker publishes each token as `{"token": ...}` JSON to `response:{session_id}`, and a
   terminal literal `[DONE]` in a `finally` — success, error, cancel, or stream-end all
   converge there. llama.cpp `/completion` does NOT emit `data: [DONE]`; it ends after a JSON
   object with `"stop": true`. The worker treats `stop:true` (and stream-end) as done. Do not
   rely on an OpenAI-style `[DONE]` sentinel from the model.
3. `session_id` is **always server-generated** (`uuid4`) because it names the pub/sub channel.
   Never let a client supply it, or one client could subscribe to another's stream.

### Cancellation protocol

On client disconnect (or any generator exit), the gateway sets `cancel:{session_id}` in Redis
(TTL `CANCEL_TTL`). The worker checks `cancel:{session_id}` before starting and every
`CANCEL_CHECK_EVERY` tokens; on cancel it returns, which closes the llama.cpp stream and frees
the worker.

## Running / gotchas

- **Flat imports:** `main.py` does `from cache import ...`. Run the gateway from inside
  `gateway/` (`uv run uvicorn main:app`), not as `gateway.main` from `chatbot/`.
- **Ingest before serving:** run `ingestion/ingest.py` to populate ChromaDB. The gateway uses
  `get_or_create_collection`, so it boots even without ingestion, but retrieval returns nothing
  until you ingest.
- **`ingest.py` heavy imports are inside `main()`** so `chunk_text` (and `--selfcheck`) run
  without chromadb/sentence-transformers installed.
- **Env vars:** `REDIS_HOST`, `REDIS_PORT`, `LLAMA_URL`, `CHROMA_PATH`, `COLLEGE_NAME`. All
  three components read `REDIS_PORT` from env.

## Tests

- `gateway/test_cache.py` — best-match cache selection + cosine zero-norm guard, no Redis.
- `ingestion/ingest.py --selfcheck` — chunking edge cases + overlap.

Run with `uv run python <file>` (system Python lacks the deps).

## Out of scope by design

O(N) semantic-cache scan (fine for a small KB — no ANN index), single gateway replica, no
auth. Don't add these without a reason.
