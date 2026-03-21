# Helix Chatbot

A RAG (Retrieval-Augmented Generation) chatbot that runs entirely on local hardware. Queries a college knowledge base using ChromaDB for semantic search, caches responses in Redis to avoid redundant inference, and streams tokens back to the client in real time via Server-Sent Events.

---

## Features

- **Retrieval-Augmented Generation** — embeds each query with `all-MiniLM-L6-v2`, retrieves the top-3 most relevant chunks from ChromaDB, and injects them into the prompt so the model answers only from verified knowledge.
- **Semantic cache** — before hitting the inference stack, every query is compared against cached embeddings in Redis using vectorized cosine similarity. A match at ≥ 0.92 similarity returns the cached answer instantly with a 7-day TTL.
- **Local LLM inference** — delegates generation to a llama.cpp server running on the host. No external API calls, no data leaves the machine.
- **Async worker pool** — 8 async workers dequeue jobs from Redis and call llama.cpp in parallel, keeping latency low under concurrent load.
- **Real-time token streaming** — workers publish tokens to Redis pub/sub; the gateway forwards them to the browser as Server-Sent Events the moment they arrive.
- **Job cancellation on disconnect** — if the client closes the connection mid-stream, the gateway writes a cancellation flag to Redis and workers skip the job before calling llama.cpp, freeing inference capacity immediately.
- **Rate limiting via Nginx** — 5 requests/minute per IP with a burst of 3, enforced at the proxy layer before reaching the application.

---

## Architecture

```
Browser
  └─ Nginx (:80)
       ├─ rate limit: 5 req/min per IP
       └─ /api/chat → FastAPI Gateway (:8000)
                           │
             ┌─────────────┼──────────────┐
             ▼             ▼              ▼
        Redis cache    ChromaDB      Redis queue
        (semantic)    (vector DB)  → Worker pool (×8)
                                   → llama.cpp (:8080)
                                   → Redis pub/sub
                                   → SSE stream → Browser
```

### Request flow

1. Query arrives at the gateway and is embedded with `all-MiniLM-L6-v2`.
2. The gateway scans Redis for a cached embedding with cosine similarity ≥ 0.92. On a hit, the cached response streams back immediately.
3. On a miss, the top-3 chunks are fetched from ChromaDB collection `college_kb` and assembled into a RAG prompt.
4. A job `{session_id, prompt, job_id}` is enqueued on the `inference_queue` Redis list.
5. A worker dequeues the job, checks for a cancellation flag, then streams a POST to `llama.cpp /completion`.
6. Each token is published to the `response:{session_id}` Redis pub/sub channel.
7. The gateway forwards tokens as SSE events. When `[DONE]` arrives, the full response is written to the semantic cache.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Web framework | FastAPI + Uvicorn |
| Reverse proxy | Nginx (Alpine) |
| Embeddings | `sentence-transformers` — `all-MiniLM-L6-v2` |
| Vector store | ChromaDB (persistent, cosine metric) |
| Cache & queue | Redis 7 |
| LLM inference | llama.cpp (`llama-server`) |
| HTTP client | httpx (async) |
| Streaming | Server-Sent Events via `sse-starlette` |
| Containerisation | Docker Compose |

---

## Prerequisites

- Docker and Docker Compose
- A llama.cpp build with a compatible GGUF model
- Python 3.11+ (only needed to run the ingestion script outside Docker)

---

## Quick Start

### 1. Run the llama.cpp server (on the host, outside Docker)

```bash
llama.cpp/build/bin/llama-server -m /path/to/model.gguf --port 8080
```

The Docker containers reach this via `host.docker.internal:8080`.

### 2. Ingest the knowledge base

Run once to chunk `college_data.md`, embed it, and populate ChromaDB:

```bash
cd chatbot
python ingestion/ingest.py
```

This writes the vector database to `chatbot/data/chroma_db/`, which is mounted read-only into the gateway container.

### 3. Start all services

```bash
cd chatbot
docker-compose up
```

This starts four containers:

| Service | Port | Role |
|---|---|---|
| `redis` | 6379 (internal) | Job queue and semantic cache |
| `gateway` | 8000 | FastAPI app |
| `worker` | — | 8 async inference workers |
| `nginx` | 80 | Rate-limiting reverse proxy |

The API is available at `http://localhost/api/chat`.

### 4. Send a query

```bash
curl -N -X POST http://localhost/api/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What courses are offered?", "session_id": "user-1"}'
```

Tokens arrive as SSE events:

```
data: {"token": "The"}
data: {"token": " college"}
...
data: [DONE]
```

---

## Running Without Docker

```bash
# Terminal 1 — Redis
redis-server

# Terminal 2 — llama.cpp
llama.cpp/build/bin/llama-server -m /path/to/model.gguf --port 8080

# Terminal 3 — Gateway
cd chatbot
uvicorn gateway.main:app --port 8000

# Terminal 4 — Workers
cd chatbot
python gateway/worker.py
```

---

## Configuration

All tunable constants live at the top of their respective files:

| File | Constant | Default | Description |
|---|---|---|---|
| `gateway/main.py` | `TOP_K_CHUNKS` | `3` | Chunks retrieved from ChromaDB per query |
| `gateway/main.py` | `COLLEGE_NAME` | `ABC Institute of Technology` | Injected into the system prompt |
| `gateway/cache.py` | `SIMILARITY_THRESHOLD` | `0.92` | Cosine similarity required for a cache hit |
| `gateway/cache.py` | `CACHE_TTL` | `604800` | Cache entry lifetime (7 days, in seconds) |
| `gateway/worker.py` | `NUM_WORKERS` | `8` | Parallel inference workers |
| `gateway/worker.py` | `LLAMA_URL` | `http://localhost:8080/completion` | llama.cpp endpoint |
| `ingestion/ingest.py` | `CHUNK_SIZE` | `512` | Characters per chunk |
| `ingestion/ingest.py` | `CHUNK_OVERLAP` | `50` | Overlap between adjacent chunks |

**Docker environment variables** (set in `docker-compose.yml`):

| Variable | Value | Used by |
|---|---|---|
| `REDIS_HOST` | `redis` | gateway, worker |
| `LLAMA_URL` | `http://host.docker.internal:8080/completion` | gateway, worker |

---

## Knowledge Base

The knowledge base lives in `chatbot/ingestion/college_data.md`. Edit that file, then re-run the ingestion script to update ChromaDB:

```bash
cd chatbot
python ingestion/ingest.py
```

The ingestion script deletes and recreates the `college_kb` collection on each run, so re-running it is idempotent.

---

## Key Files

```
chatbot/
├── docker-compose.yml        — orchestrates Redis, Gateway, Worker, Nginx
├── gateway/
│   ├── main.py               — FastAPI app; /chat and /health endpoints
│   ├── cache.py              — semantic cache: scan_iter + pipeline + vectorized cosine
│   ├── worker.py             — dequeues jobs, calls llama.cpp, publishes tokens
│   └── Dockerfile
├── ingestion/
│   ├── ingest.py             — chunks college_data.md, embeds, stores in ChromaDB
│   └── college_data.md       — knowledge base source
├── nginx/
│   └── nginx.conf            — rate limiting, SSE proxy headers
├── frontend/
│   └── index.html            — frontend (not yet implemented)
└── data/
    └── chroma_db/            — persisted vector database (git-ignored)
```
