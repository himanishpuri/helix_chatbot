# Helix Chatbot — Interview Prep

Everything here is traced to source (`file:line`). Numbers are either constants in the
code or measured by a repro (`chatbot/gateway/bench_cache.py`). Where something is _not_
implemented or _not_ measured, it says so — don't claim it in the room.

Line refs are against the tree at the tip of `main` when this doc was written; if code
moved, the symbol name still finds it.

---

## One-line pitch

A local-only RAG chatbot: a stateless FastAPI SSE gateway and a pool of async workers that
never call each other — they talk **only through Redis** (a job list + pub/sub). ChromaDB is
the vector store, llama.cpp on the host does inference, and a Redis vector index serves a
semantic cache. Application code is **754 lines** of Python across 5 files (`main.py`,
`worker.py`, `cache.py`, `templates.py`, `ingest.py`; measured by `wc -l`), plus tests and a
benchmark.

**Core intuition:** decouple the web tier from inference with a queue + pub/sub so a slow
model never ties up an HTTP connection, and put a _semantic_ (embedding-similarity) cache in
front so paraphrases of an already-answered question skip the model entirely.

---

## Architecture, stage by stage

Two processes, no direct calls between them — Redis is the only channel
(`gateway/main.py`, `gateway/worker.py`).

### 1. Request in — `POST /chat` (`main.py:159`)

Body is validated by Pydantic: `query` (1–2000 chars) + optional `conversation_id`
(`main.py:65-69`, `MAX_QUERY_LEN=2000` at `main.py:25`).

`session_id = uuid4()` is generated **server-side** (`main.py:163`) because it names the
pub/sub channel `response:{session_id}`. If a client could supply it, one client could
subscribe to another's token stream — so it's never client-controllable (`main.py:161-162`).
This is a real, deliberate security boundary, not a formality.

Two distinct ids, don't conflate them:

- `session_id` — server-generated, names the pub/sub channel, one per request.
- `conversation_id` — opaque, client-supplied, the _memory_ key. Empty ⇒ stateless
  (`main.py:66-69`, `load_history` early-returns on empty at `main.py:78`).

### 2. Memory + query rewrite (`main.py:170-174`)

- Load prior turns from Redis list `conv:{conversation_id}` (`load_history`, `main.py:77-81`).
- **Rewrite-before-retrieve** (`rewrite_standalone`, `main.py:99-126`): on a follow-up, one
  _non-streaming_ llama.cpp call condenses `history + follow-up` into a standalone question
  (temperature 0.0, `n_predict=REWRITE_MAX_TOKENS=64`, `main.py:113-118`). First turn (empty
  history) returns the query unchanged with **no** model call (`main.py:104-105`). On any
  exception it falls back to the raw query (`main.py:124-126`) — the rewrite can never break a
  request.
- **Why rewrite instead of just prepending history to the retrieval query?** Retrieval then
  embeds a focused question, not a blob of prior turns, and the cache keys on _intent_:
  "when is it?" and "when is HELIX?" resolve to the same standalone question and hit the same
  cache entry. The instruction is at `templates.py:23-28`.

### 3. Embed (`embed`, `main.py:72-74`)

`SentenceTransformer(EMBED_MODEL)` (`main.py:50`), default `BAAI/bge-small-en-v1.5`
(`main.py:34`), run in a thread (`asyncio.to_thread`) so the CPU-bound encode doesn't block
the event loop. bge wants a query-side instruction prefix — auto-applied only when the model
name contains "bge" (`QUERY_PREFIX`, `main.py:37-42`) and **only to the query, never to stored
docs** (ingestion embeds raw chunks, `ingest.py:78`). That asymmetry is intentional and is how
bge is trained.

### 4. Semantic cache check (`get_cached`, `main.py:177`; `cache.py:176-180`)

Embedding → `KNN 1` over a Redis vector index. Hit (similarity ≥ threshold) ⇒ stream the
cached answer and append the turn to history (`main.py:178-185`); the follow-up still gets
remembered even on a cache hit. Details in "The semantic cache" below.

### 5. RAG retrieval (`retrieve_context`, `main.py:129-139`)

ChromaDB `collection.query(query_embeddings=[embedding], n_results=TOP_K_CHUNKS=3)`
(`main.py:130-134`, `TOP_K_CHUNKS=3` at `main.py:24`). Collection is created with
`hnsw:space: cosine` (`main.py:55-57`, mirrored in `ingest.py:71-73`) — so **ChromaDB
retrieval is already ANN (HNSW)**; there is nothing to "upgrade to ANN" on that path.

Defensive detail worth pointing at: the result documents can be `[]` (empty store) or contain
`None`, so it filters to real strings (`main.py:138-139`). This is a fixed bug — see the
"bug I fixed" story.

### 6. Prompt build (`build_prompt`, `main.py:142-147`)

`render([system + context] + history + [user query])`. History is fed to the _answer_ prompt
too, so pronouns resolve at generation, not only at retrieval (`main.py:189-191`). Rendering
goes through the model-family preset (see "Model presets").

### 7. Subscribe → enqueue → stream (`main.py:193-251`)

**Subscribe to `response:{session_id}` BEFORE `LPUSH`** (`main.py:195-198`). Redis pub/sub has
no backlog; a worker that publishes before we've subscribed drops those leading tokens. This
ordering is load-bearing — reversing it is a classic lost-first-token bug.

Then `LPUSH` a `{session_id, prompt}` job onto `inference_queue` (`main.py:197-198`) and start
an async generator that reads pub/sub messages with a 1s timeout, forwards each `{"token":...}`
as SSE, and on the literal `[DONE]` sentinel: caches the full answer + appends history
(`main.py:201-245`).

### 8. Worker (`worker.py`)

`NUM_WORKERS=8` (`worker.py:13`) async tasks, each `BRPOP`ing `inference_queue`
(`worker.py:82`, `main`+`worker` at `worker.py:77-96`). `process_job` (`worker.py:20-74`)
streams from llama.cpp `/completion` and publishes each token to the channel.

The terminal-signal contract is the subtle part (`worker.py:40-74`):

- llama.cpp `/completion` does **not** emit OpenAI's `data: [DONE]`. It ends after a JSON
  object with `"stop": true` (`worker.py:68`). The `[DONE]`-string check at `worker.py:51` is
  only for OpenAI-compat servers.
- The worker publishes a literal `[DONE]` in a **`finally`** (`worker.py:72-74`) — success,
  `stop:true`, error, cancel, and stream-end all converge there. That's what lets the gateway
  generator terminate deterministically instead of hanging.

### 9. Cancellation (`main.py:211-213,240-243`; `worker.py:41,63-67`)

On client disconnect (or any generator exit), the gateway sets `cancel:{session_id}` in Redis
with `CANCEL_TTL=60` (`main.py:26`). The worker checks that flag **before** starting
(`worker.py:41`) and every `CANCEL_CHECK_EVERY=8` tokens (`worker.py:14,63-67`); on cancel it
returns, which closes the llama.cpp stream and frees the worker. So a client that leaves
reclaims a worker within ~8 tokens instead of generating into the void.

```
Browser ──POST /chat──▶ Gateway ──embed──▶ [cache KNN] ─hit─▶ replay
                           │  miss
                           ├─ retrieve top-3 (ChromaDB HNSW)
                           ├─ SUBSCRIBE response:{sid}   ← before enqueue
                           ├─ LPUSH inference_queue
                           ▼
                        Worker (×8) BRPOP ──stream POST──▶ llama.cpp
                           │  PUBLISH {token}… then [DONE] (finally)
                           ▼
                        Gateway SSE ──▶ Browser;  on [DONE]: set_cache + append_history
```

---

## The semantic cache (the piece with the most depth)

File: `gateway/cache.py`. Public API is two functions — `get_cached(embedding)` and
`set_cache(embedding, response)` (`cache.py:176-188`) — so `main.py` doesn't know or care which
backend is live.

### Two backends, one probe

At first use, `_ensure_index(dim)` (`cache.py:57-88`) does an idempotent
`FT.CREATE cache_idx ON HASH PREFIX cache: SCHEMA response TEXT embedding VECTOR HNSW … DIM
<dim> DISTANCE_METRIC COSINE` (`cache.py:64-74`). The dimension is **derived from the first
embedding's length** (`get_cached`/`set_cache` pass `len(query_embedding)`, `cache.py:178,185`),
so it survives an `EMBED_MODEL` swap (384 for bge-small, 768 for bge-base) — nothing hardcoded.

- `FT.CREATE` OK, or "Index already exists" ⇒ vector mode (`cache.py:75-80`).
- Any other `ResponseError` (plain Redis, no Query Engine) or connection error ⇒ set
  `_vector_ok=False` and fall back to the **O(N) scan** (`cache.py:81-88`). The `_vector_ok`
  flag is probed once and cached (`cache.py:59-61`).

So the gateway boots against _any_ Redis and simply uses whichever backend is present. The
Query Engine ships in `redis:8` (`docker-compose.yml:16`).

### KNN path (`cache.py:92-134`)

`FT.SEARCH cache_idx "*=>[KNN 1 @embedding $vec AS score]" … SORTBY score … DIALECT 2`
(`cache.py:94-102`). RediSearch returns cosine **distance**; convert to similarity as
`1 - distance` (`cache.py:114-115`), accept iff `≥ SIMILARITY_THRESHOLD` (`cache.py:116`),
refresh TTL, read `response` (`cache.py:118-121`). Vectors are stored as raw float32 bytes
(`_to_bytes`, `cache.py:53-54`; `_set_cache_knn`, `cache.py:124-134`) via a separate
non-decoding Redis client (`_get_redis_bytes`, `cache.py:36-42`).

### Scan fallback (`cache.py:138-172`)

`scan_iter("cache:*")`, deserialize each JSON embedding, cosine in Python
(`_cosine_similarity` with a zero-norm guard, `cache.py:45-50`), track best above threshold.
This is the original implementation, kept intact so the app degrades gracefully.

### Measured: KNN flat, scan linear

`gateway/bench_cache.py` populates N random 384-dim entries and times each backend (median of
50 lookups, local `redis:8`):

| N entries | KNN lookup | O(N) scan |
| --------- | ---------: | --------: |
| 100       |     0.6 ms |     40 ms |
| 500       |     0.6 ms |    203 ms |
| 2000      |     1.8 ms |    809 ms |

KNN is ~flat (HNSW is sub-linear, ~O(log N)); the scan is ~0.4 ms per stored entry — linear,
because it pulls every hash over the wire and cosines it in Python. **Caveat for honesty:** at
the shipped KB size (see below) the cache holds a handful of entries, where 40 ms vs 0.6 ms is
irrelevant. The KNN backend is a _scaling_ argument, not a current-latency win. Say that
plainly.

---

## Non-obvious design decisions (and the "why")

| Decision                                     | Where                             | Why                                                                                                          |
| -------------------------------------------- | --------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Queue + pub/sub, not gateway→llama directly  | `main.py:195-198`, `worker.py:82` | Web tier stays stateless & responsive; workers scale independently; a slow model never holds an HTTP worker. |
| Subscribe before enqueue                     | `main.py:195-198`                 | Pub/sub has no backlog → avoids losing leading tokens.                                                       |
| `[DONE]` in a `finally`                      | `worker.py:72-74`                 | llama.cpp never sends a sentinel; this guarantees the gateway generator terminates on every exit path.       |
| Server-generated `session_id`                | `main.py:163`                     | Channel-name isolation — clients can't subscribe to each other's streams.                                    |
| Rewrite follow-ups to standalone             | `main.py:99-126`                  | Focused retrieval + cache-on-intent (dedupes paraphrases).                                                   |
| Preset = render + stop together              | `templates.py:61-65`              | Chat template and stop tokens can't drift; family swap = one env var.                                        |
| bge prefix on query only                     | `main.py:37-42`, `ingest.py:78`   | Matches how bge is trained; lifts retrieval at zero cost.                                                    |
| dim derived, not hardcoded                   | `cache.py:178,185`                | Survives embedder swaps (384↔768) with no code change.                                                       |
| Cache backend probed once, graceful fallback | `cache.py:57-88`                  | Boots on any Redis; uses ANN when available.                                                                 |

---

## The bug I fixed / hard decisions

**Bug: gateway hung forever on a successful generation.** llama.cpp `/completion` doesn't send
OpenAI's `data: [DONE]`; it ends with a JSON object carrying `"stop": true`. The original code
waited for a `[DONE]` sentinel that never came, so a _successful_ stream never terminated. Fix:
the worker treats `stop:true` and stream-end as done and publishes a literal `[DONE]` in a
`finally` so success/error/cancel all converge (`worker.py:51,68,72-74`); the gateway keys off
that sentinel (`main.py:219`). (Commit `d77f2b0`, "Fix streaming/cache correctness".)

**Bug: one bad ChromaDB doc 500'd the whole request, which hung the UI.** `retrieve_context`
did `"\n\n".join(documents[0])`; a `None` entry raised `TypeError: sequence item 0: expected
str`, the request 500'd, and the frontend — which didn't check `res.ok` — sat spinning. Fix:
filter to real strings server-side (`main.py:138-139`) **and** show a server-error bubble
client-side (`frontend/index.html:145-148`). Two-sided fix; either alone leaves a bad UX.

**Hard decision: history-aware retrieval — rewrite vs. prepend.** Chose an LLM rewrite over
naive history-prepend (`main.py:99-126`) so retrieval and cache key on resolved intent. Cost:
one extra non-streaming model call per follow-up. Mitigations: first turn skips it entirely,
and it's deterministic (temp 0.0) and capped (`n_predict=64`).

---

## Deeper-probe Q&A

**Complexity / Big-O.**

- Cache lookup: KNN ~O(log N) over the HNSW graph (measured flat, table above); scan fallback
  O(N·d) — N entries × d-dim cosine (`cache.py:143-153`).
- RAG retrieval: ChromaDB HNSW, ~O(log N) over the chunk set (`main.py:55-57`).
- Chunking: single linear pass over the doc, O(chars) (`ingest.py:18-50`).

**Why these parameter values?**

- `SIMILARITY_THRESHOLD=0.92` (`cache.py:16`) — high enough that only genuine paraphrases hit
  (avoids serving a wrong cached answer), low enough to catch reworded-but-same questions.
- `TOP_K_CHUNKS=3` (`main.py:24`) — enough context for a small KB without burying the model.
- `CHUNK_SIZE=500`, `CHUNK_OVERLAP=100` (`ingest.py:13-14`) — paragraph-packed to ~500 chars
  with 100-char carry so an answer spanning a boundary isn't cut mid-idea (`ingest.py:40-45`).
- `NUM_WORKERS=8` (`worker.py:13`) — concurrency for parallel queries; these are async tasks in
  one process, bound by llama.cpp throughput, not CPU cores.
- `CANCEL_CHECK_EVERY=8` (`worker.py:14`) — check cadence; a leaving client is noticed within
  ~8 tokens. Tighter = more Redis `EXISTS` calls.
- `REWRITE_MAX_TOKENS=64` (`main.py:29`) — a standalone question is short; caps the rewrite cost.
- `MAX_HISTORY_MSGS=12` / `CONV_TTL=3600` (`main.py:28,27`) — ~6 turns, 1-hour memory window,
  trimmed with `LTRIM` (`main.py:94`).

**Failure modes / what breaks.**

- Redis down ⇒ gateway can't enqueue or read pub/sub; requests fail. Single point of
  coordination by design.
- llama.cpp down ⇒ worker's `httpx` raises, publishes `{"error":...}` then `[DONE]`
  (`worker.py:70-74`); gateway surfaces "Error processing request." (`main.py:233-236`).
- `EMBED_MODEL` mismatch between gateway and ingestion ⇒ dimension/space mismatch, retrieval
  silently wrong. Documented as a must-match invariant (`ingest.py:10`).
- Query Engine absent ⇒ silent, correct fallback to O(N) scan (`cache.py:81-88`).

**Concurrency.** Gateway is async FastAPI; CPU-bound embedding is offloaded with
`asyncio.to_thread` (`main.py:73,130`) so it doesn't block the loop. Workers are 8 async tasks
sharing one process and one Redis connection (`worker.py:17,95`). No locks — Redis is the
shared state; `BRPOP` hands each job to exactly one worker.

**Atomicity / rollback.** History append is a `MULTI` pipeline: `RPUSH` + `LTRIM` + `EXPIRE`
together (`main.py:88-96`). Cache writes are best-effort with a TTL — no rollback needed; a
stale/partial cache entry just expires (`cache.py:124-134,160-172`, `CACHE_TTL=604800`
= 7 days at `cache.py:15`). There are **no** cross-store transactions (Redis + ChromaDB are
independent); an honest limitation, not a bug at this scale.

**Testing strategy.** Logic that isn't obvious carries a runnable, service-free check:

- `gateway/test_cache.py` — best-match-wins, below-threshold, zero-norm cosine guard,
  distance→similarity, float32 packing; stubs Redis with a fake, exercises the scan path
  directly (`test_cache.py:42-95`).
- `gateway/test_templates.py` — every preset renders all roles, ends on the assistant turn, has
  a stop token present in the render, and unknown preset raises (`test_templates.py:12-42`).
- `ingestion/ingest.py --selfcheck` — chunking edge cases + overlap (`ingest.py:94-105`).
- `.github/workflows/ci.yml` runs all three on push/PR with only `numpy`+`redis` installed —
  no torch/chromadb, no live services. E2E (real Redis + GGUF) is verified manually, stated
  in CI comments (`ci.yml:16-26`).
- `gateway/bench_cache.py` — the KNN-vs-scan micro-benchmark (needs a live `redis:8`).

**Why this tech, not that?**

- **Redis** for both queue and pub/sub and cache — one dependency does job hand-off,
  fan-out streaming, and vector search (Query Engine in `redis:8`).
- **ChromaDB** — embedded persistent vector store, no server to run; HNSW cosine out of the box.
- **llama.cpp** — local GGUF inference, no API cost, data stays on the machine; the app is
  model-agnostic via presets.
- **SSE, not WebSockets** — one-way token stream; SSE is simpler and proxies cleanly through
  Nginx (`nginx.conf:19-33`).
- **FastAPI async** — SSE + concurrent requests without threads per connection.

**Honest limitations (say these before they ask).**

- Single gateway replica, **no auth**, no reranking, no hybrid (keyword+vector) search — out of
  scope by design.
- KB is tiny: `college_data.md` is ~3.8 KB / 65 paragraphs → **11 chunks** from the current
  `chunk_text` (measured). At that size the ANN cache is a scaling story, not a latency win.
- The committed ChromaDB store under `chatbot/data/chroma_db/` is **stale** (4 embeddings from
  an older KB, not the current 11) — re-run ingestion before a live demo.
- Nginx rate limit is per-IP; behind campus/office NAT many users share a limit
  (`nginx.conf:12-13`).
- No cross-store transaction between Redis and ChromaDB (see Atomicity).
- Perf beyond the cache-lookup micro-benchmark is not measured; miss latency is
  model/hardware-dependent and intentionally unquantified.

---

## Model presets (`templates.py`)

One dict entry per LLM family (`PRESETS`, `templates.py:61-65`), each carrying a `render`
function and the matching `stop` list so they can't drift. `main.py` renders the prompt
(`main.py:47,147`); `worker.py` sets llama.cpp's `stop` payload from the _same_ preset
(`worker.py:15,37`). Switch family with `MODEL_PRESET` (default `phi3`, `main.py:33`,
`worker.py:15`); unknown name raises early (`templates.py:68-74`). Adding a family = one dict
entry.

- `_phi3` folds system+user into the user turn (`templates.py:31-39`) — Phi-3's template has no
  separate system role.
- `_qwen` uses ChatML `<|im_start|>role … <|im_end|>` (`templates.py:42-47`).
- `_llama3` uses the Llama-3 header tokens (`templates.py:50-58`).

---

## File map (concern → file)

| Concern                                                         | File                                                          |
| --------------------------------------------------------------- | ------------------------------------------------------------- |
| HTTP API, SSE, memory, rewrite, cache-check, retrieval, prompt  | `chatbot/gateway/main.py`                                     |
| Async worker pool, llama.cpp streaming, cancellation, `[DONE]`  | `chatbot/gateway/worker.py`                                   |
| Semantic cache: vector KNN + O(N) scan fallback                 | `chatbot/gateway/cache.py`                                    |
| Model-family chat templates + stop tokens + instructions        | `chatbot/gateway/templates.py`                                |
| Chunking, embedding, ChromaDB load                              | `chatbot/ingestion/ingest.py`                                 |
| Knowledge base source                                           | `chatbot/ingestion/college_data.md`                           |
| Rate limiting, SSE proxy headers, static frontend               | `chatbot/nginx/nginx.conf`                                    |
| Streaming chat UI (vanilla JS)                                  | `chatbot/frontend/index.html`                                 |
| Orchestration (redis:8, gateway, worker, nginx, ingest profile) | `chatbot/docker-compose.yml`                                  |
| Multi-stage, non-root, CPU-torch image                          | `chatbot/gateway/Dockerfile`, `chatbot/ingestion/Dockerfile`  |
| Unit checks (no services)                                       | `test_cache.py`, `test_templates.py`, `ingest.py --selfcheck` |
| Cache lookup benchmark                                          | `chatbot/gateway/bench_cache.py`                              |
| CI (three self-checks)                                          | `.github/workflows/ci.yml`                                    |

---

## 30-second walkthrough script

"Browser POSTs `/chat`. Gateway loads conversation memory, rewrites a follow-up into a
standalone question, embeds it with bge-small, and checks a Redis vector-KNN semantic cache.
Miss ⇒ it retrieves top-3 chunks from ChromaDB (HNSW cosine), builds a family-specific prompt,
subscribes to a per-request pub/sub channel _before_ enqueuing the job — because pub/sub has no
backlog — then LPUSHes onto `inference_queue`. One of 8 async workers BRPOPs it, streams from
llama.cpp, publishes each token, and publishes `[DONE]` in a `finally`. The gateway forwards
tokens as SSE, and on `[DONE]` caches the answer and appends the turn. Client disconnect sets a
cancel flag the worker checks every 8 tokens. The two processes never call each other — Redis
is the only channel."
