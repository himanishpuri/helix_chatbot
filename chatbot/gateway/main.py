# gateway/main.py
import os
import re
import json
import uuid
import asyncio
import chromadb
import httpx
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from cache import get_cached, set_cache
from templates import get_preset, SYSTEM_INSTRUCTION, REWRITE_INSTRUCTION

# ── Config ────────────────────────────────────────────────────
CHROMA_PATH = os.getenv(
    "CHROMA_PATH", os.path.join(os.path.dirname(__file__), "../data/chroma_db")
)
COLLECTION = "college_kb"
QUEUE_KEY = "inference_queue"
COLLEGE_NAME = os.getenv("COLLEGE_NAME", "ABC Institute of Technology")
TOP_K_CHUNKS = 3
MAX_QUERY_LEN = 2000
CANCEL_TTL = 60  # seconds a cancel flag lives
CONV_TTL = int(os.getenv("CONV_TTL", "3600"))  # conversation memory lifetime
MAX_HISTORY_MSGS = int(os.getenv("MAX_HISTORY_MSGS", "12"))  # kept per conversation
REWRITE_MAX_TOKENS = int(os.getenv("REWRITE_MAX_TOKENS", "64"))
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080/completion")
MODEL_PRESET = os.getenv("MODEL_PRESET", "phi3")
EMBED_MODEL = os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
# bge models want a short instruction prefix on the QUERY (not the docs) for
# best retrieval. Auto-apply for bge; override or disable via QUERY_PREFIX.
QUERY_PREFIX = os.getenv(
    "QUERY_PREFIX",
    "Represent this sentence for searching relevant passages: "
    if "bge" in EMBED_MODEL.lower()
    else "",
)
# ─────────────────────────────────────────────────────────────

app = FastAPI()

PRESET = get_preset(MODEL_PRESET)

print(f"[startup] Loading embedding model {EMBED_MODEL} ...")
embedder = SentenceTransformer(EMBED_MODEL)

print("[startup] Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
# get_or_create so the gateway boots even before ingestion has run.
collection = chroma_client.get_or_create_collection(
    COLLECTION, metadata={"hnsw:space": "cosine"}
)

print("[startup] Connecting to Redis...")
redis_client = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

print("[startup] Ready.")


class ChatRequest(BaseModel):
    query: str = Field(min_length=1, max_length=MAX_QUERY_LEN)
    # Opaque client-supplied memory key. Empty = stateless single turn.
    # NOTE: never used as the pub/sub channel name (that stays server-side).
    conversation_id: str = Field(default="", max_length=128)


async def embed(text: str) -> list[float]:
    raw = await asyncio.to_thread(embedder.encode, [QUERY_PREFIX + text])
    return raw.tolist()[0]


async def load_history(conversation_id: str) -> list[dict]:
    if not conversation_id:
        return []
    raw = await redis_client.lrange(f"conv:{conversation_id}", 0, -1)
    return [json.loads(r) for r in raw]


async def append_history(conversation_id: str, user_msg: str, assistant_msg: str):
    if not conversation_id:
        return
    key = f"conv:{conversation_id}"
    async with redis_client.pipeline(transaction=True) as pipe:
        pipe.rpush(
            key,
            json.dumps({"role": "user", "content": user_msg}),
            json.dumps({"role": "assistant", "content": assistant_msg}),
        )
        pipe.ltrim(key, -MAX_HISTORY_MSGS, -1)
        pipe.expire(key, CONV_TTL)
        await pipe.execute()


async def rewrite_standalone(history: list[dict], query: str) -> str:
    """Condense history + follow-up into a standalone question for retrieval.

    First turn (no history) returns the query unchanged — no model call.
    """
    if not history:
        return query
    convo = "\n".join(f"{m['role']}: {m['content']}" for m in history)
    messages = [
        {"role": "system", "content": REWRITE_INSTRUCTION},
        {"role": "user", "content": f"Conversation:\n{convo}\n\nFollow-up: {query}"},
    ]
    prompt = PRESET["render"](messages)
    payload = {
        "prompt": prompt,
        "n_predict": REWRITE_MAX_TOKENS,
        "temperature": 0.0,
        "stream": False,
        "stop": PRESET["stop"],
    }
    try:
        async with httpx.AsyncClient(timeout=30) as client:
            r = await client.post(LLAMA_URL, json=payload)
            text = (r.json().get("content") or "").strip()
        return text or query
    except Exception as e:
        print(f"[rewrite] failed ({e}); using raw query")
        return query


async def retrieve_context(embedding: list[float]) -> str:
    results = await asyncio.to_thread(
        collection.query,
        query_embeddings=[embedding],
        n_results=TOP_K_CHUNKS,
    )
    # documents can be [] (empty collection) or contain None entries — filter to
    # real strings so an unpopulated/partial store degrades to "no context"
    # instead of crashing the whole request.
    docs = (results.get("documents") or [[]])[0] or []
    return "\n\n".join(d for d in docs if isinstance(d, str) and d)


def build_prompt(context: str, history: list[dict], query: str) -> str:
    system = {
        "role": "system",
        "content": SYSTEM_INSTRUCTION.format(college=COLLEGE_NAME, context=context),
    }
    return PRESET["render"]([system, *history, {"role": "user", "content": query}])


async def stream_from_cache(cached_response: str):
    # Split on whitespace boundaries but KEEP the whitespace, so newlines and
    # code blocks in a cached answer render identically to the live stream.
    for token in re.findall(r"\S+\s*|\s+", cached_response):
        yield f"data: {json.dumps({'token': token})}\n\n"
        await asyncio.sleep(0.01)
    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    # Server-generated session id: it names the pub/sub channel, so it must NOT
    # be client-controllable or one client could subscribe to another's stream.
    session_id = str(uuid.uuid4())
    channel = f"response:{session_id}"
    cancel_key = f"cancel:{session_id}"
    cid = req.conversation_id

    # 1. Load memory + resolve the follow-up into a standalone question so
    #    retrieval and caching key off the real intent, not a bare pronoun.
    history = await load_history(cid)
    standalone = await rewrite_standalone(history, req.query)
    if standalone != req.query:
        print(f"[rewrite] '{req.query}' -> '{standalone}'")
    embedding = await embed(standalone)

    # 2. Semantic cache check (keyed on the resolved question)
    cached = await get_cached(embedding)
    if cached:
        print(f"[cache HIT] '{standalone}'")
        await append_history(cid, req.query, cached)
        return StreamingResponse(
            stream_from_cache(cached),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    print(f"[cache MISS] '{standalone}' — queuing for inference")

    # 3. RAG retrieval + prompt (history included so pronouns resolve at gen too)
    context = await retrieve_context(embedding)
    prompt = build_prompt(context, history, req.query)

    # 4. Subscribe BEFORE enqueuing — Redis pub/sub has no backlog, so a worker
    #    that publishes before we've subscribed would lose those tokens.
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel)
    job = {"session_id": session_id, "prompt": prompt}
    await redis_client.lpush(QUEUE_KEY, json.dumps(job))

    # 5. Stream + collect + cache + remember
    async def stream_collect_cache():
        full_response = []
        completed = False
        try:
            while True:
                # Poll with a timeout so a client disconnect is noticed even if
                # the model stalls and no tokens are arriving.
                message = await pubsub.get_message(
                    ignore_subscribe_messages=True, timeout=1.0
                )
                if await request.is_disconnected():
                    print(f"[disconnect] client left session={session_id}")
                    await redis_client.set(cancel_key, "1", ex=CANCEL_TTL)
                    break
                if message is None:
                    continue

                data = message["data"]
                if data == "[DONE]":
                    answer = "".join(full_response)
                    await set_cache(embedding, answer)
                    await append_history(cid, req.query, answer)
                    completed = True
                    print(
                        f"[cache SET] '{standalone}' — {len(full_response)} tokens cached"
                    )
                    yield "data: [DONE]\n\n"
                    break
                try:
                    parsed = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if "error" in parsed:
                    yield f"data: {json.dumps({'token': 'Error processing request.'})}\n\n"
                    yield "data: [DONE]\n\n"
                    break
                token = parsed.get("token", "")
                full_response.append(token)
                yield f"data: {json.dumps({'token': token})}\n\n"
        finally:
            # Ensure a still-running worker stops if we exit for any reason.
            if not completed:
                await redis_client.set(cancel_key, "1", ex=CANCEL_TTL)
            await pubsub.unsubscribe(channel)
            await pubsub.aclose()

    return StreamingResponse(
        stream_collect_cache(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
