# gateway/main.py
import os
import re
import json
import uuid
import asyncio
import chromadb
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from cache import get_cached, set_cache

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
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080/completion")
# ─────────────────────────────────────────────────────────────

app = FastAPI()

print("[startup] Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

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


async def retrieve_context(embedding: list[float]) -> str:
    results = await asyncio.to_thread(
        collection.query,
        query_embeddings=[embedding],
        n_results=TOP_K_CHUNKS,
    )
    return "\n\n".join(results["documents"][0])


def build_prompt(query: str, context: str) -> str:
    return f"""<|user|>
You are a helpful assistant for {COLLEGE_NAME}.
Answer ONLY using the context below.
If the answer is not in the context, say "I don't have that information."

Context:
{context}

Question: {query}
<|end|>
<|assistant|>
"""


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

    # 1. Embed query (thread pool — CPU-bound, must not block event loop)
    raw = await asyncio.to_thread(embedder.encode, [req.query])
    embedding = raw.tolist()[0]

    # 2. Semantic cache check
    cached = await get_cached(embedding)
    if cached:
        print(f"[cache HIT] '{req.query}'")
        return StreamingResponse(
            stream_from_cache(cached),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    print(f"[cache MISS] '{req.query}' — queuing for inference")

    # 3. RAG retrieval + prompt
    context = await retrieve_context(embedding)
    prompt = build_prompt(req.query, context)

    # 4. Subscribe BEFORE enqueuing — Redis pub/sub has no backlog, so a worker
    #    that publishes before we've subscribed would lose those tokens.
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel)
    job = {"session_id": session_id, "prompt": prompt}
    await redis_client.lpush(QUEUE_KEY, json.dumps(job))

    # 5. Stream + collect + cache
    async def stream_collect_cache():
        full_response = []
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
                    await set_cache(embedding, "".join(full_response))
                    print(
                        f"[cache SET] '{req.query}' — {len(full_response)} tokens cached"
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
