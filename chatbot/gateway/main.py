# gateway/main.py
import os
import json
import uuid
import asyncio
import chromadb
import redis.asyncio as aioredis
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from cache import get_cached, set_cache

# ── Config ────────────────────────────────────────────────────
CHROMA_PATH = os.getenv(
    "CHROMA_PATH", os.path.join(os.path.dirname(__file__), "../data/chroma_db")
)
COLLECTION = "college_kb"
QUEUE_KEY = "inference_queue"
COLLEGE_NAME = "ABC Institute of Technology"  # change this
TOP_K_CHUNKS = 3
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
LLAMA_URL = os.getenv("LLAMA_URL", "http://localhost:8080/completion")
# ─────────────────────────────────────────────────────────────

app = FastAPI()

print("[startup] Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("[startup] Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION)

print("[startup] Connecting to Redis...")
redis_client = aioredis.Redis(host=REDIS_HOST, port=6379, decode_responses=True)

print("[startup] Ready.")


class ChatRequest(BaseModel):
    query: str
    session_id: str = ""  # auto-generate if empty


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
    words = cached_response.split(" ")
    for word in words:
        yield f"data: {json.dumps({'token': word + ' '})}\n\n"
        await asyncio.sleep(0.01)
    yield "data: [DONE]\n\n"


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    # Auto-generate unique session_id to prevent channel collisions
    session_id = req.session_id if req.session_id else str(uuid.uuid4())

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

    # 4. Push job to queue
    channel = f"response:{session_id}"
    job = {"session_id": session_id, "prompt": prompt}
    await redis_client.lpush(QUEUE_KEY, json.dumps(job))

    # 5. Stream + collect + cache
    async def stream_collect_cache():
        full_response = []
        pubsub = redis_client.pubsub()
        await pubsub.subscribe(channel)
        try:
            async for message in pubsub.listen():
                if await request.is_disconnected():
                    print(f"[disconnect] client left session={session_id}")
                    break
                if message["type"] != "message":
                    continue
                data = message["data"]
                if data == "[DONE]":
                    # Cache BEFORE closing — this was the bug
                    await set_cache(embedding, "".join(full_response))
                    print(
                        f"[cache SET] '{req.query}' — {len(full_response)} tokens cached"
                    )
                    yield "data: [DONE]\n\n"
                    break
                try:
                    parsed = json.loads(data)
                    if "error" in parsed:
                        yield f"data: {json.dumps({'token': 'Error processing request.'})}\n\n"
                        yield "data: [DONE]\n\n"
                        break
                    token = parsed.get("token", "")
                    full_response.append(token)
                    yield f"data: {json.dumps({'token': token})}\n\n"
                except json.JSONDecodeError:
                    continue
        finally:
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
