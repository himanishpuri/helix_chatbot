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
CHROMA_PATH  = os.path.join(os.path.dirname(__file__), "../data/chroma_db")
COLLECTION   = "college_kb"
QUEUE_KEY    = "inference_queue"
COLLEGE_NAME = "ABC Institute of Technology"  # change this
TOP_K_CHUNKS = 3
# ─────────────────────────────────────────────────────────────

app = FastAPI()

print("[startup] Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("[startup] Connecting to ChromaDB...")
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_collection(COLLECTION)

print("[startup] Connecting to Redis...")
redis_client = aioredis.Redis(host="localhost", port=6379, decode_responses=True)

print("[startup] Ready.")


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"


def retrieve_context(query: str) -> str:
    embedding = embedder.encode([query]).tolist()
    results   = collection.query(query_embeddings=embedding, n_results=TOP_K_CHUNKS)
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


async def stream_from_queue(session_id: str, embedding: list, prompt: str,
                            request: Request, job_id: str):
    """Push job to queue, subscribe to response channel, stream tokens."""
    channel = f"response:{session_id}"
    full_response = []

    # Push job to queue
    job = {"session_id": session_id, "prompt": prompt, "job_id": job_id}
    await redis_client.lpush(QUEUE_KEY, json.dumps(job))

    # Subscribe to response channel
    pubsub = redis_client.pubsub()
    await pubsub.subscribe(channel)

    try:
        async for message in pubsub.listen():
            if await request.is_disconnected():
                await redis_client.set(f"cancelled:{job_id}", "1", ex=60)
                break

            if message["type"] != "message":
                continue

            data = message["data"]

            if data == "[DONE]":
                yield "data: [DONE]\n\n"
                # Cache the full response
                await set_cache(redis_client, embedding, "".join(full_response))
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


@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    # 1. Embed query
    embedding = embedder.encode([req.query]).tolist()[0]

    # 2. Semantic cache check
    cached = await get_cached(redis_client, embedding)
    if cached:
        print(f"[cache HIT] '{req.query}'")
        return StreamingResponse(
            stream_from_cache(cached),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
        )

    print(f"[cache MISS] '{req.query}' — queuing for inference")

    # 3. RAG retrieval + prompt
    context = retrieve_context(req.query)
    prompt  = build_prompt(req.query, context)

    # 4. Queue + stream response
    job_id = str(uuid.uuid4())
    return StreamingResponse(
        stream_from_queue(req.session_id, embedding, prompt, request, job_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    )


@app.get("/health")
async def health():
    return {"status": "ok"}
