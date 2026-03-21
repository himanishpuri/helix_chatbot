# gateway/cache.py
import json
import uuid
import numpy as np
import redis.asyncio as aioredis

# ── Config ────────────────────────────────────────────────────
CACHE_TTL            = 7 * 24 * 60 * 60  # 7 days in seconds
SIMILARITY_THRESHOLD = 0.92
# ─────────────────────────────────────────────────────────────


def _vectorized_cosine(M: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Cosine similarity between matrix M (N×D) and vector q (D,)."""
    q_norm = np.linalg.norm(q)
    row_norms = np.linalg.norm(M, axis=1)
    denom = np.where(row_norms * q_norm == 0, 1e-10, row_norms * q_norm)
    return (M @ q) / denom


async def get_cached(redis_client: aioredis.Redis, query_embedding: list[float]) -> str | None:
    """Search all cached embeddings for a near match. Returns response text or None."""
    keys = []
    async for key in redis_client.scan_iter("cache:*"):
        keys.append(key)

    if not keys:
        return None

    # Fetch all entries in a single pipeline round trip
    pipe = redis_client.pipeline()
    for key in keys:
        pipe.hgetall(key)
    entries = await pipe.execute()

    # Build matrix of valid embeddings
    valid_keys = []
    embeddings = []
    responses = []
    for key, entry in zip(keys, entries):
        if not entry:
            continue
        try:
            emb = json.loads(entry["embedding"])
            valid_keys.append(key)
            embeddings.append(emb)
            responses.append(entry["response"])
        except (KeyError, json.JSONDecodeError):
            continue

    if not embeddings:
        return None

    M = np.array(embeddings)
    q = np.array(query_embedding)
    similarities = _vectorized_cosine(M, q)
    best_idx = int(np.argmax(similarities))

    if similarities[best_idx] >= SIMILARITY_THRESHOLD:
        await redis_client.expire(valid_keys[best_idx], CACHE_TTL)
        return responses[best_idx]

    return None


async def set_cache(redis_client: aioredis.Redis, query_embedding: list[float], response_text: str) -> None:
    """Store embedding + response in Redis."""
    key = f"cache:{uuid.uuid4()}"
    pipe = redis_client.pipeline()
    pipe.hset(key, mapping={
        "embedding": json.dumps(query_embedding),
        "response":  response_text
    })
    pipe.expire(key, CACHE_TTL)
    await pipe.execute()
