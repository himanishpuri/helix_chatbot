# gateway/cache.py
import os
import json
import uuid
import numpy as np
import redis.asyncio as aioredis

# ── Config ────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = 7 * 24 * 60 * 60  # 7 days in seconds
SIMILARITY_THRESHOLD = 0.92
# ─────────────────────────────────────────────────────────────

_redis: aioredis.Redis | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


async def get_cached(query_embedding: list[float]) -> str | None:
    """Return the most similar cached response above threshold, else None."""
    r = get_redis()
    best_sim = SIMILARITY_THRESHOLD
    best_key = None
    best_response = None
    async for key in r.scan_iter("cache:*"):
        entry = await r.hgetall(key)
        if not entry:
            continue
        try:
            stored_embedding = json.loads(entry["embedding"])
            similarity = _cosine_similarity(query_embedding, stored_embedding)
        except (KeyError, json.JSONDecodeError):
            continue
        if similarity >= best_sim:
            best_sim, best_key, best_response = similarity, key, entry["response"]

    if best_key is not None:
        # Refresh TTL on hit so popular questions stay cached
        await r.expire(best_key, CACHE_TTL)
        return best_response
    return None


async def set_cache(query_embedding: list[float], response_text: str) -> None:
    """Store embedding + response atomically with a TTL."""
    r = get_redis()
    key = f"cache:{uuid.uuid4()}"
    async with r.pipeline(transaction=True) as pipe:
        pipe.hset(
            key,
            mapping={
                "embedding": json.dumps(query_embedding),
                "response": response_text,
            },
        )
        pipe.expire(key, CACHE_TTL)
        await pipe.execute()
