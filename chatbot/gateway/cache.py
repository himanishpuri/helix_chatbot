# gateway/cache.py
import os
import json
import uuid
import numpy as np
import redis.asyncio as aioredis

# ── Config ────────────────────────────────────────────────────
REDIS_HOST           = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT           = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL            = 7 * 24 * 60 * 60  # 7 days in seconds
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
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


async def get_cached(query_embedding: list[float]) -> str | None:
    """Search all cached embeddings for a near match. Returns response text or None."""
    r = get_redis()
    async for key in r.scan_iter("cache:*"):
        entry = await r.hgetall(key)
        if not entry:
            continue
        try:
            stored_embedding = json.loads(entry["embedding"])
            similarity = _cosine_similarity(query_embedding, stored_embedding)
            if similarity >= SIMILARITY_THRESHOLD:
                # Refresh TTL on hit so popular questions stay cached
                await r.expire(key, CACHE_TTL)
                return entry["response"]
        except (KeyError, json.JSONDecodeError):
            continue
    return None


async def set_cache(query_embedding: list[float], response_text: str) -> None:
    """Store embedding + response in Redis."""
    r = get_redis()
    key = f"cache:{uuid.uuid4()}"
    await r.hset(key, mapping={
        "embedding": json.dumps(query_embedding),
        "response":  response_text,
    })
    await r.expire(key, CACHE_TTL)
