# gateway/cache.py
import json
import uuid
import numpy as np
import redis

# ── Config ────────────────────────────────────────────────────
REDIS_HOST  = "localhost"
REDIS_PORT  = 6379
CACHE_TTL   = 7 * 24 * 60 * 60  # 7 days in seconds
SIMILARITY_THRESHOLD = 0.92
# ─────────────────────────────────────────────────────────────

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def get_cached(query_embedding: list[float]) -> str | None:
    """Search all cached embeddings for a near match. Returns response text or None."""
    keys = r.keys("cache:*")
    
    for key in keys:
        entry = r.hgetall(key)
        if not entry:
            continue
        try:
            stored_embedding = json.loads(entry["embedding"])
            similarity = _cosine_similarity(query_embedding, stored_embedding)
            if similarity >= SIMILARITY_THRESHOLD:
                # Refresh TTL on hit so popular questions stay cached
                r.expire(key, CACHE_TTL)
                return entry["response"]
        except (KeyError, json.JSONDecodeError):
            continue
    
    return None


def set_cache(query_embedding: list[float], response_text: str) -> None:
    """Store embedding + response in Redis."""
    key = f"cache:{uuid.uuid4()}"
    r.hset(key, mapping={
        "embedding": json.dumps(query_embedding),
        "response":  response_text
    })
    r.expire(key, CACHE_TTL)