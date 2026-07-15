# gateway/cache.py
# Semantic response cache. Prefers a Redis vector index (ANN KNN, ~O(log N));
# falls back to an O(N) scan if the Redis Query Engine isn't available, so the
# gateway boots against any Redis. Public API: get_cached / set_cache.
import os
import json
import uuid
import numpy as np
import redis.asyncio as aioredis
from redis.exceptions import ResponseError

# ── Config ────────────────────────────────────────────────────
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
CACHE_TTL = 7 * 24 * 60 * 60  # 7 days in seconds
SIMILARITY_THRESHOLD = 0.92
INDEX_NAME = "cache_idx"
KEY_PREFIX = "cache:"
# ─────────────────────────────────────────────────────────────

# text client (decode_responses) for the scan fallback + response reads;
# bytes client for raw float32 vector payloads. Both lazy.
_redis: aioredis.Redis | None = None
_redis_bytes: aioredis.Redis | None = None
# None = not probed yet; True = vector index ready; False = scan fallback.
_vector_ok: bool | None = None


def get_redis() -> aioredis.Redis:
    global _redis
    if _redis is None:
        _redis = aioredis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)
    return _redis


def _get_redis_bytes() -> aioredis.Redis:
    global _redis_bytes
    if _redis_bytes is None:
        _redis_bytes = aioredis.Redis(
            host=REDIS_HOST, port=REDIS_PORT, decode_responses=False
        )
    return _redis_bytes


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    a, b = np.array(a), np.array(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def _to_bytes(vec: list[float]) -> bytes:
    return np.asarray(vec, dtype=np.float32).tobytes()


async def _ensure_index(dim: int) -> bool:
    """Create the vector index once. Returns True if vector search is usable."""
    global _vector_ok
    if _vector_ok is not None:
        return _vector_ok
    r = _get_redis_bytes()
    try:
        await r.execute_command(
            "FT.CREATE", INDEX_NAME,
            "ON", "HASH",
            "PREFIX", "1", KEY_PREFIX,
            "SCHEMA",
            "response", "TEXT",
            "embedding", "VECTOR", "HNSW", "6",
            "TYPE", "FLOAT32",
            "DIM", str(dim),
            "DISTANCE_METRIC", "COSINE",
        )
        _vector_ok = True
        print(f"[cache] vector KNN mode (dim={dim})")
    except ResponseError as e:
        if "Index already exists" in str(e):
            _vector_ok = True
            print("[cache] vector KNN mode (existing index)")
        else:
            # Query Engine not present (plain Redis) → linear-scan fallback.
            _vector_ok = False
            print(f"[cache] scan fallback ({e})")
    except Exception as e:  # connection / unknown command on old servers
        _vector_ok = False
        print(f"[cache] scan fallback ({e})")
    return _vector_ok


# ── Vector (ANN) path ─────────────────────────────────────────
async def _get_cached_knn(query_embedding: list[float]) -> str | None:
    r = _get_redis_bytes()
    q = "*=>[KNN 1 @embedding $vec AS score]"
    try:
        res = await r.execute_command(
            "FT.SEARCH", INDEX_NAME, q,
            "PARAMS", "2", "vec", _to_bytes(query_embedding),
            "SORTBY", "score",
            "RETURN", "2", "score", "__key",
            "DIALECT", "2",
        )
    except ResponseError:
        return None
    # res: [total, key, [field, val, field, val, ...], ...]
    if not res or int(res[0]) == 0:
        return None
    key = res[1].decode() if isinstance(res[1], bytes) else res[1]
    fields = {
        (f.decode() if isinstance(f, bytes) else f): v
        for f, v in zip(res[2][0::2], res[2][1::2])
    }
    raw_score = fields.get(b"score") or fields.get("score")
    distance = float(raw_score)
    similarity = 1.0 - distance  # COSINE distance → similarity
    if similarity < SIMILARITY_THRESHOLD:
        return None
    tr = get_redis()
    await tr.expire(key, CACHE_TTL)  # refresh TTL on hit
    resp = await tr.hget(key, "response")
    return resp


async def _set_cache_knn(query_embedding: list[float], response_text: str) -> None:
    r = _get_redis_bytes()
    key = f"{KEY_PREFIX}{uuid.uuid4()}"
    await r.hset(
        key,
        mapping={
            b"embedding": _to_bytes(query_embedding),
            b"response": response_text.encode("utf-8"),
        },
    )
    await r.expire(key, CACHE_TTL)


# ── O(N) scan fallback (works on any Redis) ───────────────────
async def _get_cached_scan(query_embedding: list[float]) -> str | None:
    r = get_redis()
    best_sim = SIMILARITY_THRESHOLD
    best_key = None
    best_response = None
    async for key in r.scan_iter(f"{KEY_PREFIX}*"):
        entry = await r.hgetall(key)
        if not entry or "embedding" not in entry:
            continue
        try:
            stored = json.loads(entry["embedding"])
            similarity = _cosine_similarity(query_embedding, stored)
        except (KeyError, json.JSONDecodeError, TypeError):
            continue
        if similarity >= best_sim:
            best_sim, best_key, best_response = similarity, key, entry["response"]
    if best_key is not None:
        await r.expire(best_key, CACHE_TTL)
        return best_response
    return None


async def _set_cache_scan(query_embedding: list[float], response_text: str) -> None:
    r = get_redis()
    key = f"{KEY_PREFIX}{uuid.uuid4()}"
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


# ── Public API ────────────────────────────────────────────────
async def get_cached(query_embedding: list[float]) -> str | None:
    """Return the most similar cached response above threshold, else None."""
    if await _ensure_index(len(query_embedding)):
        return await _get_cached_knn(query_embedding)
    return await _get_cached_scan(query_embedding)


async def set_cache(query_embedding: list[float], response_text: str) -> None:
    """Store embedding + response with a TTL, in whichever backend is active."""
    if await _ensure_index(len(query_embedding)):
        await _set_cache_knn(query_embedding, response_text)
    else:
        await _set_cache_scan(query_embedding, response_text)
