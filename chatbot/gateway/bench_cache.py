# gateway/bench_cache.py — measure semantic-cache LOOKUP cost: vector KNN vs O(N) scan.
#
# Populates N random 384-dim entries (bge-small's dim), then times get_cached()
# in each backend. Not a unit test; a reproducible micro-benchmark for the docs.
#
#   Needs a Redis on $REDIS_HOST:$REDIS_PORT. For the KNN path it must be redis:8
#   (Query Engine). Run:
#     docker run -d --rm -p 6379:6379 --name bench-redis redis:8
#     uv run python bench_cache.py
#     docker rm -f bench-redis
import asyncio
import os
import time

import numpy as np
import cache

DIM = 384
SIZES = [100, 500, 2000]
REPEATS = 50


def _rand_unit(dim: int) -> list[float]:
    v = np.random.default_rng().standard_normal(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v.tolist()


async def _populate_knn(n: int) -> None:
    r = cache._get_redis_bytes()
    async with r.pipeline(transaction=False) as pipe:
        for _ in range(n):
            key = f"{cache.KEY_PREFIX}{os.urandom(8).hex()}"
            pipe.hset(key, mapping={b"embedding": cache._to_bytes(_rand_unit(DIM)),
                                    b"response": b"x"})
        await pipe.execute()


async def _populate_scan(n: int) -> None:
    import json
    r = cache.get_redis()
    async with r.pipeline(transaction=False) as pipe:
        for _ in range(n):
            key = f"{cache.KEY_PREFIX}{os.urandom(8).hex()}"
            pipe.hset(key, mapping={"embedding": json.dumps(_rand_unit(DIM)),
                                    "response": "x"})
        await pipe.execute()


async def _time(fn, q) -> float:
    # median of REPEATS lookups, in milliseconds
    ts = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        await fn(q)
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    return ts[len(ts) // 2]


async def main():
    r = cache.get_redis()
    await r.flushdb()
    print(f"dim={DIM}  repeats={REPEATS}  (median ms per lookup)\n")
    print(f"{'N entries':>10} | {'KNN (ms)':>10} | {'scan (ms)':>10}")
    print("-" * 38)
    for n in SIZES:
        # --- KNN path (redis:8) ---
        await r.flushdb()
        cache._vector_ok = None
        ok = await cache._ensure_index(DIM)
        knn_ms = float("nan")
        if ok:
            await _populate_knn(n)
            knn_ms = await _time(cache._get_cached_knn, _rand_unit(DIM))

        # --- scan path (any Redis) ---
        await r.flushdb()
        try:
            await r.execute_command("FT.DROPINDEX", cache.INDEX_NAME)
        except Exception:
            pass
        await _populate_scan(n)
        scan_ms = await _time(cache._get_cached_scan, _rand_unit(DIM))

        print(f"{n:>10} | {knn_ms:>10.3f} | {scan_ms:>10.3f}")
    await r.flushdb()


if __name__ == "__main__":
    asyncio.run(main())
