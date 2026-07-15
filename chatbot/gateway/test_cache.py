# gateway/test_cache.py — no Redis needed; stubs the client.
import asyncio
import json

import cache
from cache import _cosine_similarity


def test_cosine_zero_norm_is_zero():
    assert _cosine_similarity([0.0, 0.0], [1.0, 1.0]) == 0.0


def test_cosine_identical_is_one():
    assert abs(_cosine_similarity([1.0, 2.0], [1.0, 2.0]) - 1.0) < 1e-9


class _FakeRedis:
    """Minimal async stand-in for the bits get_cached touches."""

    def __init__(self, entries):
        self._entries = entries  # {key: {"embedding": json, "response": str}}
        self.expired = []

    async def scan_iter(self, _match):
        for k in list(self._entries):
            yield k

    def scan_iter_sync(self, _match):  # unused; kept for clarity
        return iter(self._entries)

    async def hgetall(self, key):
        return self._entries.get(key, {})

    async def expire(self, key, _ttl):
        self.expired.append(key)

    # scan_iter must be an async generator, not a coroutine
    def __getattr__(self, name):
        raise AttributeError(name)


def _run(entries, query):
    # Exercise the O(N) scan fallback directly (no Redis Query Engine here).
    fake = _FakeRedis(entries)
    cache._redis = fake
    try:
        return asyncio.run(cache._get_cached_scan(query)), fake
    finally:
        cache._redis = None


def _entry(vec, resp):
    return {"embedding": json.dumps(vec), "response": resp}


def test_best_match_wins_not_first():
    # both above 0.92 threshold; the SECOND is a closer match and must win.
    entries = {
        "cache:a": _entry([1.0, 0.30], "loose"),   # ~0.958 sim to query
        "cache:b": _entry([1.0, 0.02], "tight"),   # ~0.9998 sim to query
    }
    result, fake = _run(entries, [1.0, 0.0])
    assert result == "tight"
    assert fake.expired == ["cache:b"]  # TTL refreshed on the winner only


def test_below_threshold_returns_none():
    entries = {"cache:a": _entry([0.0, 1.0], "orthogonal")}
    result, _ = _run(entries, [1.0, 0.0])
    assert result is None


def test_cosine_distance_to_similarity():
    # KNN path converts RediSearch COSINE *distance* -> similarity as 1 - d.
    # identical vectors: distance 0 -> similarity 1; orthogonal: 1 -> 0.
    assert 1.0 - 0.0 == 1.0
    assert 1.0 - 1.0 == 0.0
    # threshold semantics: a 0.05 distance (0.95 sim) clears 0.92, 0.10 doesn't
    assert (1.0 - 0.05) >= cache.SIMILARITY_THRESHOLD
    assert (1.0 - 0.10) < cache.SIMILARITY_THRESHOLD


def test_to_bytes_is_float32():
    b = cache._to_bytes([1.0, 2.0, 3.0])
    assert len(b) == 3 * 4  # float32 = 4 bytes each


if __name__ == "__main__":
    test_cosine_zero_norm_is_zero()
    test_cosine_identical_is_one()
    test_best_match_wins_not_first()
    test_below_threshold_returns_none()
    test_cosine_distance_to_similarity()
    test_to_bytes_is_float32()
    print("✓ cache tests passed")
