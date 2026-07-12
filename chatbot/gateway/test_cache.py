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
    fake = _FakeRedis(entries)
    cache._redis = fake
    try:
        return asyncio.run(cache.get_cached(query)), fake
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


if __name__ == "__main__":
    test_cosine_zero_norm_is_zero()
    test_cosine_identical_is_one()
    test_best_match_wins_not_first()
    test_below_threshold_returns_none()
    print("✓ cache tests passed")
