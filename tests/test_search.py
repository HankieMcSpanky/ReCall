"""Tests for vector index and hybrid retrieval."""
import numpy as np

from neuropack.search.vector_index import BruteForceIndex


def _random_vec(dim=256, seed=None):
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    v /= np.linalg.norm(v)
    return v


def test_empty_index():
    idx = BruteForceIndex()
    results = idx.search(_random_vec(), k=5)
    assert results == []


def test_add_and_search():
    idx = BruteForceIndex()
    v = _random_vec(seed=42)
    idx.add("mem-1", v)
    results = idx.search(v, k=1)
    assert len(results) == 1
    assert results[0][0] == "mem-1"
    assert abs(results[0][1] - 1.0) < 1e-5  # self-similarity


def test_top_k_ordering():
    idx = BruteForceIndex()
    target = _random_vec(seed=0)

    # Add vectors with known similarity (all L2-normalized)
    idx.add("exact", target)
    similar = target + _random_vec(seed=1) * 0.3
    similar /= np.linalg.norm(similar)  # must normalize
    idx.add("similar", similar)
    idx.add("different", _random_vec(seed=99))

    results = idx.search(target, k=3)
    assert results[0][0] == "exact"
    assert results[0][1] > results[1][1]


def test_remove():
    idx = BruteForceIndex()
    v1 = _random_vec(seed=1)
    v2 = _random_vec(seed=2)
    idx.add("mem-1", v1)
    idx.add("mem-2", v2)

    idx.remove("mem-1")
    assert len(idx) == 1
    results = idx.search(v1, k=5)
    assert all(r[0] != "mem-1" for r in results)


def test_remove_nonexistent():
    idx = BruteForceIndex()
    idx.remove("nonexistent")  # Should not raise


def test_build_from_items():
    items = [
        ("a", _random_vec(seed=10)),
        ("b", _random_vec(seed=20)),
        ("c", _random_vec(seed=30)),
    ]
    idx = BruteForceIndex()
    idx.build(items)
    assert len(idx) == 3


def test_build_empty():
    idx = BruteForceIndex()
    idx.build([])
    assert len(idx) == 0
    assert idx.search(_random_vec(), k=5) == []
