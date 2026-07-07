import pytest

from inferneo.kv.block_pool import BlockPool
from inferneo.kv.hashing import hash_block


def test_allocate_and_free_roundtrip():
    pool = BlockPool(4)
    blocks = pool.allocate(3)
    assert len(blocks) == 3
    assert pool.num_free_blocks == 1
    assert all(b.ref_count == 1 for b in blocks)
    for b in blocks:
        pool.free(b)
    assert pool.num_free_blocks == 4


def test_allocate_insufficient_returns_none():
    pool = BlockPool(2)
    assert pool.allocate(3) is None
    assert pool.num_free_blocks == 2  # nothing partially taken


def test_double_free_raises():
    pool = BlockPool(1)
    (b,) = pool.allocate(1)
    pool.free(b)
    with pytest.raises(ValueError):
        pool.free(b)


def test_usage():
    pool = BlockPool(4)
    pool.allocate(1)
    assert pool.usage() == 0.25


def test_cached_block_revived_from_free_queue():
    pool = BlockPool(2, enable_caching=True)
    (b,) = pool.allocate(1)
    h = hash_block(None, (1, 2, 3))
    pool.cache_block(b, h)
    pool.free(b)
    assert pool.num_free_blocks == 2

    hit = pool.get_cached_block(h)
    assert hit is b
    assert hit.ref_count == 1
    assert pool.num_free_blocks == 1


def test_lazy_eviction_drops_hash_on_reallocation():
    pool = BlockPool(1, enable_caching=True)
    (b,) = pool.allocate(1)
    h = hash_block(None, (7,))
    pool.cache_block(b, h)
    pool.free(b)
    # Reallocating the only block evicts its cached content.
    (b2,) = pool.allocate(1)
    assert b2 is b
    assert b2.block_hash is None
    assert pool.get_cached_block(h) is None


def test_caching_disabled_ignores_lookups():
    pool = BlockPool(2, enable_caching=False)
    (b,) = pool.allocate(1)
    h = hash_block(None, (1,))
    pool.cache_block(b, h)
    assert pool.get_cached_block(h) is None


def test_shared_block_refcounting():
    pool = BlockPool(2, enable_caching=True)
    (b,) = pool.allocate(1)
    h = hash_block(None, (9, 9))
    pool.cache_block(b, h)
    hit = pool.get_cached_block(h)  # second reference while still allocated
    assert hit is b and b.ref_count == 2
    pool.free(b)
    assert pool.num_free_blocks == 1  # still referenced once
    pool.free(b)
    assert pool.num_free_blocks == 2
