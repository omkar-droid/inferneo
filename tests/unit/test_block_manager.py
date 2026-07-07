from inferneo.engine.request import EngineRequest
from inferneo.kv.block_manager import KVCacheManager
from inferneo.sampling_params import SamplingParams


def make_req(rid: str, num_prompt: int) -> EngineRequest:
    return EngineRequest(
        request_id=rid,
        prompt_token_ids=list(range(num_prompt)),
        sampling_params=SamplingParams(),
    )


def test_allocate_slots_block_math():
    kv = KVCacheManager(num_blocks=10, block_size=4)
    req = make_req("a", 6)
    fresh = kv.allocate_slots(req, 6)  # 6 tokens -> 2 blocks
    assert len(fresh) == 2
    req.num_computed_tokens = 6

    # 2 more tokens: 8 total still fits ceil(8/4)=2 blocks -> 0 fresh
    assert kv.allocate_slots(req, 2) == []
    req.num_computed_tokens = 8
    # 1 more token crosses into block 3
    assert len(kv.allocate_slots(req, 1)) == 1


def test_allocation_failure_leaves_state_clean():
    kv = KVCacheManager(num_blocks=2, block_size=4)
    req = make_req("a", 20)  # needs 5 blocks
    assert kv.allocate_slots(req, 20) is None
    assert kv.pool.num_free_blocks == 2


def test_free_returns_blocks():
    kv = KVCacheManager(num_blocks=4, block_size=4)
    req = make_req("a", 16)
    kv.allocate_slots(req, 16)
    assert kv.pool.num_free_blocks == 0
    kv.free(req)
    assert kv.pool.num_free_blocks == 4
    assert kv.get_block_ids("a") == []


def test_prefix_cache_hit_after_free():
    kv = KVCacheManager(num_blocks=8, block_size=4, enable_caching=True)
    req_a = make_req("a", 12)  # 3 full blocks, identical prompt
    kv.allocate_slots(req_a, 12)
    kv.free(req_a)

    req_b = make_req("b", 12)
    cached, num_cached = kv.get_computed_blocks(req_b)
    # Full-prompt hit must leave the last block uncomputed for logits.
    assert num_cached == 8
    assert len(cached) == 2
    fresh = kv.allocate_slots(req_b, 4, cached)
    assert len(fresh) == 1
    # Block table = 2 shared + 1 fresh, sharing block ids with req_a's prefix
    assert kv.get_block_ids("b")[:2] == [c.block_id for c in cached]


def test_partial_prefix_hit():
    kv = KVCacheManager(num_blocks=8, block_size=4, enable_caching=True)
    req_a = make_req("a", 8)
    kv.allocate_slots(req_a, 8)
    kv.free(req_a)

    # Same first block (tokens 0..3), different second block.
    req_b = EngineRequest(
        request_id="b",
        prompt_token_ids=[0, 1, 2, 3, 99, 98, 97, 96],
        sampling_params=SamplingParams(),
    )
    cached, num_cached = kv.get_computed_blocks(req_b)
    assert num_cached == 4
    assert len(cached) == 1


def test_generated_tokens_become_cacheable():
    kv = KVCacheManager(num_blocks=8, block_size=4, enable_caching=True)
    req = make_req("a", 6)
    kv.allocate_slots(req, 6)
    req.num_computed_tokens = 6
    # Two generated tokens fill block 2 (tokens 6,7 known at schedule time).
    req.output_token_ids.extend([500, 501])
    kv.allocate_slots(req, 2)
    req.num_computed_tokens = 8
    kv.free(req)

    twin = EngineRequest(
        request_id="b",
        prompt_token_ids=list(range(6)) + [500, 501, 502],
        sampling_params=SamplingParams(),
    )
    _, num_cached = kv.get_computed_blocks(twin)
    assert num_cached == 8  # both blocks, incl. the one holding generated tokens
