"""Greedy decoding must match HF token-for-token, in the shapes that stress
the engine: single request, a ragged batch decoded together, chunked prefill,
preemption-and-resume, and prefix caching.

These are the regression tests that guarantee the paged engine computes the
same thing the reference does — the property the deleted stub never had.
"""

import pytest

from inferneo import LLM, SamplingParams
from tests.conftest import hf_greedy

MAX_NEW = 16
PROMPTS = [
    [1, 5, 9, 22, 87, 3, 44],
    [1, 100, 200, 300],
    [1, 7, 8, 15, 16, 23, 42, 4, 8, 15],
    [1, 42],
]


def greedy(**kwargs):
    return SamplingParams(max_tokens=MAX_NEW, temperature=0, ignore_eos=True, **kwargs)


@pytest.fixture(scope="module")
def refs(tiny_hf_model):
    return {tuple(p): hf_greedy(tiny_hf_model, p, MAX_NEW) for p in PROMPTS}


def test_single_request(tiny_model_dir, refs):
    llm = LLM(tiny_model_dir, device="cpu", dtype="float32", num_blocks=128)
    out = llm.generate([PROMPTS[0]], greedy())
    assert out[0].outputs[0].token_ids == refs[tuple(PROMPTS[0])]


def test_ragged_batch(tiny_model_dir, refs):
    llm = LLM(
        tiny_model_dir, device="cpu", dtype="float32",
        num_blocks=128, max_num_batched_tokens=64, max_num_seqs=8,
    )
    outs = llm.generate(PROMPTS, [greedy()] * len(PROMPTS))
    for p, out in zip(PROMPTS, outs):
        assert out.outputs[0].token_ids == refs[tuple(p)], p


def test_chunked_prefill(tiny_model_dir, refs):
    # Tiny token budget forces long prompts to prefill in chunks.
    llm = LLM(
        tiny_model_dir, device="cpu", dtype="float32",
        num_blocks=128, max_num_batched_tokens=4, block_size=4,
    )
    p = PROMPTS[2]
    out = llm.generate([p], greedy())
    assert out[0].outputs[0].token_ids == refs[tuple(p)]


def test_preemption_and_resume(tiny_model_dir, refs):
    # Scarce blocks force preemption; recompute-on-resume must reproduce output.
    llm = LLM(
        tiny_model_dir, device="cpu", dtype="float32",
        num_blocks=8, block_size=4, max_num_batched_tokens=64, max_num_seqs=8,
    )
    outs = llm.generate(PROMPTS, [greedy()] * len(PROMPTS))
    for p, out in zip(PROMPTS, outs):
        assert out.outputs[0].token_ids == refs[tuple(p)], p


def test_prefix_caching_matches(tiny_model_dir, refs):
    llm = LLM(
        tiny_model_dir, device="cpu", dtype="float32",
        num_blocks=128, block_size=4, enable_prefix_caching=True,
        max_num_batched_tokens=64, max_num_seqs=8,
    )
    # Same prompt twice + others; the second occurrence hits the cache.
    prompts = [PROMPTS[2], PROMPTS[0], PROMPTS[2]]
    outs = llm.generate(prompts, [greedy()] * len(prompts))
    for p, out in zip(prompts, outs):
        assert out.outputs[0].token_ids == refs[tuple(p)], p


def test_stop_token_id(tiny_model_dir, refs):
    llm = LLM(tiny_model_dir, device="cpu", dtype="float32", num_blocks=128)
    ref = refs[tuple(PROMPTS[0])]
    stop = ref[5]  # stop on the 6th generated token
    out = llm.generate([PROMPTS[0]], greedy(stop_token_ids=[stop]))
    got = out[0].outputs[0].token_ids
    assert got == ref[:6]
    assert out[0].outputs[0].finish_reason == "stop"
