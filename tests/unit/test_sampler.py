import torch

from inferneo.sampling.sampler import RequestSamplerState, Sampler
from inferneo.sampling_params import SamplingParams


def state(**kwargs) -> RequestSamplerState:
    params = SamplingParams(**kwargs)
    return RequestSamplerState(params=params, prompt_len=0)


def test_greedy_is_argmax():
    sampler = Sampler(seed=0)
    logits = torch.tensor([[0.1, 5.0, 0.2, -1.0]])
    tokens, _ = sampler.sample(logits, [state(temperature=0)])
    assert tokens == [1]


def test_seed_determinism():
    logits = torch.randn(1, 100)
    a, _ = Sampler().sample(logits, [state(temperature=1.0, seed=123)])
    b, _ = Sampler().sample(logits, [state(temperature=1.0, seed=123)])
    assert a == b


def test_top_k_one_is_argmax():
    sampler = Sampler(seed=0)
    logits = torch.tensor([[1.0, 2.0, 3.0, 0.5]])
    tokens, _ = sampler.sample(logits, [state(temperature=1.0, top_k=1)])
    assert tokens == [2]


def test_logprobs_returned():
    sampler = Sampler(seed=0)
    logits = torch.tensor([[0.0, 10.0, 0.0]])
    _, logprobs = sampler.sample(logits, [state(temperature=0, logprobs=2)])
    lp = logprobs[0]
    assert lp.token_id == 1
    assert len(lp.top) == 2
    assert lp.logprob == max(lp.top.values())


def test_repetition_penalty_suppresses_seen_token():
    sampler = Sampler(seed=0)
    logits = torch.tensor([[5.0, 4.9, 0.0]])
    st = state(temperature=0, repetition_penalty=2.0)
    st.token_ids = [0]  # token 0 already seen
    st.prompt_len = 1
    tokens, _ = sampler.sample(logits, [st])
    assert tokens == [1]  # penalized 0 now loses to 1


def test_batched_rows_independent():
    sampler = Sampler(seed=0)
    logits = torch.tensor([[9.0, 0.0], [0.0, 9.0]])
    tokens, _ = sampler.sample(logits, [state(temperature=0), state(temperature=0)])
    assert tokens == [0, 1]
