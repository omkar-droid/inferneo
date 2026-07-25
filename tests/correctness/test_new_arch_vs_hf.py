"""GPT-2, Gemma, and Phi must decode token-for-token like HuggingFace — the same
bar Llama/Qwen already meet. Each test builds a tiny random-weight checkpoint,
loads it through the full paged engine on CPU (fp32), and compares greedy output
against HF ``generate``. This exercises every delta end to end: GPT-2's learned
positions + Conv1D transpose, Gemma's (1+w) norm + GeGLU + embedding scale, and
Phi's partial rotary + parallel block.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from inferneo import LLM, SamplingParams  # noqa: E402
from tests.conftest import hf_greedy  # noqa: E402

MAX_NEW = 12
PROMPTS = [
    [1, 5, 9, 22, 87, 3, 44],
    [1, 100, 200, 300],
    [1, 7, 8, 15, 16, 23, 42],
]
# vocab matches the llama tokenizer the engine loads for detok (ids-in, ids-out).
VOCAB = 32000


def _configs():
    from transformers import GemmaConfig, GPT2Config, PhiConfig

    common = dict(hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, max_position_embeddings=256)
    return {
        "gpt2": GPT2Config(vocab_size=VOCAB, n_embd=64, n_head=4, n_layer=2,
                           n_positions=256, n_inner=128),
        "gemma": GemmaConfig(vocab_size=VOCAB, num_key_value_heads=1, head_dim=16,
                             **common),
        "phi": PhiConfig(vocab_size=VOCAB, num_key_value_heads=4,
                         partial_rotary_factor=0.5, **common),
    }


def _make_checkpoint(tmp_path, name, config):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    d = tmp_path / name
    model.save_pretrained(d)
    # The engine builds a tokenizer from the model dir; reuse the llama one (only
    # used for detok, which we ignore — the assertion is on token ids).
    AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer").save_pretrained(d)
    return model, str(d)


@pytest.mark.parametrize("name", ["gpt2", "gemma", "phi"])
def test_greedy_matches_hf(tmp_path, name):
    config = _configs()[name]
    hf_model, model_dir = _make_checkpoint(tmp_path, name, config)
    refs = {tuple(p): hf_greedy(hf_model, p, MAX_NEW) for p in PROMPTS}

    llm = LLM(model_dir, device="cpu", dtype="float32",
              num_blocks=128, max_num_batched_tokens=64, max_num_seqs=8)
    params = SamplingParams(max_tokens=MAX_NEW, temperature=0, ignore_eos=True)
    outs = llm.generate(PROMPTS, [params] * len(PROMPTS))
    for p, out in zip(PROMPTS, outs):
        assert out.outputs[0].token_ids == refs[tuple(p)], (name, p)


@pytest.mark.parametrize("name", ["gpt2", "gemma", "phi"])
def test_chunked_prefill_matches_hf(tmp_path, name):
    """A tiny token budget forces chunked prefill — positions must stay correct
    across chunks (learned positions for GPT-2, RoPE offsets for Gemma/Phi)."""
    config = _configs()[name]
    hf_model, model_dir = _make_checkpoint(tmp_path, name, config)
    p = PROMPTS[2]
    ref = hf_greedy(hf_model, p, MAX_NEW)

    llm = LLM(model_dir, device="cpu", dtype="float32",
              num_blocks=128, block_size=4, max_num_batched_tokens=4)
    out = llm.generate([p], SamplingParams(max_tokens=MAX_NEW, temperature=0, ignore_eos=True))
    assert out[0].outputs[0].token_ids == ref
