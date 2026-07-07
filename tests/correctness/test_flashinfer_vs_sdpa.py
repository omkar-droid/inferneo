"""On CUDA, the FlashInfer fast path must agree with the SDPA reference.

This is the cross-check that lets the kernel backend be trusted: same engine,
same weights, same prompts — only the attention math differs. Marked ``gpu``
so it is skipped in CPU CI and run on the GPU box.
"""

import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.gpu


@pytest.fixture
def _cuda():
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    try:
        import flashinfer  # noqa: F401
    except ImportError:
        pytest.skip("flashinfer not installed")


@pytest.fixture(scope="module")
def fi_model_dir(tmp_path_factory) -> str:
    """Tiny Llama with head_dim=64 (FlashInfer supports 64/128/256, not 16)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=512,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,  # GQA
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    path = tmp_path_factory.mktemp("fi_llama")
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer").save_pretrained(path)
    return str(path)


def _run(monkeypatch, model_dir, backend, prompts, max_new):
    from inferneo import LLM, SamplingParams

    # FlashInfer compiles fp16/bf16 kernels only (no fp32), so both backends
    # run fp16 here; on a small well-conditioned model greedy tokens match.
    monkeypatch.setenv("INFERNEO_ATTENTION", backend)
    llm = LLM(model_dir, device="cuda", dtype="float16", max_num_seqs=8)
    outs = llm.generate(
        prompts,
        [SamplingParams(max_tokens=max_new, temperature=0, ignore_eos=True)] * len(prompts),
    )
    del llm
    torch.cuda.empty_cache()
    return [o.outputs[0].token_ids for o in outs]


def test_flashinfer_matches_sdpa(_cuda, fi_model_dir, monkeypatch):
    prompts = [
        [1, 5, 9, 22, 87, 3, 44],
        [1, 100, 200, 300],
        [1, 7, 8, 15, 16, 23, 42, 4, 8, 15],
    ]
    sdpa = _run(monkeypatch, fi_model_dir, "sdpa", prompts, max_new=16)
    flashinfer = _run(monkeypatch, fi_model_dir, "flashinfer", prompts, max_new=16)
    for p, a, b in zip(prompts, sdpa, flashinfer):
        assert a == b, f"backends diverged on {p}: sdpa={a} flashinfer={b}"
