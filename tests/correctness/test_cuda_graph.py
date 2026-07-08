"""On CUDA, the graph-captured decode path must produce identical greedy tokens
to eager execution — same engine, same weights, only the decode step differs."""

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
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

    torch.manual_seed(0)
    config = LlamaConfig(
        vocab_size=32000, hidden_size=128, intermediate_size=256,
        num_hidden_layers=2, num_attention_heads=2, num_key_value_heads=1,
        max_position_embeddings=2048, rms_norm_eps=1e-6, tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(config).eval()
    path = tmp_path_factory.mktemp("fi_llama")
    model.save_pretrained(path)
    AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer").save_pretrained(path)
    return str(path)


def _run(model_dir, enable_cuda_graph, prompts, max_new):
    from inferneo import LLM, SamplingParams

    llm = LLM(
        model_dir, device="cuda", dtype="float16", max_num_seqs=8,
        enable_cuda_graph=enable_cuda_graph,
    )
    assert (llm.engine.runner.graph_runner is not None) == enable_cuda_graph
    outs = llm.generate(
        prompts,
        [SamplingParams(max_tokens=max_new, temperature=0, ignore_eos=True)] * len(prompts),
    )
    del llm
    torch.cuda.empty_cache()
    return [o.outputs[0].token_ids for o in outs]


def test_graph_matches_eager(_cuda, fi_model_dir):
    prompts = [
        [1, 5, 9, 22, 87, 3, 44],
        [1, 100, 200, 300],
        [1, 7, 8, 15, 16, 23, 42, 4, 8, 15],
    ]
    eager = _run(fi_model_dir, False, prompts, max_new=20)
    graphed = _run(fi_model_dir, True, prompts, max_new=20)
    for p, a, b in zip(prompts, eager, graphed):
        assert a == b, f"graph diverged from eager on {p}: eager={a} graph={b}"
