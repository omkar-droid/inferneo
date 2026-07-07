import pytest


@pytest.fixture(scope="session")
def tiny_model_dir(tmp_path_factory) -> str:
    """A tiny random-weight Llama (2 layers, GQA) saved as HF safetensors.

    Small enough that the full correctness suite runs on CPU in seconds.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaConfig

    torch.manual_seed(0)
    config = LlamaConfig(
        # vocab matches the llama tokenizer so text-prompt paths (server) work.
        vocab_size=32000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,  # GQA path exercised
        max_position_embeddings=2048,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    model = AutoModelForCausalLM.from_config(config)
    model.eval()
    path = tmp_path_factory.mktemp("tiny_llama")
    model.save_pretrained(path)
    tok = AutoTokenizer.from_pretrained("hf-internal-testing/llama-tokenizer")
    tok.save_pretrained(path)
    return str(path)


@pytest.fixture(scope="session")
def tiny_hf_model(tiny_model_dir):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(tiny_model_dir, dtype=torch.float32)
    model.eval()
    return model


def hf_greedy(hf_model, prompt_ids: list[int], max_new: int) -> list[int]:
    """Reference: HF generate, greedy, eos disabled."""
    import torch

    with torch.no_grad():
        out = hf_model.generate(
            torch.tensor([prompt_ids]),
            max_new_tokens=max_new,
            do_sample=False,
            pad_token_id=0,
            eos_token_id=None,
        )
    return out[0][len(prompt_ids) :].tolist()
