"""Qwen wiring — the small deltas that make it not-Llama must be exactly right.

Numerical correctness (logits match HuggingFace to 0.0000) is verified on GPU with
the real checkpoints; these fast CPU tests lock in the structural differences so a
refactor can't silently revert them.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from inferneo.attention.selector import get_attention_backend  # noqa: E402


def _backend(cfg):
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    return get_attention_backend(
        num_heads=cfg.num_attention_heads, num_kv_heads=cfg.num_key_value_heads,
        head_dim=head_dim, block_size=16, device=torch.device("cpu"), dtype=torch.float32,
    )


def test_qwen2_has_qkv_bias_but_no_o_bias():
    from transformers import Qwen2Config

    from inferneo.models.qwen import Qwen2ForCausalLM

    cfg = Qwen2Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128)
    attn = Qwen2ForCausalLM(cfg, _backend(cfg)).model.layers[0].self_attn
    assert attn.qkv_proj.bias is not None   # Qwen2: q/k/v carry a bias
    assert attn.o_proj.bias is None         # ... but the output projection doesn't
    assert not attn.qk_norm


def test_qwen3_has_qk_norm_but_no_qkv_bias():
    from transformers import Qwen3Config

    from inferneo.models.qwen import Qwen3ForCausalLM

    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      head_dim=16, max_position_embeddings=128)
    attn = Qwen3ForCausalLM(cfg, _backend(cfg)).model.layers[0].self_attn
    assert attn.qk_norm and hasattr(attn, "q_norm") and hasattr(attn, "k_norm")
    assert attn.qkv_proj.bias is None       # Qwen3 dropped the qkv bias


def test_llama_unchanged_by_generalization():
    """The default (Llama) must still have no biases and no QK-norm."""
    from transformers import LlamaConfig

    from inferneo.models.llama import LlamaForCausalLM

    cfg = LlamaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=2,
                      max_position_embeddings=128)
    attn = LlamaForCausalLM(cfg, _backend(cfg)).model.layers[0].self_attn
    assert attn.qkv_proj.bias is None and attn.o_proj.bias is None and not attn.qk_norm


def test_registry_resolves_qwen():
    from inferneo.models.qwen import Qwen2ForCausalLM, Qwen3ForCausalLM
    from inferneo.models.registry import get_model_class

    assert get_model_class(["Qwen2ForCausalLM"]) is Qwen2ForCausalLM
    assert get_model_class(["Qwen3ForCausalLM"]) is Qwen3ForCausalLM


def test_fuse_merges_qkv_bias_in_order():
    """A Qwen2 checkpoint has separate q/k/v biases; fuse_state_dict must merge them
    into qkv_proj.bias in q,k,v order (matching the weight merge)."""
    from inferneo.models.llama import LlamaForCausalLM

    p = "model.layers.0.self_attn."
    state = {
        p + "q_proj.bias": torch.zeros(8),
        p + "k_proj.bias": torch.ones(4),
        p + "v_proj.bias": torch.full((4,), 2.0),
    }
    fused = LlamaForCausalLM.fuse_state_dict(state)
    assert fused[p + "qkv_proj.bias"].tolist() == [0] * 8 + [1] * 4 + [2] * 4
