"""Structural guards for GPT-2, Gemma, and Phi — the deltas that make each
not-Llama must be exactly right. Full numerical equality vs HuggingFace is in
tests/correctness/test_new_arch_vs_hf.py; these fast CPU tests lock in the shape
so a refactor can't silently revert it.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from torch import nn  # noqa: E402

from inferneo.attention.selector import get_attention_backend  # noqa: E402


def _backend(cfg):
    head_dim = getattr(cfg, "head_dim", None) or cfg.hidden_size // cfg.num_attention_heads
    n_kv = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    return get_attention_backend(
        num_heads=cfg.num_attention_heads, num_kv_heads=n_kv,
        head_dim=head_dim, block_size=16, device=torch.device("cpu"), dtype=torch.float32,
    )


# ---------------------------------------------------------------- Gemma

def test_gemma_deltas():
    from transformers import GemmaConfig

    from inferneo.models.gemma import GemmaForCausalLM
    from inferneo.models.layers import GemmaRMSNorm

    cfg = GemmaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=1, num_attention_heads=4, num_key_value_heads=1,
                      head_dim=16, max_position_embeddings=128)
    model = GemmaForCausalLM(cfg, _backend(cfg))
    layer = model.model.layers[0]
    assert isinstance(layer.input_layernorm, GemmaRMSNorm)   # (1+weight), float32
    assert isinstance(model.model.norm, GemmaRMSNorm)
    assert isinstance(layer.mlp.act_fn, nn.GELU)             # GeGLU, not SwiGLU
    assert layer.mlp.act_fn.approximate == "tanh"
    # embeddings scaled by sqrt(hidden) before layer 0
    assert model.model.embedding_multiplier == pytest.approx(64**0.5)
    # explicit head_dim that isn't hidden/num_heads is honored (MQA: 1 kv head)
    assert layer.self_attn.head_dim == 16
    assert layer.self_attn.kv_size == 16


def test_gemma_norm_uses_one_plus_weight():
    from inferneo.models.layers import GemmaRMSNorm, RMSNorm

    torch.manual_seed(0)
    x = torch.randn(3, 8)
    g, r = GemmaRMSNorm(8, 1e-6), RMSNorm(8, 1e-6)
    with torch.no_grad():
        r.weight.zero_()  # give Llama's norm the same zero weight
    # zero weight: Gemma scales by (1+0)=1 (passes normalized x through), while
    # Llama scales by 0 (zeros everything out) — that's the (1+weight) delta.
    assert g(x).abs().sum() > 0
    assert torch.allclose(r(x), torch.zeros_like(x))


# ---------------------------------------------------------------- Phi

def test_phi_deltas():
    from transformers import PhiConfig

    from inferneo.models.phi import PhiForCausalLM

    cfg = PhiConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                    num_hidden_layers=1, num_attention_heads=4,
                    partial_rotary_factor=0.5, max_position_embeddings=128)
    model = PhiForCausalLM(cfg, _backend(cfg))
    layer = model.model.layers[0]
    # parallel block: ONE shared norm, no separate post-attention norm
    assert isinstance(layer.input_layernorm, nn.LayerNorm)
    assert not hasattr(layer, "post_attention_layernorm")
    # partial rotary: rotate only half of the 16-wide head
    assert model.model.rotary.rotary_dim == 8
    assert layer.self_attn.head_dim == 16
    # biases throughout; output proj named `dense`; lm_head has a bias and isn't tied
    assert layer.self_attn.qkv_proj.bias is not None
    assert layer.self_attn.dense.bias is not None
    assert model.lm_head.bias is not None


def test_phi_parallel_residual():
    """attn and mlp both read the SAME normed input (not chained), and both add
    to the original residual — the parallel block, verified behaviorally."""
    from transformers import PhiConfig

    from inferneo.models.phi import PhiForCausalLM

    cfg = PhiConfig(vocab_size=256, hidden_size=8, intermediate_size=16,
                    num_hidden_layers=1, num_attention_heads=2,
                    partial_rotary_factor=0.5, max_position_embeddings=32)
    layer = PhiForCausalLM(cfg, _backend(cfg)).model.layers[0]

    seen = {}

    class FakeAttn(nn.Module):
        def forward(self, h, *a, **k):
            seen["attn_in"] = h
            return torch.zeros_like(h)

    class FakeMLP(nn.Module):
        def forward(self, h):
            seen["mlp_in"] = h
            return torch.zeros_like(h)

    layer.self_attn = FakeAttn()
    layer.mlp = FakeMLP()

    x = torch.randn(3, 8)
    out = layer(x, None, None, None, None)
    assert torch.allclose(seen["attn_in"], seen["mlp_in"])          # same input
    assert torch.allclose(seen["attn_in"], layer.input_layernorm(x))  # = norm(x)
    assert torch.allclose(out, x)  # residual is original x (both stubs returned 0)


# ---------------------------------------------------------------- GPT-2

def test_gpt2_deltas():
    from transformers import GPT2Config

    from inferneo.models.gpt2 import GPT2LMHeadModel

    cfg = GPT2Config(vocab_size=256, n_embd=64, n_head=4, n_layer=1, n_positions=128,
                     n_inner=128)
    model = GPT2LMHeadModel(cfg, _backend(cfg))
    # learned absolute positions, not RoPE
    assert isinstance(model.transformer.wpe, nn.Embedding)
    assert model.transformer.wpe.num_embeddings == 128
    block = model.transformer.h[0]
    assert isinstance(block.ln_1, nn.LayerNorm)             # LayerNorm, not RMSNorm
    assert block.attn.c_attn.out_features == 3 * 64         # fused q|k|v
    assert isinstance(block.mlp.act_fn, nn.GELU)            # non-gated GELU MLP
    assert model.lm_head.bias is None                       # tied to wte, no bias


def test_gpt2_fuse_transposes_conv1d_drops_mask_and_adds_prefix():
    from inferneo.models.gpt2 import GPT2LMHeadModel

    # The distributed `gpt2` checkpoint uses base-model keys with NO
    # `transformer.` prefix and Conv1D-layout weights ([in, out]).
    state = {
        "h.0.attn.c_attn.weight": torch.zeros(64, 192),   # Conv1D [in, out]
        "h.0.attn.c_attn.bias": torch.zeros(192),
        "h.0.mlp.c_fc.weight": torch.zeros(64, 128),
        "h.0.attn.bias": torch.ones(1, 1, 128, 128),      # causal-mask buffer
        "h.0.attn.masked_bias": torch.tensor(-1e4),
        "wte.weight": torch.zeros(256, 64),
    }
    fused = GPT2LMHeadModel.fuse_state_dict(state)
    assert fused["transformer.h.0.attn.c_attn.weight"].shape == (192, 64)  # transposed
    assert fused["transformer.h.0.mlp.c_fc.weight"].shape == (128, 64)
    assert fused["transformer.h.0.attn.c_attn.bias"].shape == (192,)       # bias untouched
    assert "transformer.wte.weight" in fused                               # prefix added
    assert "transformer.h.0.attn.bias" not in fused                        # mask dropped
    assert "transformer.h.0.attn.masked_bias" not in fused


# ---------------------------------------------------------------- registry

def test_backbone_accessor_points_at_the_right_module():
    """The CUDA-graph runner captures `model.backbone` (all layers but lm_head).
    Its attribute name differs by family — this guards the GPT-2 fp16 regression
    where the fast path hardcoded `.model` and GPT-2 uses `.transformer`."""
    from transformers import GemmaConfig, GPT2Config, PhiConfig

    from inferneo.models.gemma import GemmaForCausalLM
    from inferneo.models.gpt2 import GPT2LMHeadModel
    from inferneo.models.phi import PhiForCausalLM

    gpt2_cfg = GPT2Config(vocab_size=256, n_embd=64, n_head=4, n_layer=1, n_positions=128)
    m = GPT2LMHeadModel(gpt2_cfg, _backend(gpt2_cfg))
    assert m.backbone is m.transformer

    phi_cfg = PhiConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                        num_hidden_layers=1, num_attention_heads=4,
                        max_position_embeddings=128)
    m = PhiForCausalLM(phi_cfg, _backend(phi_cfg))
    assert m.backbone is m.model

    gemma_cfg = GemmaConfig(vocab_size=256, hidden_size=64, intermediate_size=128,
                            num_hidden_layers=1, num_attention_heads=4,
                            num_key_value_heads=1, head_dim=16, max_position_embeddings=128)
    m = GemmaForCausalLM(gemma_cfg, _backend(gemma_cfg))
    assert m.backbone is m.model


def test_registry_resolves_new_families():
    from inferneo.models.gemma import GemmaForCausalLM
    from inferneo.models.gpt2 import GPT2LMHeadModel
    from inferneo.models.phi import PhiForCausalLM
    from inferneo.models.registry import get_model_class

    assert get_model_class(["GemmaForCausalLM"]) is GemmaForCausalLM
    assert get_model_class(["PhiForCausalLM"]) is PhiForCausalLM
    assert get_model_class(["GPT2LMHeadModel"]) is GPT2LMHeadModel
