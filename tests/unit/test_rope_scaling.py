"""RoPE scaling must match HuggingFace exactly.

These are the frequencies that tell the model *where* each token sits. If they
drift even slightly from the reference, attention degrades in ways that don't
crash and don't show up in shapes — the model just gets quietly worse at long
range. So we check against HF's own implementation, not against intuition.
"""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


def _hf_reference(rope: dict, head_dim: int):
    from transformers import LlamaConfig
    from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS

    old = rope.get("original_max_position_embeddings", 4096)
    cfg = LlamaConfig(
        hidden_size=head_dim * 32,
        num_attention_heads=32,
        max_position_embeddings=int(old * rope["factor"]),
    )
    cfg.rope_parameters = dict(rope)
    return ROPE_INIT_FUNCTIONS[rope["rope_type"]](cfg, device="cpu")


@pytest.mark.parametrize(
    "rope",
    [
        {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 4.0,
         "original_max_position_embeddings": 4096},
        {"rope_type": "yarn", "rope_theta": 10000.0, "factor": 8.0,
         "original_max_position_embeddings": 8192, "beta_fast": 32, "beta_slow": 1},
        {"rope_type": "yarn", "rope_theta": 500000.0, "factor": 2.0,
         "original_max_position_embeddings": 8192},
        {"rope_type": "linear", "rope_theta": 10000.0, "factor": 2.0},
    ],
)
def test_rope_scaling_matches_hf(rope):
    from inferneo.models.layers import RotaryEmbedding

    head_dim = 128
    hf_inv_freq, hf_attn = _hf_reference(rope, head_dim)
    ours = RotaryEmbedding(head_dim, dict(rope))

    assert torch.allclose(ours.inv_freq, hf_inv_freq, atol=1e-6), "inv_freq drifted from HF"
    assert abs(ours.attention_scaling - hf_attn) < 1e-6, "attention scaling drifted from HF"


def test_yarn_preserves_high_frequencies():
    """YaRN's whole point: leave the fast (local) dimensions alone and only stretch
    the slow ones. Plain linear interpolation squashes everything — that is the
    flaw YaRN exists to fix, so the two must NOT agree on the fast dims."""
    from inferneo.models.layers import RotaryEmbedding

    head_dim, factor = 128, 8.0
    base = {"rope_theta": 10000.0, "factor": factor,
            "original_max_position_embeddings": 4096}
    yarn = RotaryEmbedding(head_dim, {"rope_type": "yarn", **base}).inv_freq
    linear = RotaryEmbedding(head_dim, {"rope_type": "linear", **base}).inv_freq
    plain = RotaryEmbedding(head_dim, {"rope_type": "default", **base}).inv_freq

    # fastest dimension: linear squashes it by `factor`; YaRN leaves it untouched.
    assert torch.isclose(yarn[0], plain[0], rtol=1e-4), "YaRN must not scale the fastest dim"
    assert linear[0] < plain[0] / 2, "linear interpolation should squash even the fast dims"
    # slowest dimension: both interpolate it.
    assert yarn[-1] < plain[-1] / 2, "YaRN must interpolate the slow dims"


def test_unsupported_rope_type_fails_loudly():
    """An unimplemented scaling must raise, never silently fall back to default —
    silently wrong positions produce a model that is subtly bad, not obviously broken."""
    from inferneo.models.layers import RotaryEmbedding

    with pytest.raises(NotImplementedError):
        RotaryEmbedding(128, {"rope_theta": 10000.0, "rope_type": "some_future_thing",
                              "factor": 2.0})
