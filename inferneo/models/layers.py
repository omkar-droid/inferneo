"""Shared model layers. Numerics deliberately match HF Transformers so the
correctness suite can demand exact greedy-token equality in fp32."""

from __future__ import annotations

import torch
from torch import nn


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return self.weight * x.to(dtype)


def _llama3_scaled_inv_freq(inv_freq: torch.Tensor, rope_scaling: dict) -> torch.Tensor:
    """Llama-3.x rope scaling (matches HF's ``_compute_llama3_parameters``)."""
    factor = rope_scaling["factor"]
    low_factor = rope_scaling["low_freq_factor"]
    high_factor = rope_scaling["high_freq_factor"]
    old_len = rope_scaling["original_max_position_embeddings"]

    low_wavelen = old_len / low_factor
    high_wavelen = old_len / high_factor
    wavelen = 2 * torch.pi / inv_freq

    scaled = torch.where(wavelen > low_wavelen, inv_freq / factor, inv_freq)
    smooth = (old_len / wavelen - low_factor) / (high_factor - low_factor)
    smoothed = (1 - smooth) / factor * inv_freq + smooth * inv_freq
    is_medium = (wavelen >= high_wavelen) & (wavelen <= low_wavelen)
    return torch.where(is_medium, smoothed, scaled)


def get_rope_parameters(config) -> dict:
    """Unified rope params across transformers 4.x (rope_theta/rope_scaling)
    and 5.x (rope_parameters dict)."""
    params = getattr(config, "rope_parameters", None)
    if params:
        return dict(params)
    out = {
        "rope_theta": getattr(config, "rope_theta", 10000.0),
        "rope_type": "default",
    }
    scaling = getattr(config, "rope_scaling", None)
    if scaling:
        out.update(scaling)
        out["rope_type"] = scaling.get("rope_type", scaling.get("type", "default"))
    return out


class RotaryEmbedding(nn.Module):
    """Rotary position embedding over flat token batches ([num_tokens, H, D])."""

    def __init__(self, head_dim: int, rope_parameters: dict):
        super().__init__()
        base = rope_parameters["rope_theta"]
        inv_freq = 1.0 / (
            base ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        rope_type = rope_parameters.get("rope_type", "default")
        if rope_type == "llama3":
            inv_freq = _llama3_scaled_inv_freq(inv_freq, rope_parameters)
        elif rope_type != "default":
            raise NotImplementedError(f"rope scaling {rope_type!r}")
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(
        self, positions: torch.Tensor, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        freqs = positions.to(torch.float32)[:, None] * self.inv_freq[None, :]
        emb = torch.cat((freqs, freqs), dim=-1)  # [T, D]
        cos = emb.cos().to(q.dtype)[:, None, :]  # [T, 1, D]
        sin = emb.sin().to(q.dtype)[:, None, :]
        return _rotate(q, cos, sin), _rotate(k, cos, sin)


def _rotate(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), dim=-1)
    return x * cos + rotated * sin
