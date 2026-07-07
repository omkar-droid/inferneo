"""Choose an attention backend for the device.

Phase 1 ships the SDPA reference everywhere. Phase 2 adds FlashInfer on
CUDA (selected here), with SDPA staying available via
``INFERNEO_ATTENTION=sdpa`` for cross-checking kernels.
"""

from __future__ import annotations

import os

import torch

from inferneo.attention.interface import AttentionBackend
from inferneo.attention.sdpa_backend import SDPABackend


def get_attention_backend(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AttentionBackend:
    name = os.environ.get("INFERNEO_ATTENTION", "auto")
    if name in ("auto", "sdpa"):
        return SDPABackend(num_heads, num_kv_heads, head_dim, block_size, device, dtype)
    raise ValueError(f"unknown attention backend {name!r}")
