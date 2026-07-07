"""Choose an attention backend for the device.

Default: FlashInfer on CUDA (fast path), the pure-torch SDPA reference
everywhere else. Force either with ``INFERNEO_ATTENTION=flashinfer|sdpa``
(handy for cross-checking the kernel against the reference on the same GPU).
"""

from __future__ import annotations

import os

import torch

from inferneo.attention.interface import AttentionBackend
from inferneo.attention.sdpa_backend import SDPABackend

# FlashInfer ships kernels only for these attention configs.
_FLASHINFER_HEAD_DIMS = {64, 128, 256}
_FLASHINFER_DTYPES = {torch.float16, torch.bfloat16}


def _flashinfer_available() -> bool:
    try:
        import flashinfer  # noqa: F401

        return True
    except ImportError:
        return False


def _flashinfer_supports(head_dim: int, dtype: torch.dtype) -> bool:
    return head_dim in _FLASHINFER_HEAD_DIMS and dtype in _FLASHINFER_DTYPES


def get_attention_backend(
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> AttentionBackend:
    args = (num_heads, num_kv_heads, head_dim, block_size, device, dtype)
    name = os.environ.get("INFERNEO_ATTENTION", "auto")

    if name == "sdpa":
        return SDPABackend(*args)
    if name == "flashinfer":
        if not _flashinfer_supports(head_dim, dtype):
            raise ValueError(
                f"flashinfer requested but unsupported for head_dim={head_dim}, "
                f"dtype={dtype} (needs head_dim in {sorted(_FLASHINFER_HEAD_DIMS)}, "
                f"fp16/bf16)"
            )
        from inferneo.attention.flashinfer_backend import FlashInferBackend

        return FlashInferBackend(*args)
    if name != "auto":
        raise ValueError(f"unknown attention backend {name!r}")

    if (
        device.type == "cuda"
        and _flashinfer_supports(head_dim, dtype)
        and _flashinfer_available()
    ):
        from inferneo.attention.flashinfer_backend import FlashInferBackend

        return FlashInferBackend(*args)
    return SDPABackend(*args)
