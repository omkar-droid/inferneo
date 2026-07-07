"""Attention backend seam.

A backend owns (a) the physical layout of the paged KV cache tensor and
(b) the attention computation over block tables. Adding a kernel (FlashInfer,
flash-attn, a custom Triton kernel) means implementing this interface —
nothing in the engine or models changes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import torch


class AttentionBackend(ABC):
    def __init__(
        self,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = device
        self.dtype = dtype
        self.scale = head_dim**-0.5

    @abstractmethod
    def make_kv_cache(self, num_blocks: int) -> torch.Tensor:
        """Allocate one layer's paged KV cache in this backend's layout."""

    @abstractmethod
    def build_metadata(
        self,
        query_lens: list[int],
        seq_lens: list[int],
        block_tables: list[list[int]],
    ) -> Any:
        """Precompute per-step indexing shared by every layer.

        ``query_lens[i]`` new tokens are being computed for request i, whose
        total KV length after this step is ``seq_lens[i]``, stored in the
        blocks listed in ``block_tables[i]``.
        """

    @abstractmethod
    def forward(
        self,
        q: torch.Tensor,  # [num_tokens, num_heads, head_dim]
        k: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        v: torch.Tensor,  # [num_tokens, num_kv_heads, head_dim]
        kv_cache: torch.Tensor,
        metadata: Any,
    ) -> torch.Tensor:  # [num_tokens, num_heads, head_dim]
        """Scatter k/v into the cache at the new tokens' slots, then attend
        causally over each request's full cached sequence."""
