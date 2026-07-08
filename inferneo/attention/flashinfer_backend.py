"""FlashInfer paged-attention backend (CUDA fast path).

One ``BatchPrefillWithPagedKVCacheWrapper`` serves every step — prefill,
decode, and mixed batches alike — which is the natural fit for the unified
scheduler: with ``causal=True`` FlashInfer aligns each request's query tokens
to the tail of its cached sequence, exactly how appended tokens sit. GQA is
handled by the kernel (no head repetition), and RoPE is pre-applied by the
model, so ``pos_encoding_mode`` stays NONE.

KV cache layout (NHD, combined): ``[num_blocks, 2, page_size, num_kv_heads,
head_dim]`` — index 1 of dim 1 selects keys/values, addressable by the flat
slot id ``block_id * page_size + offset`` for scatter.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch

from inferneo.attention.interface import AttentionBackend


@dataclass
class FlashInferMetadata:
    # Physical (block_id, offset) for each new token's k/v this step.
    block_ids: torch.Tensor  # [num_tokens]
    offsets: torch.Tensor  # [num_tokens]
    # attend(q, kv_cache) -> attention output. Carried here (rather than a
    # fixed self._wrapper) so the CUDA-graph decode path can substitute a
    # graph-mode decode wrapper without the model knowing.
    attend: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class FlashInferBackend(AttentionBackend):
    _WORKSPACE_BYTES = 128 * 1024 * 1024

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import flashinfer

        self._workspace = torch.empty(
            self._WORKSPACE_BYTES, dtype=torch.uint8, device=self.device
        )
        self._wrapper = flashinfer.BatchPrefillWithPagedKVCacheWrapper(
            self._workspace, kv_layout="NHD"
        )

    def make_kv_cache(self, num_blocks: int) -> torch.Tensor:
        return torch.zeros(
            num_blocks,
            2,
            self.block_size,
            self.num_kv_heads,
            self.head_dim,
            dtype=self.dtype,
            device=self.device,
        )

    def build_metadata(
        self,
        query_lens: list[int],
        seq_lens: list[int],
        block_tables: list[list[int]],
    ) -> FlashInferMetadata:
        bs = self.block_size
        dev = self.device

        qo_indptr = [0]
        kv_indptr = [0]
        kv_indices: list[int] = []
        last_page_len: list[int] = []
        block_chunks: list[torch.Tensor] = []
        offset_chunks: list[torch.Tensor] = []
        for q_len, s_len, table in zip(query_lens, seq_lens, block_tables):
            qo_indptr.append(qo_indptr[-1] + q_len)
            num_pages = -(-s_len // bs)  # ceil
            kv_indices.extend(table[:num_pages])
            kv_indptr.append(kv_indptr[-1] + num_pages)
            last_page_len.append(s_len - (num_pages - 1) * bs)
            # Physical location of the q_len new tokens (tail of the sequence).
            bt = torch.tensor(table, dtype=torch.long)
            pos = torch.arange(s_len - q_len, s_len)
            block_chunks.append(bt[pos // bs])
            offset_chunks.append(pos % bs)

        self._wrapper.plan(
            torch.tensor(qo_indptr, dtype=torch.int32, device=dev),
            torch.tensor(kv_indptr, dtype=torch.int32, device=dev),
            torch.tensor(kv_indices, dtype=torch.int32, device=dev),
            torch.tensor(last_page_len, dtype=torch.int32, device=dev),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            causal=True,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )
        return FlashInferMetadata(
            block_ids=torch.cat(block_chunks).to(dev),
            offsets=torch.cat(offset_chunks).to(dev),
            attend=self._wrapper.run,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: FlashInferMetadata,
    ) -> torch.Tensor:
        # Scatter new k/v into their pages: kv_cache[block, 0|1, offset].
        kv_cache[metadata.block_ids, 0, metadata.offsets] = k
        kv_cache[metadata.block_ids, 1, metadata.offsets] = v
        return metadata.attend(q, kv_cache)
