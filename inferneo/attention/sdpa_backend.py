"""Reference paged-attention backend in pure torch SDPA.

Exact and device-portable (CPU / MPS / CUDA): this is what makes the whole
engine — scheduler, block manager, prefix caching, sampler — testable on a
laptop with the *identical* control flow the CUDA fast path uses. It is not
fast: it gathers each request's KV out of the paged cache and attends
per-request in a Python loop. FlashInfer (Phase 2) replaces the math, not
the interface.

KV cache layout: ``[2, num_blocks * block_size, num_kv_heads, head_dim]`` —
index 0 keys, 1 values, addressed by flat slot id
``block_id * block_size + offset``.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from inferneo.attention.interface import AttentionBackend


@dataclass
class SDPAMetadata:
    query_lens: list[int]
    seq_lens: list[int]
    # Flat slot ids where this step's new tokens are written: [num_tokens].
    slot_mapping: torch.Tensor
    # Per request: flat slot ids covering its full sequence (seq_lens[i]).
    seq_slots: list[torch.Tensor]
    # Per request: causal bool mask [query_len, seq_len] (True = attend).
    masks: list[torch.Tensor | None]


class SDPABackend(AttentionBackend):
    def make_kv_cache(self, num_blocks: int) -> torch.Tensor:
        return torch.zeros(
            2,
            num_blocks * self.block_size,
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
    ) -> SDPAMetadata:
        bs = self.block_size
        slot_chunks: list[torch.Tensor] = []
        seq_slots: list[torch.Tensor] = []
        masks: list[torch.Tensor | None] = []
        for q_len, s_len, table in zip(query_lens, seq_lens, block_tables):
            bt = torch.tensor(table, dtype=torch.long)
            pos = torch.arange(s_len)
            slots = bt[pos // bs] * bs + pos % bs
            seq_slots.append(slots.to(self.device))
            slot_chunks.append(slots[s_len - q_len :])
            if q_len == s_len:
                masks.append(None)  # plain causal, use SDPA's fast path
            else:
                # query token i sits at absolute position s_len - q_len + i
                allowed = torch.arange(s_len) <= (
                    s_len - q_len + torch.arange(q_len)[:, None]
                )
                masks.append(allowed.to(self.device))
        return SDPAMetadata(
            query_lens=query_lens,
            seq_lens=seq_lens,
            slot_mapping=torch.cat(slot_chunks).to(self.device),
            seq_slots=seq_slots,
            masks=masks,
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        metadata: SDPAMetadata,
    ) -> torch.Tensor:
        kv_cache[0][metadata.slot_mapping] = k
        kv_cache[1][metadata.slot_mapping] = v

        n_rep = self.num_heads // self.num_kv_heads
        out = torch.empty_like(q)
        start = 0
        for i, q_len in enumerate(metadata.query_lens):
            q_i = q[start : start + q_len]  # [L, H, D]
            slots = metadata.seq_slots[i]
            k_i = kv_cache[0][slots]  # [S, KVH, D]
            v_i = kv_cache[1][slots]
            if n_rep > 1:
                k_i = k_i.repeat_interleave(n_rep, dim=1)
                v_i = v_i.repeat_interleave(n_rep, dim=1)
            attn = F.scaled_dot_product_attention(
                q_i.permute(1, 0, 2).unsqueeze(0),  # [1, H, L, D]
                k_i.permute(1, 0, 2).unsqueeze(0),  # [1, H, S, D]
                v_i.permute(1, 0, 2).unsqueeze(0),
                attn_mask=None if metadata.masks[i] is None else metadata.masks[i],
                is_causal=metadata.masks[i] is None and q_len > 1,
            )
            out[start : start + q_len] = attn.squeeze(0).permute(1, 0, 2)
            start += q_len
        return out
