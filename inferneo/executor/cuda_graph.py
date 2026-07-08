"""CUDA graphs for the decode step.

In steady-state decode, each engine step runs one token per request through the
whole model — hundreds of tiny kernels whose launch overhead dominates the
actual math. A CUDA graph records that step once and replays it with a single
launch, which is the main lever for closing the gap to vLLM.

Design:
  * Only *pure-decode* steps are graphed (every request advances by one token
    and samples). Prefill / chunked / mixed steps run eager.
  * One graph per batch-size bucket; a real batch of N is padded up to the
    smallest bucket >= N, with padding rows pointed at a reserved scratch block
    so their KV writes and reads are harmless.
  * FlashInfer's decode wrapper runs in CUDA-graph mode (fixed batch size,
    persistent indptr/indices/last-page buffers); ``plan()`` runs on the host
    before each replay to refresh those buffers for the step's real sequence
    lengths, and the captured ``run()`` reads them.
  * The graph produces logits; sampling stays eager outside the graph so every
    sampling mode keeps working.
"""

from __future__ import annotations

import torch

from inferneo.attention.flashinfer_backend import FlashInferMetadata


def default_buckets(max_num_seqs: int) -> list[int]:
    """Powers of two up to max_num_seqs — few enough to bound graph-capture
    time and memory, dense enough that padding waste stays under 2x."""
    buckets, b = [], 1
    while b < max_num_seqs:
        buckets.append(b)
        b *= 2
    buckets.append(max_num_seqs)
    return sorted(dict.fromkeys(buckets))


class _Bucket:
    __slots__ = ("graph", "wrapper", "logits", "indptr", "indices", "last_page")


class CUDAGraphDecodeRunner:
    def __init__(
        self,
        model,
        kv_caches: list[torch.Tensor],
        *,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        block_size: int,
        max_blocks_per_seq: int,
        scratch_block_id: int,
        max_num_seqs: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        import flashinfer

        self._flashinfer = flashinfer
        self.model = model
        self.kv_caches = kv_caches
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.max_blocks_per_seq = max_blocks_per_seq
        self.scratch_block_id = scratch_block_id
        self.device = device
        self.dtype = dtype

        self.buckets = default_buckets(max_num_seqs)
        self.max_bs = self.buckets[-1]

        # Static input buffers (max-sized; sliced per bucket).
        self.g_input_ids = torch.zeros(self.max_bs, dtype=torch.long, device=device)
        self.g_positions = torch.zeros(self.max_bs, dtype=torch.long, device=device)
        self.g_block_ids = torch.zeros(self.max_bs, dtype=torch.long, device=device)
        self.g_offsets = torch.zeros(self.max_bs, dtype=torch.long, device=device)

        self._graphs: dict[int, _Bucket] = {}
        for b in self.buckets:
            self._graphs[b] = self._capture(b)

    def pick_bucket(self, n: int) -> int | None:
        if n > self.max_bs:
            return None
        for b in self.buckets:
            if b >= n:
                return b
        return None

    # ------------------------------------------------------------------ #

    def _make_wrapper(self, b: int) -> _Bucket:
        bucket = _Bucket()
        bucket.indptr = torch.zeros(b + 1, dtype=torch.int32, device=self.device)
        bucket.indices = torch.zeros(
            b * self.max_blocks_per_seq, dtype=torch.int32, device=self.device
        )
        bucket.last_page = torch.zeros(b, dtype=torch.int32, device=self.device)
        workspace = torch.empty(128 * 1024 * 1024, dtype=torch.uint8, device=self.device)
        bucket.wrapper = self._flashinfer.BatchDecodeWithPagedKVCacheWrapper(
            workspace,
            kv_layout="NHD",
            use_cuda_graph=True,
            paged_kv_indptr_buffer=bucket.indptr,
            paged_kv_indices_buffer=bucket.indices,
            paged_kv_last_page_len_buffer=bucket.last_page,
        )
        return bucket

    def _plan(self, bucket: _Bucket, b: int, seq_lens: list[int], block_tables: list[list[int]]):
        """Refresh the wrapper's paged-KV metadata for this step (host side).

        Rows [len(seq_lens), b) are padding: one scratch block, length 1.
        """
        indptr = [0]
        indices: list[int] = []
        last: list[int] = []
        for i in range(b):
            if i < len(seq_lens):
                s = seq_lens[i]
                nb = -(-s // self.block_size)
                indices.extend(block_tables[i][:nb])
                last.append(s - (nb - 1) * self.block_size)
            else:
                nb = 1
                indices.append(self.scratch_block_id)
                last.append(1)
            indptr.append(indptr[-1] + nb)
        bucket.wrapper.plan(
            torch.tensor(indptr, dtype=torch.int32, device=self.device),
            torch.tensor(indices, dtype=torch.int32, device=self.device),
            torch.tensor(last, dtype=torch.int32, device=self.device),
            self.num_heads,
            self.num_kv_heads,
            self.head_dim,
            self.block_size,
            q_data_type=self.dtype,
            kv_data_type=self.dtype,
        )

    def _decode_metadata(self, b: int, bucket: _Bucket) -> FlashInferMetadata:
        return FlashInferMetadata(
            block_ids=self.g_block_ids[:b],
            offsets=self.g_offsets[:b],
            attend=bucket.wrapper.run,
        )

    def _forward(self, b: int, meta: FlashInferMetadata) -> torch.Tensor:
        hidden = self.model(self.g_input_ids[:b], self.g_positions[:b], self.kv_caches, meta)
        return self.model.compute_logits(hidden)

    def _capture(self, b: int) -> _Bucket:
        bucket = self._make_wrapper(b)
        # A valid dummy decode batch: every row length 1 in the scratch block.
        self.g_input_ids[:b].zero_()
        self.g_positions[:b].zero_()
        self.g_block_ids[:b].fill_(self.scratch_block_id)
        self.g_offsets[:b].zero_()
        self._plan(bucket, b, [], [])  # all-padding plan
        meta = self._decode_metadata(b, bucket)

        # Warm up on a side stream (required before capture).
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._forward(b, meta)
        torch.cuda.current_stream().wait_stream(stream)

        bucket.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(bucket.graph):
            bucket.logits = self._forward(b, meta)
        return bucket

    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def run(
        self,
        input_ids: list[int],
        positions: list[int],
        seq_lens: list[int],
        block_tables: list[list[int]],
    ) -> torch.Tensor:
        """Replay the decode graph; returns logits for the N real rows."""
        n = len(input_ids)
        b = self.pick_bucket(n)
        assert b is not None
        bucket = self._graphs[b]

        # Scatter target for each new token: last position of its sequence.
        block_ids, offsets = [], []
        for i in range(b):
            if i < n:
                pos = seq_lens[i] - 1
                block_ids.append(block_tables[i][pos // self.block_size])
                offsets.append(pos % self.block_size)
            else:
                block_ids.append(self.scratch_block_id)
                offsets.append(0)

        pad_ids = input_ids + [0] * (b - n)
        pad_pos = positions + [0] * (b - n)
        self.g_input_ids[:b].copy_(torch.tensor(pad_ids, device=self.device))
        self.g_positions[:b].copy_(torch.tensor(pad_pos, device=self.device))
        self.g_block_ids[:b].copy_(torch.tensor(block_ids, device=self.device))
        self.g_offsets[:b].copy_(torch.tensor(offsets, device=self.device))

        self._plan(bucket, b, seq_lens, block_tables)
        bucket.graph.replay()
        return bucket.logits[:n]
