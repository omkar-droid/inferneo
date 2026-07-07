"""Per-request block tables over the shared block pool.

Control plane: no torch imports. This is the module a KV-cache research idea
usually edits: allocation policy, prefix reuse, and (later) radix indexes or
KV connectors all live behind this class.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from inferneo.kv.block_pool import BlockPool, KVBlock
from inferneo.kv.hashing import BlockHash, hash_block

if TYPE_CHECKING:
    from inferneo.engine.request import EngineRequest


class KVCacheManager:
    def __init__(self, num_blocks: int, block_size: int, enable_caching: bool = False):
        self.block_size = block_size
        self.enable_caching = enable_caching
        self.pool = BlockPool(num_blocks, enable_caching)
        self._req_blocks: dict[str, list[KVBlock]] = {}
        # Hash chain of this request's leading full blocks (for caching).
        self._req_hashes: dict[str, list[BlockHash]] = {}

    def get_block_ids(self, request_id: str) -> list[int]:
        return [b.block_id for b in self._req_blocks.get(request_id, [])]

    def usage(self) -> float:
        return self.pool.usage()

    def get_computed_blocks(self, req: EngineRequest) -> tuple[list[KVBlock], int]:
        """Longest run of already-cached leading blocks for a new request.

        Takes a reference on each returned block. Always leaves at least one
        token uncomputed so the model has a position to produce logits from.
        """
        if not self.enable_caching:
            return [], 0
        tokens = req.all_token_ids
        blocks: list[KVBlock] = []
        parent: BlockHash | None = None
        for start in range(0, len(tokens) - self.block_size + 1, self.block_size):
            parent = hash_block(parent, tokens[start : start + self.block_size])
            block = self.pool.get_cached_block(parent)
            if block is None:
                break
            blocks.append(block)
        if blocks and len(blocks) * self.block_size == len(tokens):
            self.pool.free(blocks.pop())
        return blocks, len(blocks) * self.block_size

    def allocate_slots(
        self,
        req: EngineRequest,
        num_new_tokens: int,
        new_computed_blocks: list[KVBlock] | None = None,
    ) -> list[KVBlock] | None:
        """Ensure the request's block table covers its tokens after this step.

        ``new_computed_blocks`` are prefix-cache hits from
        :meth:`get_computed_blocks`, attached ahead of freshly allocated
        blocks (admission only). Returns the newly allocated blocks, or None
        if the pool cannot satisfy the request — in that case any
        ``new_computed_blocks`` references are released.
        """
        new_computed = new_computed_blocks or []
        cur_blocks = self._req_blocks.get(req.request_id, [])
        assert not (cur_blocks and new_computed), "cache hits only attach at admission"

        num_computed = req.num_computed_tokens + len(new_computed) * self.block_size
        total_tokens = num_computed + num_new_tokens
        total_blocks = -(-total_tokens // self.block_size)  # ceil div
        num_fresh = total_blocks - len(cur_blocks) - len(new_computed)

        fresh: list[KVBlock] = []
        if num_fresh > 0:
            allocated = self.pool.allocate(num_fresh)
            if allocated is None:
                for b in new_computed:
                    self.pool.free(b)
                return None
            fresh = allocated

        blocks = cur_blocks + new_computed + fresh
        self._req_blocks[req.request_id] = blocks
        if new_computed:
            self._req_hashes[req.request_id] = [
                b.block_hash for b in new_computed if b.block_hash is not None
            ]
        if self.enable_caching:
            self._cache_full_blocks(req, total_tokens)
        return fresh

    def free(self, req: EngineRequest) -> None:
        """Release all blocks. Reverse order so deep suffixes evict first."""
        for block in reversed(self._req_blocks.pop(req.request_id, [])):
            self.pool.free(block)
        self._req_hashes.pop(req.request_id, None)

    def _cache_full_blocks(self, req: EngineRequest, num_known_tokens: int) -> None:
        """Hash-and-register blocks that became full as of this step.

        All ``num_known_tokens`` token ids are known now (prompt tokens or
        previously sampled ones), so their full blocks are immutable and
        shareable.
        """
        blocks = self._req_blocks[req.request_id]
        hashes = self._req_hashes.setdefault(req.request_id, [])
        tokens = req.all_token_ids
        num_full = num_known_tokens // self.block_size
        for i in range(len(hashes), num_full):
            parent = hashes[-1] if hashes else None
            h = hash_block(parent, tokens[i * self.block_size : (i + 1) * self.block_size])
            hashes.append(h)
            self.pool.cache_block(blocks[i], h)
