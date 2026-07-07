"""Physical KV block pool with refcounts and cached-block reuse.

Control plane: no torch imports. Blocks here are bookkeeping objects; the
tensor-plane runner owns the actual KV tensors and indexes them by block_id.

Prefix-caching design (vLLM-V1 style): a freed block keeps its hash and sits
in an LRU free queue. It can be revived on a prefix hit any time before it is
reallocated; eviction (dropping the hash) happens lazily only when the block
is handed out for new content.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field

from inferneo.kv.hashing import BlockHash


@dataclass
class KVBlock:
    block_id: int
    ref_count: int = 0
    block_hash: BlockHash | None = field(default=None, repr=False)


class BlockPool:
    def __init__(self, num_blocks: int, enable_caching: bool = False):
        if num_blocks <= 0:
            raise ValueError("num_blocks must be positive")
        self.num_blocks = num_blocks
        self.enable_caching = enable_caching
        self.blocks = [KVBlock(i) for i in range(num_blocks)]
        # LRU free queue: insertion order = eviction order.
        self._free: OrderedDict[int, KVBlock] = OrderedDict(
            (b.block_id, b) for b in self.blocks
        )
        # hash -> block, for prefix reuse. Includes blocks in the free queue.
        self._cached: dict[BlockHash, KVBlock] = {}

    @property
    def num_free_blocks(self) -> int:
        return len(self._free)

    def usage(self) -> float:
        return 1.0 - len(self._free) / self.num_blocks

    def allocate(self, n: int) -> list[KVBlock] | None:
        """Take ``n`` fresh blocks, or None if not enough are free."""
        if n > len(self._free):
            return None
        out = []
        for _ in range(n):
            _, block = self._free.popitem(last=False)  # LRU end
            if block.block_hash is not None:
                # Lazy eviction: this cached content is now gone for good.
                self._cached.pop(block.block_hash, None)
                block.block_hash = None
            block.ref_count = 1
            out.append(block)
        return out

    def free(self, block: KVBlock) -> None:
        if block.ref_count <= 0:
            raise ValueError(f"double free of block {block.block_id}")
        block.ref_count -= 1
        if block.ref_count == 0:
            self._free[block.block_id] = block  # most-recently-used end

    def get_cached_block(self, block_hash: BlockHash) -> KVBlock | None:
        """Look up a full block by content hash and take a reference to it."""
        if not self.enable_caching:
            return None
        block = self._cached.get(block_hash)
        if block is None:
            return None
        if block.ref_count == 0:
            # Revive from the free queue.
            del self._free[block.block_id]
        block.ref_count += 1
        return block

    def cache_block(self, block: KVBlock, block_hash: BlockHash) -> None:
        """Register a now-full block's content hash for future reuse."""
        if not self.enable_caching or block.block_hash is not None:
            return
        if block_hash in self._cached:
            return  # first writer wins; this block stays private
        block.block_hash = block_hash
        self._cached[block_hash] = block
