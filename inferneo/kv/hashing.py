"""Block hash chain for prefix caching. Control plane: no torch imports.

A block's hash commits to its own tokens *and* everything before it via the
parent hash, so equal hashes mean equal full prefixes. Only completely full
blocks are ever hashed — partial blocks are private to their request.
"""

from __future__ import annotations

from collections.abc import Sequence

BlockHash = int


def hash_block(parent_hash: BlockHash | None, token_ids: Sequence[int]) -> BlockHash:
    return hash((parent_hash, tuple(token_ids)))


def hash_prompt_blocks(token_ids: Sequence[int], block_size: int) -> list[BlockHash]:
    """Hashes for every *full* block of a token sequence, chained left to right."""
    hashes: list[BlockHash] = []
    parent: BlockHash | None = None
    for start in range(0, len(token_ids) - block_size + 1, block_size):
        parent = hash_block(parent, token_ids[start : start + block_size])
        hashes.append(parent)
    return hashes
