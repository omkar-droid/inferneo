"""The one common internal input. Every transport (REST, WebSocket, later gRPC)
and every modality (text, image, audio) converges on this before the engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class EngineInput:
    """What the engine actually consumes.

    ``token_ids`` is always the full prompt (for an image, the <image> placeholder
    tokens are already expanded, so the length is the true sequence length — the
    KV cache and scheduler need no special cases).

    ``prompt_embeds`` is set only when a modality produced embeddings directly
    (an image). When present it is the embedding for *every* prompt position, so
    the runner can slice any prefill chunk out of it. When absent the runner just
    embeds the token ids as usual.
    """

    token_ids: list[int]
    prompt_embeds: torch.Tensor | None = None  # [len(token_ids), hidden]
    text: str | None = None                    # original text, for echoing back
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.prompt_embeds is not None and self.prompt_embeds.shape[0] != len(self.token_ids):
            raise ValueError(
                f"prompt_embeds has {self.prompt_embeds.shape[0]} rows but there "
                f"are {len(self.token_ids)} tokens — they must line up 1:1"
            )

    @property
    def is_multimodal(self) -> bool:
        return self.prompt_embeds is not None

    def __len__(self) -> int:
        return len(self.token_ids)
