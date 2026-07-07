"""Per-request sampling parameters."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SamplingParams:
    """Controls how tokens are sampled for one request.

    temperature=0 means greedy decoding. Penalties follow the OpenAI/vLLM
    convention: presence/frequency apply to generated tokens only, repetition
    penalty applies to prompt + generated tokens.
    """

    max_tokens: int = 128
    min_tokens: int = 0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1  # -1 disables
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop: list[str] = field(default_factory=list)
    stop_token_ids: list[int] = field(default_factory=list)
    ignore_eos: bool = False
    # Number of top logprobs to return per generated token (None = off,
    # 0 = only the sampled token's logprob).
    logprobs: int | None = None

    def __post_init__(self) -> None:
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        if self.min_tokens < 0 or self.min_tokens > self.max_tokens:
            raise ValueError("min_tokens must be in [0, max_tokens]")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if not 0 < self.top_p <= 1:
            raise ValueError("top_p must be in (0, 1]")
        if self.top_k == 0 or self.top_k < -1:
            raise ValueError("top_k must be -1 (disabled) or >= 1")
        if not 0 <= self.min_p <= 1:
            raise ValueError("min_p must be in [0, 1]")
        if not -2 <= self.presence_penalty <= 2:
            raise ValueError("presence_penalty must be in [-2, 2]")
        if not -2 <= self.frequency_penalty <= 2:
            raise ValueError("frequency_penalty must be in [-2, 2]")
        if self.repetition_penalty <= 0:
            raise ValueError("repetition_penalty must be > 0")
        if self.logprobs is not None and self.logprobs < 0:
            raise ValueError("logprobs must be >= 0")

    @property
    def greedy(self) -> bool:
        return self.temperature == 0

    @property
    def needs_penalties(self) -> bool:
        return (
            self.presence_penalty != 0
            or self.frequency_penalty != 0
            or self.repetition_penalty != 1
        )
