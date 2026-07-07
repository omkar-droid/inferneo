"""Public output types returned by the engine."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TokenLogprob:
    """Logprob info for one position: the sampled token and top alternatives."""

    token_id: int
    logprob: float
    top: dict[int, float] = field(default_factory=dict)  # token_id -> logprob


@dataclass
class CompletionOutput:
    """One generated completion."""

    index: int
    text: str
    token_ids: list[int]
    finish_reason: str | None = None  # "stop" | "length" | "abort" | None (in progress)
    logprobs: list[TokenLogprob] | None = None
    cumulative_logprob: float | None = None


@dataclass
class RequestMetrics:
    """Wall-clock timestamps for one request (seconds, time.monotonic)."""

    arrival_time: float = 0.0
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    finished_time: float | None = None


@dataclass
class RequestOutput:
    """Everything the engine returns for one request."""

    request_id: str
    prompt: str | None
    prompt_token_ids: list[int]
    outputs: list[CompletionOutput]
    finished: bool
    metrics: RequestMetrics = field(default_factory=RequestMetrics)
