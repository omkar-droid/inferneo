"""Seams between the control plane, tensor plane, and serving plane.

Everything in this module is plain data (ids, ints, floats) so a future
process split (ZMQ engine core, vLLM-V1 style) needs no redesign. No torch.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

from inferneo.outputs import TokenLogprob
from inferneo.sampling_params import SamplingParams


@dataclass
class ScheduledRequest:
    """One request's work assignment for a single engine step."""

    request_id: str
    # The token ids to run through the model this step:
    # all_token_ids[start_pos : start_pos + len(chunk_token_ids)].
    chunk_token_ids: list[int]
    start_pos: int  # == num_computed_tokens before this step
    block_ids: list[int]  # full block table for the request
    # True when this chunk reaches the end of the request's known tokens,
    # i.e. the model should sample a new token from the final position.
    do_sample: bool
    # True the first time the runner sees this request (or again after a
    # preemption); the runner must (re)register per-request sampling state.
    is_new: bool
    sampling_params: SamplingParams | None = None  # set when is_new
    prompt_len: int = 0  # set when is_new; penalties split prompt vs output
    # Set when is_new and start_pos > 0 (prefix-cache hit or resumed after
    # preemption): the tokens before start_pos, so the runner's penalty
    # tracking sees the full history it never ran through the model.
    cached_prefix_token_ids: list[int] = field(default_factory=list)

    @property
    def num_new_tokens(self) -> int:
        return len(self.chunk_token_ids)


@dataclass
class SchedulerOutput:
    """Everything the model runner needs to execute one step."""

    scheduled: list[ScheduledRequest] = field(default_factory=list)
    # Requests preempted this step: runner drops their state.
    preempted_ids: list[str] = field(default_factory=list)
    # Requests that finished since the previous step: runner drops their state.
    finished_ids: list[str] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(s.num_new_tokens for s in self.scheduled)


@dataclass
class ModelRunnerOutput:
    """Per-step results, already synced to CPU (the one sync point per step)."""

    # request_id -> sampled token id, for requests scheduled with do_sample.
    sampled: dict[str, int] = field(default_factory=dict)
    # request_id -> logprob info for the sampled token (when requested).
    logprobs: dict[str, TokenLogprob] = field(default_factory=dict)


class ModelRunner(Protocol):
    """Tensor-plane seam. A JAX/TensorRT backend is a new implementation of
    this protocol; the scheduler and engine never import torch."""

    def load_model(self) -> None: ...

    def init_kv_cache(self) -> int:
        """Allocate the paged KV cache; returns the number of blocks."""
        ...

    def execute(self, scheduler_output: SchedulerOutput) -> ModelRunnerOutput: ...
