"""Request state tracked by the scheduler. Control plane: no torch imports.

The central representation (vLLM V1 style): a request is just its token ids
plus a count of how many of them have had KV computed. There is no separate
prefill/decode state — a decode step is simply the case where exactly one
token (the newest) remains to be computed.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field

from inferneo.sampling_params import SamplingParams


class RequestStatus(enum.Enum):
    WAITING = enum.auto()
    RUNNING = enum.auto()
    PREEMPTED = enum.auto()
    FINISHED_STOPPED = enum.auto()
    FINISHED_LENGTH = enum.auto()
    FINISHED_ABORTED = enum.auto()

    @property
    def finished(self) -> bool:
        return self in _FINISHED

    @property
    def finish_reason(self) -> str | None:
        return {
            RequestStatus.FINISHED_STOPPED: "stop",
            RequestStatus.FINISHED_LENGTH: "length",
            RequestStatus.FINISHED_ABORTED: "abort",
        }.get(self)


_FINISHED = {
    RequestStatus.FINISHED_STOPPED,
    RequestStatus.FINISHED_LENGTH,
    RequestStatus.FINISHED_ABORTED,
}


@dataclass
class EngineRequest:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    eos_token_id: int | None = None
    prompt: str | None = None
    # Scheduling priority: higher is served first; ties broken by arrival (FCFS).
    priority: int = 0
    arrival_time: float = field(default_factory=time.monotonic)

    status: RequestStatus = RequestStatus.WAITING
    output_token_ids: list[int] = field(default_factory=list)
    # How many of all_token_ids have had their KV computed.
    num_computed_tokens: int = 0
    # Timestamps for metrics.
    first_scheduled_time: float | None = None
    first_token_time: float | None = None
    finished_time: float | None = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def num_tokens(self) -> int:
        """All tokens known for this request (prompt + generated)."""
        return len(self.prompt_token_ids) + len(self.output_token_ids)

    @property
    def all_token_ids(self) -> list[int]:
        return self.prompt_token_ids + self.output_token_ids

    @property
    def is_finished(self) -> bool:
        return self.status.finished

    def append_output_token(self, token_id: int) -> None:
        self.output_token_ids.append(token_id)

    def check_stop(self, max_model_len: int) -> RequestStatus | None:
        """Return a finished status if the newest token ends the request."""
        params = self.sampling_params
        last = self.output_token_ids[-1]
        if self.num_output_tokens >= params.min_tokens:
            if not params.ignore_eos and self.eos_token_id is not None and last == self.eos_token_id:
                return RequestStatus.FINISHED_STOPPED
            if last in params.stop_token_ids:
                return RequestStatus.FINISHED_STOPPED
        if self.num_output_tokens >= params.max_tokens:
            return RequestStatus.FINISHED_LENGTH
        if self.num_tokens >= max_model_len:
            return RequestStatus.FINISHED_LENGTH
        return None
