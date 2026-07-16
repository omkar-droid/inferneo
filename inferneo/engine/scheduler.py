"""Unified token-budget scheduler (vLLM-V1 style). No torch imports.

There is no prefill phase and no decode phase. Every step, each request is
assigned ``num_new_tokens = num_tokens - num_computed_tokens`` capped by the
shared token budget: a decode is 1 token, a fresh prompt is N, a chunked
prefill is whatever fits. Mixed batches fall out for free, and chunked
prefill is implicit.

Scheduling policy: priority, then FCFS. RUNNING requests are served first (a
decode never starves behind new prompts); WAITING requests are admitted in
priority order (higher priority first, arrival time as tiebreak) while budget and
KV blocks remain — so an interactive request jumps ahead of a queued batch job
and is admitted as soon as a slot frees. Under memory pressure the most recently
admitted running request is preempted — its blocks are freed and it re-queues for
recompute-on-resume (with prefix caching on, resume hits the cache for any blocks
not yet evicted).

Priority affects *admission order*, not preemption: a high-priority arrival waits
for the next freed slot rather than evicting a running low-priority request.
Preemptive priority would be the next step.
"""

from __future__ import annotations

import time

from inferneo.config import SchedulerConfig
from inferneo.engine.interfaces import ModelRunnerOutput, ScheduledRequest, SchedulerOutput
from inferneo.engine.request import EngineRequest, RequestStatus
from inferneo.kv.block_manager import KVCacheManager


class Scheduler:
    def __init__(
        self,
        config: SchedulerConfig,
        kv: KVCacheManager,
        max_model_len: int,
    ):
        self.config = config
        self.kv = kv
        self.max_model_len = max_model_len

        self.waiting: list[EngineRequest] = []
        self.running: list[EngineRequest] = []
        self.requests: dict[str, EngineRequest] = {}
        # Finished/aborted since the last schedule(); reported to the runner
        # so it can drop per-request state.
        self._newly_finished: list[str] = []

    # ------------------------------------------------------------------ #
    # request lifecycle
    # ------------------------------------------------------------------ #

    def add_request(self, req: EngineRequest) -> None:
        if req.request_id in self.requests:
            raise ValueError(f"duplicate request_id {req.request_id!r}")
        self.requests[req.request_id] = req
        self.waiting.append(req)

    def abort(self, request_id: str) -> None:
        req = self.requests.get(request_id)
        if req is None or req.is_finished:
            return
        if req.status == RequestStatus.RUNNING:
            self.running.remove(req)
            self.kv.free(req)
        elif req in self.waiting:
            self.waiting.remove(req)
        self._finish(req, RequestStatus.FINISHED_ABORTED)

    def has_unfinished(self) -> bool:
        return bool(self.waiting or self.running)

    # ------------------------------------------------------------------ #
    # the scheduling step
    # ------------------------------------------------------------------ #

    def schedule(self) -> SchedulerOutput:
        out = SchedulerOutput(finished_ids=self._newly_finished)
        self._newly_finished = []
        budget = self.config.max_num_batched_tokens
        preempted: list[EngineRequest] = []

        # -- 1. running requests, FCFS -------------------------------------
        idx = 0
        while idx < len(self.running) and budget > 0:
            req = self.running[idx]
            num_new = min(req.num_tokens - req.num_computed_tokens, budget)
            assert num_new > 0, "running request with nothing to compute"

            while (new_blocks := self.kv.allocate_slots(req, num_new)) is None:
                # Out of KV blocks: preempt the most recently admitted
                # running request (never one already scheduled this step —
                # those sit at positions < idx).
                victim = self.running.pop()
                self._preempt(victim)
                preempted.append(victim)
                if victim is req:
                    break
            if req.status == RequestStatus.PREEMPTED:
                break  # req was its own victim; everything behind it is gone
            del new_blocks  # table is fetched fresh below
            budget -= num_new
            out.scheduled.append(self._make_scheduled(req, num_new, is_new=False))
            idx += 1

        # -- 2. waiting requests (skip if we just preempted: no churn) ------
        # Priority order: highest priority first, arrival time (FCFS) as tiebreak.
        # Sorting each step keeps the scheduler a plain list (readable, torch-free)
        # and is cheap for realistic queue depths.
        if not preempted and self.waiting:
            self.waiting.sort(key=lambda r: (-r.priority, r.arrival_time))
        while (
            not preempted
            and self.waiting
            and budget > 0
            and len(self.running) < self.config.max_num_seqs
        ):
            req = self.waiting[0]
            cached_blocks, num_cached = self.kv.get_computed_blocks(req)
            num_new = req.num_tokens - num_cached
            if self.config.long_prefill_token_threshold > 0:
                num_new = min(num_new, self.config.long_prefill_token_threshold)
            num_new = min(num_new, budget)
            assert num_new > 0

            if self.kv.allocate_slots(req, num_new, cached_blocks) is None:
                if not self.running and not out.scheduled:
                    raise RuntimeError(
                        f"request {req.request_id!r} needs more KV blocks than the "
                        f"pool holds even with nothing else running; increase "
                        f"cache num_blocks or reduce max_num_batched_tokens"
                    )
                break  # top-priority request can't fit — it waits (don't skip it)
            self.waiting.pop(0)
            req.status = RequestStatus.RUNNING
            req.num_computed_tokens = num_cached
            if req.first_scheduled_time is None:
                req.first_scheduled_time = time.monotonic()
            self.running.append(req)
            budget -= num_new
            out.scheduled.append(self._make_scheduled(req, num_new, is_new=True))

        out.preempted_ids = [r.request_id for r in preempted]
        return out

    def update_from_output(
        self, scheduler_output: SchedulerOutput, runner_output: ModelRunnerOutput
    ) -> list[EngineRequest]:
        """Advance request state after a model step.

        Returns requests that produced a new token (including any that
        finished), for the engine to stream/report.
        """
        updated: list[EngineRequest] = []
        now = time.monotonic()
        for sched in scheduler_output.scheduled:
            req = self.requests.get(sched.request_id)
            if req is None or req.is_finished:
                continue  # aborted between schedule and update
            req.num_computed_tokens += sched.num_new_tokens
            if not sched.do_sample:
                continue  # mid-prefill chunk, nothing sampled
            token = runner_output.sampled[req.request_id]
            req.append_output_token(token)
            if req.first_token_time is None:
                req.first_token_time = now
            stop = req.check_stop(self.max_model_len)
            if stop is not None:
                self.running.remove(req)
                self.kv.free(req)
                self._finish(req, stop)
            updated.append(req)
        return updated

    # ------------------------------------------------------------------ #
    # internals
    # ------------------------------------------------------------------ #

    def _preempt(self, req: EngineRequest) -> None:
        self.kv.free(req)
        req.status = RequestStatus.PREEMPTED
        req.num_computed_tokens = 0
        # Re-queue for recompute-on-resume; the priority sort in schedule() puts it
        # back near the front (its original arrival_time is preserved).
        self.waiting.append(req)

    def _finish(self, req: EngineRequest, status: RequestStatus) -> None:
        req.status = status
        req.finished_time = time.monotonic()
        self._newly_finished.append(req.request_id)
        del self.requests[req.request_id]

    def _make_scheduled(
        self, req: EngineRequest, num_new: int, is_new: bool
    ) -> ScheduledRequest:
        start = req.num_computed_tokens
        all_tokens = req.all_token_ids
        return ScheduledRequest(
            request_id=req.request_id,
            chunk_token_ids=all_tokens[start : start + num_new],
            start_pos=start,
            block_ids=self.kv.get_block_ids(req.request_id),
            do_sample=(start + num_new) == req.num_tokens,
            is_new=is_new,
            sampling_params=req.sampling_params if is_new else None,
            prompt_len=req.num_prompt_tokens if is_new else 0,
            cached_prefix_token_ids=all_tokens[:start] if is_new else [],
        )
