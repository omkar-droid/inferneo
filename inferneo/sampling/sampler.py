"""Token sampler: temperature, top-k/top-p/min-p, penalties, seeds, logprobs.

Phase-1 reference implementation: correctness and portability first. Logits
rows are brought to CPU float32 and processed per request — deterministic
across CUDA/MPS/CPU and trivially auditable. Fully-batched on-device
sampling is a Phase-2 optimization behind the same interface.

Conventions (OpenAI/vLLM-compatible):
- presence/frequency penalties apply to *generated* tokens only;
  repetition penalty applies to prompt + generated tokens.
- returned logprobs are computed from the raw logits (before penalties and
  temperature), like vLLM's default.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F

from inferneo.outputs import TokenLogprob
from inferneo.sampling_params import SamplingParams


@dataclass
class RequestSamplerState:
    """Per-request state the sampler needs across steps."""

    params: SamplingParams
    prompt_len: int
    token_ids: list[int] = field(default_factory=list)  # all known tokens
    generator: torch.Generator | None = None  # set when params.seed is given


class Sampler:
    def __init__(self, seed: int | None = None):
        self._generator = torch.Generator()
        if seed is not None:
            self._generator.manual_seed(seed)

    def sample(
        self, logits: torch.Tensor, states: list[RequestSamplerState]
    ) -> tuple[list[int], list[TokenLogprob | None]]:
        """``logits``: [len(states), vocab], one row per sampling request."""
        rows = logits.detach().to(torch.float32).cpu()
        tokens: list[int] = []
        logprobs: list[TokenLogprob | None] = []
        for row, state in zip(rows, states):
            params = state.params
            raw_logprobs = (
                F.log_softmax(row, dim=-1) if params.logprobs is not None else None
            )
            if params.needs_penalties:
                row = self._apply_penalties(row.clone(), state)
            token = self._pick(row, state)
            tokens.append(token)
            logprobs.append(self._token_logprob(raw_logprobs, token, params))
        return tokens, logprobs

    def _apply_penalties(self, row: torch.Tensor, state: RequestSamplerState) -> torch.Tensor:
        params = state.params
        if params.repetition_penalty != 1.0:
            seen = torch.tensor(sorted(set(state.token_ids)), dtype=torch.long)
            vals = row[seen]
            row[seen] = torch.where(
                vals > 0, vals / params.repetition_penalty, vals * params.repetition_penalty
            )
        output_ids = state.token_ids[state.prompt_len :]
        if output_ids and (params.frequency_penalty or params.presence_penalty):
            ids, counts = torch.tensor(output_ids).unique(return_counts=True)
            row[ids] -= params.frequency_penalty * counts.to(row.dtype)
            row[ids] -= params.presence_penalty
        return row

    def _pick(self, row: torch.Tensor, state: RequestSamplerState) -> int:
        params = state.params
        if params.greedy:
            return int(row.argmax())
        row = row / params.temperature
        row = self._filter(row, params)
        probs = F.softmax(row, dim=-1)
        gen = state.generator or self._generator
        return int(torch.multinomial(probs, 1, generator=gen))

    @staticmethod
    def _filter(row: torch.Tensor, params: SamplingParams) -> torch.Tensor:
        if params.top_k > 0 and params.top_k < row.shape[-1]:
            kth = torch.topk(row, params.top_k).values[-1]
            row = row.masked_fill(row < kth, float("-inf"))
        if params.top_p < 1.0:
            sorted_logits, sorted_idx = row.sort(descending=True)
            cum = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
            # Drop tokens whose *preceding* cumulative mass already reached p.
            drop_sorted = (cum - F.softmax(sorted_logits, dim=-1)) >= params.top_p
            drop = torch.zeros_like(drop_sorted).scatter(-1, sorted_idx, drop_sorted)
            row = row.masked_fill(drop, float("-inf"))
        if params.min_p > 0.0:
            probs = F.softmax(row, dim=-1)
            row = row.masked_fill(probs < params.min_p * probs.max(), float("-inf"))
        return row

    @staticmethod
    def _token_logprob(
        raw_logprobs: torch.Tensor | None, token: int, params: SamplingParams
    ) -> TokenLogprob | None:
        if raw_logprobs is None:
            return None
        top: dict[int, float] = {}
        if params.logprobs:
            vals, idx = torch.topk(raw_logprobs, params.logprobs)
            top = {int(i): float(v) for i, v in zip(idx, vals)}
        return TokenLogprob(token_id=token, logprob=float(raw_logprobs[token]), top=top)
