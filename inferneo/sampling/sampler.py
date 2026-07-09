"""Token sampler: temperature, top-k/top-p/min-p, penalties, seeds, logprobs.

Fully batched on-device: the whole step's logits stay on the GPU and are
processed as one [batch, vocab] tensor, with a single small sync to read the
sampled ids back. No per-request CPU round-trip, no Python per-row loop on the
hot path.

Sampling uses the Gumbel-max trick — ``argmax(masked_logits + gumbel_noise)`` is
an exact categorical draw — which keeps sampling batched *and* lets a seeded
request reproduce its own noise row independently of the rest of the batch.

Conventions (OpenAI/vLLM-compatible):
- presence/frequency penalties apply to generated tokens only; repetition
  penalty applies to prompt + generated tokens.
- returned logprobs are computed from the raw logits, before penalties and
  temperature.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch

from inferneo.outputs import TokenLogprob
from inferneo.sampling_params import SamplingParams

_NEG_INF = float("-inf")


@dataclass
class RequestSamplerState:
    """Per-request state the sampler needs across steps."""

    params: SamplingParams
    prompt_len: int
    token_ids: list[int] = field(default_factory=list)  # all known tokens
    generator: torch.Generator | None = None  # set when params.seed is given


class Sampler:
    def __init__(self, seed: int | None = None):
        self._seed = seed
        self._generators: dict[torch.device, torch.Generator] = {}

    def _generator(self, device: torch.device) -> torch.Generator:
        gen = self._generators.get(device)
        if gen is None:
            gen = torch.Generator(device=device)
            if self._seed is not None:
                gen.manual_seed(self._seed)
            self._generators[device] = gen
        return gen

    @torch.inference_mode()
    def sample(
        self, logits: torch.Tensor, states: list[RequestSamplerState]
    ) -> tuple[list[int], list[TokenLogprob | None]]:
        """``logits``: [len(states), vocab], one row per sampling request."""
        params = [s.params for s in states]

        # Fast path: whole batch greedy, no penalties, no logprobs — argmax and
        # a single sync, skipping even the float32 upcast and softmax.
        if all(p.greedy and p.logprobs is None and not p.needs_penalties for p in params):
            return logits.argmax(dim=-1).tolist(), [None] * len(states)

        device = logits.device
        logits = logits.to(torch.float32)

        want_logprobs = any(p.logprobs is not None for p in params)
        raw_logprobs = torch.log_softmax(logits, dim=-1) if want_logprobs else None

        if any(p.needs_penalties for p in params):
            logits = self._apply_penalties(logits, states)

        temps = torch.tensor(
            [1.0 if p.greedy else p.temperature for p in params],
            device=device, dtype=torch.float32,
        )
        logits = logits / temps.unsqueeze(1)
        logits = self._filter(logits, params, device)

        tokens_t = self._pick(logits, states, device)
        tokens = tokens_t.tolist()

        logprobs = self._logprobs(raw_logprobs, tokens_t, params)
        return tokens, logprobs

    # ------------------------------------------------------------------ #

    def _apply_penalties(
        self, logits: torch.Tensor, states: list[RequestSamplerState]
    ) -> torch.Tensor:
        """Per-request penalties. Loops only over penalized rows, but every op
        stays on-device (no host transfer)."""
        device = logits.device
        for i, state in enumerate(states):
            p = state.params
            if not p.needs_penalties:
                continue
            row = logits[i]
            if p.repetition_penalty != 1.0 and state.token_ids:
                seen = torch.tensor(sorted(set(state.token_ids)), device=device)
                vals = row[seen]
                row[seen] = torch.where(
                    vals > 0, vals / p.repetition_penalty, vals * p.repetition_penalty
                )
            output_ids = state.token_ids[state.prompt_len :]
            if output_ids and (p.frequency_penalty or p.presence_penalty):
                ids, counts = torch.tensor(output_ids, device=device).unique(return_counts=True)
                row[ids] -= p.frequency_penalty * counts.to(row.dtype)
                row[ids] -= p.presence_penalty
        return logits

    @staticmethod
    def _filter(
        logits: torch.Tensor, params: list[SamplingParams], device: torch.device
    ) -> torch.Tensor:
        vocab = logits.shape[-1]
        top_k = torch.tensor(
            [p.top_k if 0 < p.top_k < vocab else vocab for p in params], device=device
        )
        top_p = torch.tensor([p.top_p for p in params], device=device)
        min_p = torch.tensor([p.min_p for p in params], device=device)

        if bool((top_k < vocab).any()) or bool((top_p < 1.0).any()):
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            ranks = torch.arange(vocab, device=device).unsqueeze(0)
            drop = ranks >= top_k.unsqueeze(1)  # top-k: drop rank >= k
            if bool((top_p < 1.0).any()):
                probs = sorted_logits.softmax(dim=-1)
                # Drop tokens whose *preceding* cumulative mass already reached p
                # (keeps at least the top token).
                preceding = probs.cumsum(dim=-1) - probs
                drop |= preceding >= top_p.unsqueeze(1)
            drop = torch.zeros_like(drop).scatter(-1, sorted_idx, drop)
            logits = logits.masked_fill(drop, _NEG_INF)

        if bool((min_p > 0.0).any()):
            probs = logits.softmax(dim=-1)
            thresh = min_p.unsqueeze(1) * probs.amax(dim=-1, keepdim=True)
            logits = logits.masked_fill(probs < thresh, _NEG_INF)
        return logits

    def _pick(
        self, logits: torch.Tensor, states: list[RequestSamplerState], device: torch.device
    ) -> torch.Tensor:
        """Gumbel-max categorical draw for sampled rows, argmax for greedy rows."""
        noise = torch.rand(logits.shape, device=device, generator=self._generator(device))
        for i, state in enumerate(states):
            if state.params.seed is not None and state.generator is not None:
                # Reproduce this row's noise from its own generator (may be a CPU
                # generator); everything else stays batched.
                row = torch.rand(logits.shape[-1], generator=state.generator)
                noise[i] = row.to(device)
        gumbel = -torch.log(-torch.log(noise.clamp_min(1e-20)))
        sampled = (logits + gumbel).argmax(dim=-1)
        greedy = (logits.argmax(dim=-1))
        greedy_mask = torch.tensor(
            [s.params.greedy for s in states], device=device, dtype=torch.bool
        )
        return torch.where(greedy_mask, greedy, sampled)

    @staticmethod
    def _logprobs(
        raw_logprobs: torch.Tensor | None,
        tokens: torch.Tensor,
        params: list[SamplingParams],
    ) -> list[TokenLogprob | None]:
        if raw_logprobs is None:
            return [None] * len(params)
        chosen = raw_logprobs.gather(1, tokens.unsqueeze(1)).squeeze(1).tolist()
        max_k = max((p.logprobs or 0) for p in params)
        top_vals, top_idx = (
            torch.topk(raw_logprobs, max_k, dim=-1) if max_k else (None, None)
        )
        out: list[TokenLogprob | None] = []
        for i, p in enumerate(params):
            if p.logprobs is None:
                out.append(None)
                continue
            top: dict[int, float] = {}
            if p.logprobs:
                vals = top_vals[i, : p.logprobs].tolist()
                idx = top_idx[i, : p.logprobs].tolist()
                top = {int(t): float(v) for t, v in zip(idx, vals)}
            out.append(TokenLogprob(token_id=int(tokens[i]), logprob=float(chosen[i]), top=top))
        return out
