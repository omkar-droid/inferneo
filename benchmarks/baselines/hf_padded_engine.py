"""Padded continuous-batching baseline — owns the decode loop over an HF model.

This is the honest internal baseline the paged inferneo engine is benchmarked
against. It implements token-level continuous batching: requests are admitted
into a running batch, evicted the moment they finish, and freed slots are
backfilled from the waiting queue — so the GPU stays busy across a stream of
ragged requests instead of waiting for the slowest sequence in a static batch.

Mechanics:
  * batched forward passes with explicit left-padding + position_ids
  * ``index_select`` eviction of finished rows from the KV cache
  * admission by prefilling *only the new* requests and merging their KV into
    the running batch (no re-prefill of in-flight sequences)

The KV merge pads the shorter cache along the sequence axis and concatenates
along the batch axis. It is verified to match a jointly-prefilled batch, and
greedy decoding matches per-sequence generation token-for-token. The paged
inferneo engine replaces these padded merges with block tables to avoid the
padding waste.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def pick_device() -> str:
    """Best available device: cuda > mps > cpu."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@dataclass
class Request:
    """A single generation request tracked through the batch."""

    id: int
    prompt: str
    max_new_tokens: int = 64
    prompt_ids: list[int] = field(default_factory=list)
    generated: list[int] = field(default_factory=list)
    done: bool = False
    t_admit: float = 0.0
    t_finish: float = 0.0


class ContinuousBatchingEngine:
    """Token-level continuous batching over a HuggingFace causal LM."""

    def __init__(self, model_name: str = "gpt2", max_batch_size: int = 32,
                 device: str | None = None, dtype: torch.dtype | None = None):
        device = device or pick_device()
        if dtype is None:
            dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device).eval()
        )
        self.device = device
        self.max_batch_size = max_batch_size
        self.eos_id = self.tokenizer.eos_token_id
        self.pad_id = self.tokenizer.pad_token_id

    @torch.no_grad()
    def _prefill(self, batch: list[Request]):
        """Prefill a set of requests into a fresh batched KV cache."""
        seqs = [r.prompt_ids + r.generated for r in batch]
        maxlen = max(len(s) for s in seqs)
        ids, mask = [], []
        for s in seqs:
            pad = maxlen - len(s)
            ids.append([self.pad_id] * pad + s)
            mask.append([0] * pad + [1] * len(s))
        ids = torch.tensor(ids, device=self.device)
        mask = torch.tensor(mask, device=self.device)
        pos = (mask.cumsum(-1) - 1).clamp(min=0)
        out = self.model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        return out.past_key_values, mask, pos[:, -1:], nxt

    @torch.no_grad()
    def _decode(self, state):
        """Advance every active sequence by one token."""
        cache, mask, cur_pos, last = state
        ones = torch.ones((mask.shape[0], 1), dtype=mask.dtype, device=self.device)
        mask = torch.cat([mask, ones], dim=1)
        cur_pos = cur_pos + 1
        out = self.model(input_ids=last, attention_mask=mask, past_key_values=cache,
                         position_ids=cur_pos, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        return out.past_key_values, mask, cur_pos, nxt

    def _merge(self, a, b):
        """Merge running batch ``a`` with newly-prefilled batch ``b``.

        Pads the shorter cache along the sequence axis (front) and concatenates
        along the batch axis.
        """
        ca, ma, pa, la = a
        cb, mb, pb, lb = b
        la_len, lb_len = ma.shape[1], mb.shape[1]
        length = max(la_len, lb_len)
        for layer in ca.layers:
            if la_len < length:
                layer.keys = F.pad(layer.keys, (0, 0, length - la_len, 0))
                layer.values = F.pad(layer.values, (0, 0, length - la_len, 0))
        for layer in cb.layers:
            if lb_len < length:
                layer.keys = F.pad(layer.keys, (0, 0, length - lb_len, 0))
                layer.values = F.pad(layer.values, (0, 0, length - lb_len, 0))
        for layer_a, layer_b in zip(ca.layers, cb.layers):
            layer_a.keys = torch.cat([layer_a.keys, layer_b.keys], dim=0)
            layer_a.values = torch.cat([layer_a.values, layer_b.values], dim=0)
        mask = torch.cat([F.pad(ma, (length - la_len, 0)), F.pad(mb, (length - lb_len, 0))], dim=0)
        return ca, mask, torch.cat([pa, pb], dim=0), torch.cat([la, lb], dim=0)

    def _retire(self, active, state, results, t0):
        """Record the newest token per row; evict finished rows.

        Returns (surviving_active, surviving_state | None).
        """
        cache, mask, cur_pos, last = state
        toks = last.squeeze(1).tolist()  # single GPU->CPU sync per step
        keep = []
        for i, r in enumerate(active):
            r.generated.append(toks[i])
            if toks[i] == self.eos_id or len(r.generated) >= r.max_new_tokens:
                r.done = True
                r.t_finish = time.time() - t0
                results[r.id] = r
            else:
                keep.append(i)
        if not keep:
            return [], None
        if len(keep) < len(active):
            kt = torch.tensor(keep, device=self.device)
            for layer in cache.layers:
                layer.keys = layer.keys.index_select(0, kt).contiguous()
                layer.values = layer.values.index_select(0, kt).contiguous()
            state = (cache, mask.index_select(0, kt), cur_pos.index_select(0, kt),
                     last.index_select(0, kt))
        return [active[i] for i in keep], state

    def _take(self, waiting: deque, n: int, t0: float) -> list[Request]:
        batch = []
        while waiting and len(batch) < n:
            r = waiting.popleft()
            r.t_admit = time.time() - t0
            batch.append(r)
        return batch

    def run(self, requests: list[Request]) -> dict[int, Request]:
        """Run all requests to completion with continuous batching."""
        for r in requests:
            r.prompt_ids = self.tokenizer(r.prompt).input_ids
        waiting = deque(requests)
        active: list[Request] = []
        state = None
        t0 = time.time()
        results: dict[int, Request] = {}

        while waiting or active:
            if not active:
                batch = self._take(waiting, self.max_batch_size, t0)
                state = self._prefill(batch)
                active, state = self._retire(batch, state, results, t0)
                continue

            state = self._decode(state)
            active, state = self._retire(active, state, results, t0)

            # Backfill freed slots by prefilling only the new requests.
            free = self.max_batch_size - len(active)
            if active and waiting and free > 0:
                batch = self._take(waiting, free, t0)
                new_state = self._prefill(batch)
                new_active, new_state = self._retire(batch, new_state, results, t0)
                if new_active:
                    state = self._merge(state, new_state)
                    active = active + new_active
            elif not active and waiting:
                state = None

        return results

    def decode_text(self, req: Request) -> str:
        return self.tokenizer.decode(req.generated, skip_special_tokens=True)
