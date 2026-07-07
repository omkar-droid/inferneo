#!/usr/bin/env python3
"""Padded continuous batching vs. static batching, with correctness check.

This drives the *baseline* engine (benchmarks/baselines/hf_padded_engine.py),
not the paged inferneo engine.

Static batching runs a wave of requests until the *longest* one finishes, so
short requests hold GPU slots idle. Continuous batching evicts finished
requests immediately and backfills their slots, keeping the batch full. With
ragged output lengths, continuous batching finishes the same work faster.

Run from the repo root:

    python benchmarks/continuous_batching_demo.py
"""

import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.baselines.hf_padded_engine import ContinuousBatchingEngine, Request  # noqa: E402

MODEL = "gpt2"
MAX_BATCH = 16
N_REQUESTS = 64
BASE_PROMPTS = [
    "The capital of France is", "In the year 2026, artificial intelligence",
    "Once upon a time in a distant galaxy", "The best way to learn programming is",
    "Quantum computing works by", "Climate change is caused by",
    "The recipe for a good pizza starts with", "History teaches us that",
]


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def make_requests():
    random.seed(0)
    reqs = []
    for i in range(N_REQUESTS):
        # Ragged lengths: a mix of short and long generations.
        mnt = random.choice([16, 24, 32, 64, 96, 128])
        reqs.append(Request(id=i, prompt=BASE_PROMPTS[i % len(BASE_PROMPTS)], max_new_tokens=mnt))
    return reqs


@torch.no_grad()
def sequential(engine, requests):
    """Naive serving: one request at a time, no batching (the baseline
    continuous batching replaces)."""
    tok, model = engine.tokenizer, engine.model
    toks, t0 = 0, time.time()
    for r in requests:
        enc = tok(r.prompt, return_tensors="pt").to(engine.device)
        out = model.generate(**enc, max_new_tokens=r.max_new_tokens, do_sample=False,
                             pad_token_id=engine.pad_id)
        toks += out.shape[1] - enc.input_ids.shape[1]
    return toks, time.time() - t0


@torch.no_grad()
def static_batching(engine, requests):
    """Wave-by-wave batched generate; each wave runs until its longest req."""
    tok, model = engine.tokenizer, engine.model
    results, t0 = {}, time.time()
    for i in range(0, len(requests), MAX_BATCH):
        wave = requests[i:i + MAX_BATCH]
        wave_max = max(r.max_new_tokens for r in wave)
        enc = tok([r.prompt for r in wave], return_tensors="pt", padding=True).to(engine.device)
        out = model.generate(**enc, max_new_tokens=wave_max, do_sample=False,
                             pad_token_id=engine.pad_id)
        for j, r in enumerate(wave):
            gen = out[j][enc.input_ids.shape[1]:][:r.max_new_tokens].tolist()
            results[r.id] = gen
    return results, time.time() - t0


def main():
    engine = ContinuousBatchingEngine(MODEL, max_batch_size=MAX_BATCH)

    # ---- Continuous batching ----
    reqs = make_requests()
    _sync(engine.device)
    t0 = time.time()
    cont = engine.run(reqs)
    _sync(engine.device)
    cont_dt = time.time() - t0
    cont_tokens = sum(len(r.generated) for r in cont.values())

    # ---- Baselines ----
    seq_tokens, seq_dt = sequential(engine, make_requests())
    stat, stat_dt = static_batching(engine, make_requests())
    stat_tokens = sum(len(g) for g in stat.values())

    # ---- Correctness: continuous greedy == static greedy ----
    checked = matched = 0
    for rid, r in cont.items():
        ref = stat[rid][:len(r.generated)]
        got = r.generated[:len(ref)]
        checked += 1
        matched += int(got == ref)
    print(f"correctness: {matched}/{checked} requests match static greedy token-for-token\n")

    print(f"{'mode':<26}{'reqs':>5}{'tokens':>8}{'time':>9}{'tok/s':>10}")
    print("-" * 58)
    print(f"{'sequential (1-at-a-time)':<26}{N_REQUESTS:>5}{seq_tokens:>8}{seq_dt:>8.2f}s{seq_tokens / seq_dt:>10.1f}")
    print(f"{'continuous batching':<26}{N_REQUESTS:>5}{cont_tokens:>8}{cont_dt:>8.2f}s{cont_tokens / cont_dt:>10.1f}")
    print(f"{'static (offline, optimized)':<26}{N_REQUESTS:>5}{stat_tokens:>8}{stat_dt:>8.2f}s{stat_tokens / stat_dt:>10.1f}")
    print(f"\ncontinuous vs naive sequential: {(cont_tokens / cont_dt) / (seq_tokens / seq_dt):.2f}x throughput")
    print(f"static vs continuous (optimization gap): {(stat_tokens / stat_dt) / (cont_tokens / cont_dt):.2f}x")

    sample = cont[0]
    print(f"\nsample (req 0, {len(sample.generated)} toks): "
          f"{engine.decode_text(sample)!r}")


if __name__ == "__main__":
    main()
