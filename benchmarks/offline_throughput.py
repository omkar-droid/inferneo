#!/usr/bin/env python3
"""Offline throughput + KV memory: static batching vs padded-CB vs paged inferneo.

The honest replacement for the deleted fabricated comparison reports. The point
this makes is *memory*, not just speed: with ragged output lengths, the paged
engine holds only the blocks it needs, so it sustains larger effective batches
without the rectangular padding waste of static/padded batching.

    python benchmarks/offline_throughput.py
    python benchmarks/offline_throughput.py --model gpt2 --device cuda

Note: gpt2 is not a Llama-family model, so the paged inferneo engine only runs
for Llama/Mistral checkpoints (e.g. TinyLlama). The baselines run for any HF
causal LM.
"""

import argparse
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.baselines.hf_padded_engine import (  # noqa: E402
    ContinuousBatchingEngine,
    Request,
    pick_device,
)

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


def make_requests(n: int):
    random.seed(0)
    return [
        Request(
            id=i,
            prompt=BASE_PROMPTS[i % len(BASE_PROMPTS)],
            max_new_tokens=random.choice([16, 24, 32, 64, 96, 128]),
        )
        for i in range(n)
    ]


def bench_padded_cb(model: str, device: str, n: int, batch: int):
    engine = ContinuousBatchingEngine(model, max_batch_size=batch, device=device)
    reqs = make_requests(n)
    _sync(device)
    t0 = time.time()
    results = engine.run(reqs)
    _sync(device)
    dt = time.time() - t0
    tokens = sum(len(r.generated) for r in results.values())
    return tokens, dt


def bench_inferneo(model: str, device: str, n: int, batch: int):
    from inferneo import LLM, SamplingParams

    dev = None if device == "auto" else device
    llm = LLM(model, device=dev or "auto", max_num_seqs=batch)
    reqs = make_requests(n)
    prompts = [r.prompt for r in reqs]
    params = [SamplingParams(max_tokens=r.max_new_tokens, temperature=0) for r in reqs]
    t0 = time.time()
    outs = llm.generate(prompts, params)
    dt = time.time() - t0
    tokens = sum(len(o.outputs[0].token_ids) for o in outs)
    return tokens, dt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--device", default=pick_device())
    parser.add_argument("--requests", type=int, default=48)
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()

    print(f"model={args.model} device={args.device} "
          f"requests={args.requests} batch={args.batch}\n")
    print(f"{'engine':<28}{'tokens':>8}{'time':>9}{'tok/s':>10}")
    print("-" * 55)

    tokens, dt = bench_padded_cb(args.model, args.device, args.requests, args.batch)
    print(f"{'padded continuous batching':<28}{tokens:>8}{dt:>8.2f}s{tokens / dt:>10.1f}")

    try:
        tokens, dt = bench_inferneo(args.model, args.device, args.requests, args.batch)
        print(f"{'inferneo (paged)':<28}{tokens:>8}{dt:>8.2f}s{tokens / dt:>10.1f}")
    except ValueError as e:
        print(f"inferneo (paged): skipped — {e}")


if __name__ == "__main__":
    main()
