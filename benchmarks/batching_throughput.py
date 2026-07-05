#!/usr/bin/env python3
"""Throughput benchmark: sequential vs. batched generation.

Demonstrates the core lever behind batching inference engines (vLLM, SGLang):
on a small model the GPU is far from saturated by a single sequence, so
batching packs more work into each forward pass at nearly constant latency,
scaling total throughput roughly linearly with batch size.

Run from the repo root:

    PYTHONPATH=. python benchmarks/batching_throughput.py
"""

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "gpt2"
MAX_NEW = 64

_tok = AutoTokenizer.from_pretrained(MODEL)
_tok.padding_side = "left"
if _tok.pad_token is None:
    _tok.pad_token = _tok.eos_token
_model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float16).cuda().eval()

BASE_PROMPTS = [
    "The capital of France is", "In the year 2026, artificial intelligence",
    "Once upon a time in a distant galaxy", "The best way to learn programming is",
    "Quantum computing works by", "Climate change is caused by",
    "The recipe for a good pizza starts with", "History teaches us that",
]


def _generate(prompts):
    enc = _tok(prompts, return_tensors="pt", padding=True).to("cuda")
    with torch.no_grad():
        out = _model.generate(**enc, max_new_tokens=MAX_NEW, do_sample=False,
                              pad_token_id=_tok.pad_token_id)
    new = out.shape[1] - enc.input_ids.shape[1]
    return new * len(prompts)


def _requests(n):
    return (BASE_PROMPTS * ((n // len(BASE_PROMPTS)) + 1))[:n]


def main():
    _generate(BASE_PROMPTS[:2])  # warmup
    torch.cuda.synchronize()

    print(f"{'mode':<22}{'reqs':>5}{'latency':>10}{'tok/s':>12}{'vs seq':>9}")
    print("-" * 58)

    n = 16
    torch.cuda.synchronize()
    t0 = time.time()
    seq_tokens = sum(_generate([p]) for p in _requests(n))
    torch.cuda.synchronize()
    seq_dt = time.time() - t0
    seq_tps = seq_tokens / seq_dt
    print(f"{'sequential (1-by-1)':<22}{n:>5}{seq_dt:>9.2f}s{seq_tps:>12.1f}{'1.0x':>9}")

    for bs in [1, 4, 8, 16, 32, 64]:
        torch.cuda.synchronize()
        t0 = time.time()
        total = _generate(_requests(bs))
        torch.cuda.synchronize()
        dt = time.time() - t0
        tps = total / dt
        print(f"{f'batched (bs={bs})':<22}{bs:>5}{dt:>9.2f}s{tps:>12.1f}{f'{tps / seq_tps:.1f}x':>9}")


if __name__ == "__main__":
    main()
