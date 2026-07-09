#!/usr/bin/env python3
"""Serving benchmark: TTFT, TPOT/ITL, and throughput under poisson arrivals.

Drives the AsyncEngine with requests arriving at a target rate and measures, per
request, the client-observed latencies:

  * TTFT  — time to first token (arrival -> first streamed token); prefill-bound.
  * TPOT  — mean time per output token after the first (a.k.a. ITL); decode-bound.

Reports p50/p99 of each plus output throughput. Use --shared-prefix to prepend a
long common prompt and --prefix-caching to see its effect on TTFT.

    python benchmarks/serve_benchmark.py --requests 200 --rate 25
    python benchmarks/serve_benchmark.py --requests 200 --rate 25 \
        --shared-prefix 800 --prefix-caching
"""

import argparse
import asyncio
import random
import time

from inferneo.engine.async_engine import AsyncEngine
from inferneo.sampling_params import SamplingParams

_PROMPTS = [
    "Summarize the causes of the French Revolution.",
    "Write a Python function that reverses a linked list.",
    "Explain how photosynthesis works, step by step.",
    "What are the tradeoffs between TCP and UDP?",
    "Describe the plot of a heist movie set on Mars.",
    "Give three tips for improving sleep quality.",
    "How does a transformer neural network attend to tokens?",
    "Compare electric and hydrogen vehicles.",
]


def percentile(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


async def one_request(engine, prompt, params, delay, sink):
    await asyncio.sleep(delay)
    arrival = time.perf_counter()
    first = None
    stamps = []
    async for out in engine.generate(prompt, params):
        now = time.perf_counter()
        n = len(out.outputs[0].token_ids)
        if n and first is None:
            first = now
        stamps.append((now, n))
    end = time.perf_counter()
    n_out = stamps[-1][1] if stamps else 0
    if first is None or n_out < 1:
        return
    ttft = first - arrival
    tpot = (end - first) / max(1, n_out - 1)
    sink.append({"ttft": ttft, "tpot": tpot, "n_out": n_out, "end": end})


async def main_async(args):
    engine = AsyncEngine.from_model(
        args.model, dtype="float16", max_num_seqs=args.max_num_seqs,
        max_num_batched_tokens=args.max_num_batched_tokens,
        long_prefill_token_threshold=args.chunked_prefill,
        enable_prefix_caching=args.prefix_caching,
        gpu_memory_utilization=0.85,
    )
    engine.start()

    random.seed(0)
    shared = ""
    if args.shared_prefix:
        # A long, fixed system preamble shared by every request.
        shared = ("You are a careful, concise assistant. " * 200)
        shared = " ".join(shared.split()[: args.shared_prefix]) + "\n\n"
    prompts = [shared + _PROMPTS[i % len(_PROMPTS)] for i in range(args.requests)]
    out_lens = [random.choice([64, 128, 192, 256]) for _ in range(args.requests)]

    # Poisson arrivals: exponential inter-arrival gaps at `rate` req/s.
    delays, t = [], 0.0
    for _ in range(args.requests):
        delays.append(t)
        t += random.expovariate(args.rate)

    sink: list[dict] = []
    # warm up prefill/graph/compile paths
    async for _ in engine.generate(prompts[0], SamplingParams(max_tokens=8, temperature=0)):
        pass

    wall0 = time.perf_counter()
    tasks = [
        asyncio.create_task(
            one_request(
                engine, p,
                SamplingParams(max_tokens=L, temperature=0, ignore_eos=True),
                d, sink,
            )
        )
        for p, L, d in zip(prompts, out_lens, delays)
    ]
    await asyncio.gather(*tasks)
    wall = time.perf_counter() - wall0
    engine.shutdown()

    ttfts = [r["ttft"] * 1000 for r in sink]
    tpots = [r["tpot"] * 1000 for r in sink]
    total_out = sum(r["n_out"] for r in sink)
    print(f"\nmodel={args.model}  requests={len(sink)}/{args.requests}  "
          f"rate={args.rate}/s  prefix_caching={args.prefix_caching}  "
          f"shared_prefix={args.shared_prefix} tok")
    print(f"{'metric':<10}{'p50':>10}{'p99':>10}")
    print("-" * 30)
    print(f"{'TTFT ms':<10}{percentile(ttfts,50):>10.1f}{percentile(ttfts,99):>10.1f}")
    print(f"{'TPOT ms':<10}{percentile(tpots,50):>10.2f}{percentile(tpots,99):>10.2f}")
    print(f"\noutput throughput: {total_out/wall:.0f} tok/s   "
          f"({len(sink)/wall:.1f} req/s over {wall:.1f}s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--rate", type=float, default=25.0, help="poisson arrival rate (req/s)")
    ap.add_argument("--shared-prefix", type=int, default=0, help="shared prefix length (tokens)")
    ap.add_argument("--prefix-caching", action="store_true")
    ap.add_argument("--chunked-prefill", type=int, default=0,
                    help="max prompt tokens per step (0 = unchunked)")
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=8192)
    args = ap.parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
