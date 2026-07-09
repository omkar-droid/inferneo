#!/usr/bin/env python3
"""OpenAI-server load generator — measures TTFT, TPOT/ITL, and throughput over
HTTP against any OpenAI-compatible /v1/completions endpoint (inferneo, vLLM, …).

This is the apples-to-apples serving benchmark: point it at each engine's server
in turn, same workload, and compare the client-observed latencies.

    python benchmarks/openai_load.py --base-url http://localhost:8000 \
        --model tinyllama --requests 200 --rate 30
"""

import argparse
import asyncio
import json
import random
import time

import httpx

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


def pct(xs, p):
    if not xs:
        return 0.0
    s = sorted(xs)
    return s[min(len(s) - 1, int(p / 100 * len(s)))]


async def one(client, url, model, prompt, max_tokens, delay, sink):
    await asyncio.sleep(delay)
    t0 = time.perf_counter()
    first, n = None, 0
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "stream": True, "ignore_eos": True,
    }
    try:
        async with client.stream("POST", url, json=payload) as r:
            async for line in r.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                txt = json.loads(data)["choices"][0].get("text", "")
                if txt:
                    if first is None:
                        first = time.perf_counter()
                    n += 1
    except Exception as e:  # noqa: BLE001
        print("request failed:", e)
        return
    end = time.perf_counter()
    if first is not None and n > 1:
        sink.append({"ttft": first - t0, "tpot": (end - first) / (n - 1), "n": n})


async def main_async(args):
    random.seed(0)
    prefix = ("You are a careful, concise assistant. " * ((args.shared_prefix // 6) + 1)) \
        if args.shared_prefix else ""
    prompts = [prefix + _PROMPTS[i % len(_PROMPTS)] for i in range(args.requests)]
    lens = [random.choice([64, 128, 192, 256]) for _ in range(args.requests)]
    delays, t = [], 0.0
    for _ in range(args.requests):
        delays.append(t)
        t += random.expovariate(args.rate)

    url = args.base_url.rstrip("/") + "/v1/completions"
    sink = []
    limits = httpx.Limits(max_connections=args.requests + 10)
    async with httpx.AsyncClient(timeout=120.0, limits=limits) as client:
        # warm up
        await one(client, url, args.model, prompts[0], 8, 0.0, [])
        wall0 = time.perf_counter()
        await asyncio.gather(*[
            one(client, url, args.model, p, L, d, sink)
            for p, L, d in zip(prompts, lens, delays)
        ])
        wall = time.perf_counter() - wall0

    ttft = [r["ttft"] * 1000 for r in sink]
    tpot = [r["tpot"] * 1000 for r in sink]
    tot = sum(r["n"] for r in sink)
    print(f"\n{args.base_url}  model={args.model}  ok={len(sink)}/{args.requests}  "
          f"rate={args.rate}/s  shared_prefix={args.shared_prefix}")
    print(f"{'metric':<12}{'p50':>10}{'p99':>10}")
    print("-" * 32)
    print(f"{'TTFT ms':<12}{pct(ttft,50):>10.1f}{pct(ttft,99):>10.1f}")
    print(f"{'TPOT ms':<12}{pct(tpot,50):>10.2f}{pct(tpot,99):>10.2f}")
    print(f"throughput  {tot/wall:>10.0f} tok/s   ({len(sink)/wall:.1f} req/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:8000")
    ap.add_argument("--model", default="model")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--rate", type=float, default=30.0)
    ap.add_argument("--shared-prefix", type=int, default=0)
    asyncio.run(main_async(ap.parse_args()))


if __name__ == "__main__":
    main()
