#!/usr/bin/env python3
"""Offline batch generation with the paged inferneo engine.

Runs on any device (cuda / mps / cpu) — dtype and device are auto-selected.

    python examples/offline_inference.py
    python examples/offline_inference.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0
"""

import argparse

from inferneo import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    parser.add_argument("--max-tokens", type=int, default=48)
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    llm = LLM(args.model)
    params = SamplingParams(max_tokens=args.max_tokens, temperature=args.temperature)
    prompts = [
        "The capital of France is",
        "In 2026, the state of the art in LLM inference is",
        "A haiku about GPUs:",
    ]

    for out in llm.generate(prompts, params):
        completion = out.outputs[0]
        n = len(completion.token_ids)
        print(f"\nPrompt: {out.prompt!r}")
        print(f"Output ({n} tokens, {completion.finish_reason}): {completion.text!r}")


if __name__ == "__main__":
    main()
