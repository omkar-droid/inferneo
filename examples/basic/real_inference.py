#!/usr/bin/env python3
"""Minimal, runnable inference example for Inferneo.

Unlike ``basic_usage.py`` (which drives the multi-model orchestration layer),
this example exercises the real generation path directly: the
``TransformersModel`` backend loads a HuggingFace model and runs it on GPU.

Run from the repo root:

    PYTHONPATH=. python examples/basic/real_inference.py
"""

import asyncio
import time

import torch

from inferneo.models.transformers import TransformersModel
from inferneo.models.base import GenerationConfig


class _LoadConfig:
    """Loader options read by TransformersModel.initialize (via getattr)."""

    trust_remote_code = True
    revision = None


async def main() -> None:
    model_name = "gpt2"
    print(f"CUDA available: {torch.cuda.is_available()}")

    model = TransformersModel(model_name, _LoadConfig())
    t0 = time.time()
    await model.initialize(_LoadConfig())
    print(f"Loaded {model_name} in {time.time() - t0:.2f}s (state={model.state})")

    gen_config = GenerationConfig(max_tokens=40, temperature=0.7, top_p=0.9, top_k=50)
    prompts = [
        "The capital of France is",
        "In 2026, AI inference engines",
    ]
    for prompt in prompts:
        t1 = time.time()
        result = await model.generate(prompt, gen_config)
        dt = time.time() - t1
        n = result.usage["completion_tokens"]
        print(f"\nPrompt : {prompt!r}")
        print(f"Output : {result.text!r}")
        print(f"Speed  : {n} tokens in {dt:.2f}s = {n / dt:.1f} tok/s")


if __name__ == "__main__":
    asyncio.run(main())
