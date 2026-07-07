# Inferneo

A research testbed for LLM inference serving.

Inferneo is a small, readable inference engine in the spirit of vLLM and SGLang: real
paged-KV continuous batching, an OpenAI-compatible server (planned), and honest benchmarks —
designed so that a scheduling / KV-cache / disaggregation idea from a paper is a small diff,
not a fork of a 500k-line production engine.

**It does not claim to beat vLLM on throughput.** It claims to be understandable, correct,
and measurable. Every benchmark published here includes the vLLM number on the same
hardware, even when inferneo loses.

## What works today

| Component | Status |
|---|---|
| Padded continuous-batching baseline (HF models, own decode loop) | ✅ working, correctness-verified vs static batching |
| Paged-KV engine (block tables, token-budget scheduler) | 🚧 in progress |
| OpenAI-compatible server | ⏳ planned |
| Prefix caching, speculative decoding, P/D disaggregation | ⏳ research roadmap |

Earlier revisions of this repository contained placeholder engine code and benchmark
reports generated against it. Those numbers were meaningless and are retracted; the
history is preserved in git.

## Architecture (target)

- **Control plane** (`inferneo/engine/`, `inferneo/kv/`): pure Python, no torch imports.
  A vLLM-V1-style unified scheduler (no prefill/decode distinction — each step schedules
  `{request: num_tokens}` under a token budget) and a block-based KV cache manager with
  hash-chain prefix caching.
- **Tensor plane** (`inferneo/executor/`, `inferneo/attention/`, `inferneo/models/`,
  `inferneo/sampling/`): PyTorch. Flat varlen batching (no padding), pluggable attention
  backends — pure-torch SDPA reference (CPU/MPS/CUDA) and FlashInfer (CUDA fast path).
- **Serving plane** (`inferneo/server/`): FastAPI OpenAI-compatible API behind an
  `EngineClient` protocol.

## Quick start

```bash
pip install -e ".[dev]"

# Run the padded continuous-batching baseline demo (any device: cuda/mps/cpu)
python benchmarks/continuous_batching_demo.py
```

## Benchmarks

See `benchmarks/`. Methodology: same GPU, same model, same dtype, warmed servers, 3 runs
per point; TTFT/ITL percentiles, output tok/s, and goodput under an SLO; vLLM (pinned
version, flags recorded) as the external baseline.
