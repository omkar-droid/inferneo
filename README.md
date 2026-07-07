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
| Paged-KV engine: block tables, unified token-budget scheduler, continuous batching, chunked prefill, preemption | ✅ working, greedy output matches HF token-for-token (incl. TinyLlama-1.1B on CPU/MPS/CUDA) |
| Llama-family models (Llama 2/3, TinyLlama, Mistral) via HF safetensors | ✅ |
| Sampler: temperature, top-k / top-p / min-p, penalties, seeds, logprobs | ✅ |
| Offline `LLM` API | ✅ |
| Hash-chain prefix caching | ✅ (opt-in: `enable_prefix_caching=True`) |
| FlashInfer CUDA fast path | ✅ auto-selected on CUDA (fp16/bf16, head_dim 64/128/256); SDPA reference elsewhere |
| OpenAI-compatible server | ✅ `/v1/completions`, `/v1/chat/completions` (+SSE), `/v1/models`, `/health` |
| CUDA graphs, on-GPU sampling (perf) | ⏳ next — the main gap vs vLLM |
| Speculative decoding, P/D disaggregation, radix cache | ⏳ research roadmap |

On an H100 NVL (TinyLlama-1.1B, fp16, 200 ragged requests) inferneo's paged +
FlashInfer engine runs at **7,671 tok/s** — ~5× a naive padded baseline, and
~6× behind **vLLM's 47,094 tok/s**. The gap is per-step overhead (no CUDA
graphs yet), not the attention kernel; output is token-for-token identical to
HuggingFace. See [benchmarks/README.md](benchmarks/README.md) for the honest
breakdown.

```python
from inferneo import LLM, SamplingParams

llm = LLM("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
outs = llm.generate(["The capital of France is"], SamplingParams(max_tokens=32))
print(outs[0].outputs[0].text)
```

Or run it as an OpenAI-compatible server:

```bash
inferneo serve --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --port 8000
curl http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"tinyllama","messages":[{"role":"user","content":"Hi!"}],"stream":true}'
```

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

# Offline generation with the paged engine (any device: cuda/mps/cpu)
python examples/offline_inference.py

# Compare the paged engine against the padded baseline
python benchmarks/offline_throughput.py

# Padded continuous-batching baseline demo, with correctness check
python benchmarks/continuous_batching_demo.py
```

Run the tests (CPU only, seconds):

```bash
pytest            # unit tests (torch-free scheduler/KV) + correctness vs HF
```

## Benchmarks

See `benchmarks/`. Methodology: same GPU, same model, same dtype, warmed servers, 3 runs
per point; TTFT/ITL percentiles, output tok/s, and goodput under an SLO; vLLM (pinned
version, flags recorded) as the external baseline.
