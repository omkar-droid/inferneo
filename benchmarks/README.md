# Benchmarks

Honest measurement only. Every comparison records the hardware, model, dtype,
and the competing engine's version and flags. When inferneo loses, the number
stays in.

## Methodology

- **Baselines**: vLLM (external reference) and `baselines/hf_padded_engine.py`
  (internal padded continuous-batching baseline) and static batching.
- **Fair comparison**: same GPU, same model, same dtype, same `max_model_len`,
  same prompt set and output lengths, both engines warmed before timing.
- **Metrics**: offline output tok/s here; serving TTFT/ITL percentiles and
  goodput-under-SLO arrive with the server (Phase 3).

## Results

### Offline throughput vs vLLM — the ratio improves with model size

Same H100, same dtype, same 200-request ragged workload, both engines warmed:

| Model | inferneo tok/s | vLLM 0.24.0 tok/s | ratio |
|---|---:|---:|---:|
| **Mistral-7B-Instruct-v0.2** | **8,216** | **13,222** | **0.62×** |
| TinyLlama-1.1B | 18,900 | 47,094 | 0.40× |

The tiny model is our *worst* case: it is so small that our fixed per-step Python
overhead dominates and decode is latency-bound (the GPU idles between kernels). On
a real 7B model the GPU is actually busy, so that overhead shrinks in relative
terms and inferneo reaches **0.62× of vLLM** — for a ~2,000-line readable engine
against years of vLLM optimization. Larger models should close the gap further.

### Offline throughput — TinyLlama-1.1B, H100 NVL (96GB), fp16, 200 requests, ragged 64–256 output tokens

| Engine | tok/s | Relative |
|---|---:|---:|
| vLLM 0.24.0 (CUDA graphs) | 47,094 | 1.00× |
| **inferneo (FlashInfer + CUDA graphs)** | **17,640** | **0.37×** |
| inferneo (FlashInfer, eager) | 7,671 | 0.16× |
| inferneo padded baseline | 1,498 | 0.03× |
| inferneo (SDPA reference, eager) | 389 | 0.008× |

**Reading this honestly:** inferneo's paged + FlashInfer engine is correct
(greedy output matches HuggingFace token-for-token). CUDA graphs on the decode
step give a **2.3× speedup** (7,671 → 17,640 tok/s) by collapsing the hundreds
of per-step kernel launches into one replay, closing the gap to vLLM from ~6× to
~2.7×. The remaining gap is per-step *host* overhead, not the GPU work:

1. **Per-step host work.** The scheduler builds a `SchedulerOutput` in Python and
   the runner rebuilds index tensors and calls FlashInfer `plan()` every step;
   vLLM overlaps and amortizes more of this. This is now the largest remaining lever.
2. **Sampling.** A batched greedy fast path (on-GPU argmax, single sync) is in;
   the general sampler still round-trips to CPU. Full on-GPU sampling is next.

CUDA graphs cover *pure-decode* steps (every request advances one token); prefill
and mixed steps run eager. Toggle with `enable_cuda_graph=False`.

### Sampling throughput — same setup, temperature 0.8 + top-p 0.95

| Sampler | tok/s |
|---|---:|
| **on-GPU batched (current)** | **13,694** |
| per-request CPU loop (previous) | 342 |

The old sampler brought the full `[batch, vocab]` logits to the CPU and looped
over requests in Python — **40× slower** on a sampling workload than the batched
on-GPU sampler, which does temperature / top-k / top-p / penalties and a
Gumbel-max draw entirely on device with one sync. Since `temperature > 0` is the
default for chat and creative generation, this was the difference between usable
and unusable at scale. (Greedy throughput is unchanged — it already used an
on-GPU argmax fast path.)

### Single-stream latency — same model/GPU, greedy, 128 tokens

| Decode forward | tok/s | ms/token |
|---|---:|---:|
| **torch.compile (fused pointwise)** | **425** | **2.35** |
| eager (cuBLAS + separate pointwise kernels) | 267 | 3.74 |

Profiling showed the decode forward is *kernel-latency bound* — even at batch 1
it took 3.3 ms, dominated by executing hundreds of tiny sequential kernels.
`torch.compile` fuses the pointwise ops (RMSNorm, RoPE, SiLU, residual adds) into
far fewer kernels; the fused kernels are then captured in the same CUDA graph.
At low concurrency this cuts per-token latency ~37% (**+59% tok/s**).

The catch: at *large* batch the forward becomes compute/bandwidth-bound, where
cuBLAS already wins and the compiled kernels are slightly slower. So inferneo
compiles only the small batch-size buckets (≤ 64) and keeps the eager cuBLAS
forward for large ones — a latency win with no throughput cost (batch-256
throughput is unchanged). Toggle with `enable_torch_compile=False`.

The point of inferneo is that closing each of these is a small, isolated change
against a readable engine — not a fork of a production system.

## Serving latency — TTFT and TPOT

`serve_benchmark.py` drives the async engine under poisson arrivals and reports
the client-observed latencies that actually matter for serving:

- **TTFT** (time to first token) — arrival → first token; **prefill-bound**.
- **TPOT / ITL** (time per output token) — the steady-state decode latency.

Baseline (TinyLlama-1.1B, H100, fp16, 200 req @ 30 req/s, short prompts):

| metric | p50 | p99 |
|---|---:|---:|
| TTFT | 15.8 ms | 26.5 ms |
| TPOT | 4.03 ms | 5.26 ms |

### Prefix caching — TTFT with a shared prompt

The headline use case: a long shared prefix (system prompt, few-shot examples)
that every request repeats. Hash-chain prefix caching skips re-prefilling it on
cache hits. 1500-token shared prefix, 160 req @ 25 req/s:

| | TTFT p50 | TTFT p99 |
|---|---:|---:|
| prefix caching **off** | 50.5 ms | 140.9 ms |
| prefix caching **on** | **18.2 ms** | **32.5 ms** |
| | **−64%** | **−77%** |

The longer the shared prefix (and the larger the model), the bigger the win —
prefill cost that used to repeat per request is paid once.

### Chunked prefill — an honest null result here

Chunked prefill (`--chunked-prefill N`, i.e. `long_prefill_token_threshold`)
caps prompt tokens per step so a long prefill doesn't stall concurrent decodes,
trading a little TTFT for smoother TPOT. On this setup it **did not help** — with
a 1500-token shared prefix at 35 req/s, chunking to 512 left TPOT unchanged and
raised TTFT. The reason is honest and expected: on a 1.1B model an H100 prefills
1500 tokens in ~2–3 ms, so prefill barely disrupts decode and there is nothing to
smooth — chunking only adds overhead. The win shows up on **large models**, where
a long prefill costs 100+ ms and genuinely stalls decoders. The knob is there for
that regime; off is the right default here.

## Reproduce

```bash
# offline throughput
python benchmarks/offline_throughput.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda

# serving TTFT/TPOT, and the prefix-caching effect
python benchmarks/serve_benchmark.py --requests 200 --rate 30
python benchmarks/serve_benchmark.py --requests 160 --rate 25 --shared-prefix 1500 --prefix-caching
```
