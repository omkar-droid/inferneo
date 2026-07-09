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

The point of inferneo is that closing each of these is a small, isolated change
against a readable engine — not a fork of a production system.

## Reproduce

```bash
# inferneo (in the inferneo venv)
python benchmarks/offline_throughput.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda
```
