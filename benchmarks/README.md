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
| inferneo (FlashInfer, eager) | 7,671 | 0.16× |
| inferneo padded baseline | 1,498 | 0.03× |
| inferneo (SDPA reference, eager) | 389 | 0.008× |

**Reading this honestly:** inferneo's paged + FlashInfer engine is correct
(greedy output matches HuggingFace token-for-token) and already ~5× the naive
padded baseline, but it is ~6× behind vLLM. The gap is not the attention kernel
(both use FlashInfer-class kernels) — it is per-step overhead:

1. **No CUDA graphs.** inferneo runs eager, so every decode step pays full
   Python + kernel-launch cost. vLLM captures the decode step in a CUDA graph.
   This is the single largest lever and the top Phase-5 optimization.
2. **Per-step host work.** Rebuilding index tensors from Python lists and
   calling FlashInfer `plan()` every step; vLLM reuses persistent buffers.
3. **Sampling.** A batched greedy fast path (on-GPU argmax, single sync) is in;
   the general sampler still round-trips to CPU. Full on-GPU sampling is next.

The point of inferneo is that closing each of these is a small, isolated change
against a readable engine — not a fork of a production system.

## Reproduce

```bash
# inferneo (in the inferneo venv)
python benchmarks/offline_throughput.py --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --device cuda
```
