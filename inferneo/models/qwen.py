"""Qwen2 / Qwen3 — Llama-family models with two small deltas each.

This is what "generalized, not model-specific" looks like in practice: the whole
engine (paged KV, scheduler, CUDA graphs, sampler, loader, fused QKV/gate-up) is
reused unchanged. Only the attention block differs, and only slightly:

    Qwen2:  q/k/v projections have a bias, o_proj does not.   (no QK-norm)
    Qwen3:  no qkv bias, but RMS-norm q and k per head before RoPE.

Weights load through the same fuse_state_dict (it already merges the q/k/v biases,
and Qwen3's q_norm/k_norm names match HF's), so nothing special is needed there.
"""

from __future__ import annotations

from inferneo.models.llama import LlamaAttention, LlamaForCausalLM


class Qwen2Attention(LlamaAttention):
    qkv_bias = True     # Qwen2 puts a bias on q/k/v ...
    o_bias = False      # ... but not on the output projection


class Qwen3Attention(LlamaAttention):
    qkv_bias = False    # Qwen3 dropped the qkv bias ...
    o_bias = False
    qk_norm = True       # ... and added per-head RMSNorm on q and k


class Qwen2ForCausalLM(LlamaForCausalLM):
    attention_cls = Qwen2Attention


class Qwen3ForCausalLM(LlamaForCausalLM):
    attention_cls = Qwen3Attention
