"""Llama-family causal LM over the paged attention backend.

Covers Llama 2/3, TinyLlama, and Mistral-style configs (GQA, SwiGLU, RMSNorm,
RoPE). Module names mirror HF Transformers exactly, so HF safetensors load
with no renaming. Unlike HF modules, forward takes a *flat* token batch
(no padding, no rectangular [batch, seq] tensors) plus attention metadata —
the shape the unified scheduler produces.
"""

from __future__ import annotations

import torch
from torch import nn

from inferneo.attention.interface import AttentionBackend
from inferneo.models.layers import RMSNorm, RotaryEmbedding, get_rope_parameters


class LlamaAttention(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = getattr(config, "head_dim", None) or hidden // self.num_heads
        bias = getattr(config, "attention_bias", False)

        self.q_proj = nn.Linear(hidden, self.num_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(hidden, self.num_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, hidden, bias=bias)
        self.backend = backend

    def forward(self, x, positions, rotary, kv_cache, attn_metadata):
        t = x.shape[0]
        q = self.q_proj(x).view(t, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(t, self.num_kv_heads, self.head_dim)
        v = self.v_proj(x).view(t, self.num_kv_heads, self.head_dim)
        q, k = rotary(positions, q, k)
        out = self.backend.forward(q, k, v, kv_cache, attn_metadata)
        return self.o_proj(out.reshape(t, self.num_heads * self.head_dim))


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, inner = config.hidden_size, config.intermediate_size
        bias = getattr(config, "mlp_bias", False)
        self.gate_proj = nn.Linear(hidden, inner, bias=bias)
        self.up_proj = nn.Linear(hidden, inner, bias=bias)
        self.down_proj = nn.Linear(inner, hidden, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        self.self_attn = LlamaAttention(config, backend)
        self.mlp = LlamaMLP(config)
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

    def forward(self, x, positions, rotary, kv_cache, attn_metadata):
        x = x + self.self_attn(
            self.input_layernorm(x), positions, rotary, kv_cache, attn_metadata
        )
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaModel(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            LlamaDecoderLayer(config, backend) for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        head_dim = getattr(config, "head_dim", None) or (
            config.hidden_size // config.num_attention_heads
        )
        self.rotary = RotaryEmbedding(head_dim, get_rope_parameters(config))

    def forward(self, input_ids, positions, kv_caches, attn_metadata):
        x = self.embed_tokens(input_ids)
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer(x, positions, self.rotary, kv_cache, attn_metadata)
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, backend)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(self, input_ids, positions, kv_caches, attn_metadata) -> torch.Tensor:
        """Returns hidden states [num_tokens, hidden]; call compute_logits on
        the (few) rows that actually sample — skipping lm_head for the rest."""
        return self.model(input_ids, positions, kv_caches, attn_metadata)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight
