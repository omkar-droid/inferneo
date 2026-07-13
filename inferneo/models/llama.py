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

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # Fused QKV: one GEMM instead of three cuts kernel count in the
        # latency-bound decode path. Split back after the projection.
        self.qkv_proj = nn.Linear(hidden, self.q_size + 2 * self.kv_size, bias=bias)
        self.o_proj = nn.Linear(self.q_size, hidden, bias=bias)
        self.backend = backend

    def forward(self, x, positions, rotary, kv_cache, attn_metadata):
        t = x.shape[0]
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(t, self.num_heads, self.head_dim)
        k = k.view(t, self.num_kv_heads, self.head_dim)
        v = v.view(t, self.num_kv_heads, self.head_dim)
        q, k = rotary(positions, q, k)
        out = self.backend.forward(q, k, v, kv_cache, attn_metadata)
        return self.o_proj(out.reshape(t, self.q_size))


class LlamaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden, self.inner = config.hidden_size, config.intermediate_size
        bias = getattr(config, "mlp_bias", False)
        # Fused gate+up: one GEMM instead of two.
        self.gate_up_proj = nn.Linear(hidden, 2 * self.inner, bias=bias)
        self.down_proj = nn.Linear(self.inner, hidden, bias=bias)
        self.act_fn = nn.SiLU()

    def forward(self, x):
        gate, up = self.gate_up_proj(x).split(self.inner, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


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

    def forward(self, input_ids, positions, kv_caches, attn_metadata, inputs_embeds=None):
        # `inputs_embeds` lets a caller supply the embedding sequence directly —
        # the seam multimodal needs, since an image arrives as embeddings from a
        # vision tower, not as token ids. Everything downstream (paged attention,
        # KV cache, scheduler) is unchanged: by this point an image is just rows
        # in the sequence. Decode always uses input_ids, so the CUDA-graph path
        # is untouched.
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer(x, positions, self.rotary, kv_cache, attn_metadata)
        return self.norm(x)


class LlamaForCausalLM(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        self.config = config
        self.model = LlamaModel(config, backend)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, input_ids, positions, kv_caches, attn_metadata, inputs_embeds=None
    ) -> torch.Tensor:
        """Returns hidden states [num_tokens, hidden]; call compute_logits on
        the (few) rows that actually sample — skipping lm_head for the rest."""
        return self.model(input_ids, positions, kv_caches, attn_metadata, inputs_embeds)

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Token ids -> embeddings. Multimodal preprocessing needs this to build a
        mixed text+image embedding sequence before the forward pass."""
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.model.embed_tokens.weight

    @staticmethod
    def fuse_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Merge HF's separate q/k/v and gate/up weights into the fused layers.

        Concatenation is along the output-feature axis, so the fused GEMM is
        numerically identical to the three (resp. two) separate ones.
        """
        merged = dict(state)
        for suffix in ("weight", "bias"):
            _merge(merged, ["q_proj", "k_proj", "v_proj"], "qkv_proj", suffix)
            _merge(merged, ["gate_proj", "up_proj"], "gate_up_proj", suffix)
        return merged


def _merge(state, parts, fused, suffix):
    """For every layer prefix that has `{prefix}{parts[0]}.{suffix}`, concat all
    `parts` into `{prefix}{fused}.{suffix}` and drop the originals."""
    tail = f"{parts[0]}.{suffix}"  # e.g. q_proj.weight
    for key in [k for k in state if k.endswith(tail)]:
        prefix = key[: -len(tail)]  # up to self_attn./mlp.
        part_keys = [f"{prefix}{p}.{suffix}" for p in parts]
        if not all(pk in state for pk in part_keys):
            continue
        state[f"{prefix}{fused}.{suffix}"] = torch.cat([state[pk] for pk in part_keys], dim=0)
        for pk in part_keys:
            del state[pk]
