"""Phi-1 / 1.5 / 2 (``PhiForCausalLM``) — a GPT-NeoX-style architecture that
differs from Llama in shape, not just constants:

    * **Partial rotary** — RoPE is applied to only the first ``partial_rotary_factor``
      of each head's channels (0.4 for Phi-2); the rest pass through unrotated.
    * **Parallel block** — attention and MLP both read the *same* layernorm'd input
      and their outputs are added to the residual together (not sequentially).
    * **LayerNorm** (with bias), not RMSNorm; biases throughout attention/MLP.
    * **Non-gated MLP** — ``fc2(gelu(fc1(x)))``, like GPT-2, not SwiGLU.
    * ``lm_head`` has a bias and is **not** tied to the embeddings.

The paged-attention backend, sampler, KV cache, and scheduler are all reused
unchanged; q/k/v fuse through the shared ``fuse_state_dict``.
"""

from __future__ import annotations

import torch
from torch import nn

from inferneo.attention.interface import AttentionBackend
from inferneo.models.layers import RotaryEmbedding, get_rope_parameters
from inferneo.models.llama import LlamaForCausalLM


class PhiAttention(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        hidden = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = getattr(config, "num_key_value_heads", None) or self.num_heads
        self.head_dim = hidden // self.num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.qkv_proj = nn.Linear(hidden, self.q_size + 2 * self.kv_size, bias=True)
        self.dense = nn.Linear(self.q_size, hidden, bias=True)
        self.backend = backend

    def forward(self, x, positions, rotary, kv_cache, attn_metadata):
        t = x.shape[0]
        q, k, v = self.qkv_proj(x).split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q = q.view(t, self.num_heads, self.head_dim)
        k = k.view(t, self.num_kv_heads, self.head_dim)
        v = v.view(t, self.num_kv_heads, self.head_dim)
        q, k = rotary(positions, q, k)  # partial rotary: only the first rotary_dim
        out = self.backend.forward(q, k, v, kv_cache, attn_metadata)
        return self.dense(out.reshape(t, self.q_size))


class PhiMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size, bias=True)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size, bias=True)
        self.act_fn = nn.GELU(approximate="tanh")  # HF "gelu_new"

    def forward(self, x):
        return self.fc2(self.act_fn(self.fc1(x)))


class PhiDecoderLayer(nn.Module):
    """Parallel residual: attn and mlp both read one shared LayerNorm."""

    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        self.self_attn = PhiAttention(config, backend)
        self.mlp = PhiMLP(config)
        self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, x, positions, rotary, kv_cache, attn_metadata):
        h = self.input_layernorm(x)
        attn = self.self_attn(h, positions, rotary, kv_cache, attn_metadata)
        ff = self.mlp(h)
        return x + attn + ff


class PhiModel(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            PhiDecoderLayer(config, backend) for _ in range(config.num_hidden_layers)
        )
        self.final_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        head_dim = config.hidden_size // config.num_attention_heads
        rotary_dim = int(head_dim * getattr(config, "partial_rotary_factor", 1.0))
        self.rotary = RotaryEmbedding(head_dim, get_rope_parameters(config), rotary_dim)

    def forward(self, input_ids, positions, kv_caches, attn_metadata, inputs_embeds=None):
        x = inputs_embeds if inputs_embeds is not None else self.embed_tokens(input_ids)
        for layer, kv_cache in zip(self.layers, kv_caches):
            x = layer(x, positions, self.rotary, kv_cache, attn_metadata)
        return self.final_layernorm(x)


class PhiForCausalLM(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        self.config = config
        self.model = PhiModel(config, backend)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

    def forward(
        self, input_ids, positions, kv_caches, attn_metadata, inputs_embeds=None
    ) -> torch.Tensor:
        return self.model(input_ids, positions, kv_caches, attn_metadata, inputs_embeds)

    @property
    def backbone(self) -> nn.Module:
        return self.model

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_tokens(input_ids)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    # Reuse the shared q/k/v merge (Phi has no gate/up, so that half is a no-op).
    fuse_state_dict = staticmethod(LlamaForCausalLM.fuse_state_dict)
