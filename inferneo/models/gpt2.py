"""GPT-2 (``GPT2LMHeadModel``) — the classic pre-RoPE decoder. Proves the engine
isn't tied to rotary models: positions here are a *learned* embedding table
(``wpe``) added to the token embeddings, so the paged-attention backend sees
plain q/k/v with no rotation at all.

Other departures from Llama:
    * **LayerNorm** (with bias), not RMSNorm.
    * **Non-gated MLP** — ``c_proj(gelu(c_fc(x)))``.
    * **Multi-head attention** (no GQA); fused ``c_attn`` already produces q|k|v.
    * **Conv1D weights** — HF stores GPT-2's linears transposed (``[in, out]``);
      ``fuse_state_dict`` transposes them into ``nn.Linear``'s ``[out, in]``.

Module names mirror HF (``transformer.wte`` / ``wpe`` / ``h.{i}`` / ``ln_f``) so
weights load by name after the transpose. ``lm_head`` is tied to ``wte``.
"""

from __future__ import annotations

import torch
from torch import nn

from inferneo.attention.interface import AttentionBackend


class GPT2Attention(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        self.embed_dim = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.embed_dim // self.num_heads
        self.c_attn = nn.Linear(self.embed_dim, 3 * self.embed_dim, bias=True)
        self.c_proj = nn.Linear(self.embed_dim, self.embed_dim, bias=True)
        self.backend = backend

    def forward(self, x, kv_cache, attn_metadata):
        t = x.shape[0]
        q, k, v = self.c_attn(x).split(self.embed_dim, dim=-1)
        q = q.view(t, self.num_heads, self.head_dim)
        k = k.view(t, self.num_heads, self.head_dim)
        v = v.view(t, self.num_heads, self.head_dim)
        out = self.backend.forward(q, k, v, kv_cache, attn_metadata)  # no RoPE
        return self.c_proj(out.reshape(t, self.embed_dim))


class GPT2MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        inner = config.n_inner or 4 * config.hidden_size
        self.c_fc = nn.Linear(config.hidden_size, inner, bias=True)
        self.c_proj = nn.Linear(inner, config.hidden_size, bias=True)
        self.act_fn = nn.GELU(approximate="tanh")  # HF "gelu_new"

    def forward(self, x):
        return self.c_proj(self.act_fn(self.c_fc(x)))


class GPT2Block(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        eps = config.layer_norm_epsilon
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.attn = GPT2Attention(config, backend)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=eps)
        self.mlp = GPT2MLP(config)

    def forward(self, x, kv_cache, attn_metadata):
        x = x + self.attn(self.ln_1(x), kv_cache, attn_metadata)
        return x + self.mlp(self.ln_2(x))


class GPT2Model(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        self.wte = nn.Embedding(config.vocab_size, config.hidden_size)
        self.wpe = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.h = nn.ModuleList(
            GPT2Block(config, backend) for _ in range(config.num_hidden_layers)
        )
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_epsilon)

    def forward(self, input_ids, positions, kv_caches, attn_metadata, inputs_embeds=None):
        x = inputs_embeds if inputs_embeds is not None else self.wte(input_ids)
        x = x + self.wpe(positions)  # learned absolute positions
        for block, kv_cache in zip(self.h, kv_caches):
            x = block(x, kv_cache, attn_metadata)
        return self.ln_f(x)


class GPT2LMHeadModel(nn.Module):
    def __init__(self, config, backend: AttentionBackend):
        super().__init__()
        self.config = config
        self.transformer = GPT2Model(config, backend)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def forward(
        self, input_ids, positions, kv_caches, attn_metadata, inputs_embeds=None
    ) -> torch.Tensor:
        return self.transformer(input_ids, positions, kv_caches, attn_metadata, inputs_embeds)

    @property
    def backbone(self) -> nn.Module:
        return self.transformer  # GPT-2 names its backbone `transformer`, not `model`

    def embed(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.transformer.wte(input_ids)

    def compute_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden)

    def tie_weights(self) -> None:
        self.lm_head.weight = self.transformer.wte.weight

    @staticmethod
    def fuse_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Transpose GPT-2's Conv1D weights into nn.Linear layout, drop HF's
        causal-mask buffers (`.attn.bias` / `.attn.masked_bias`), and normalize the
        module prefix. The distributed `gpt2` checkpoint stores *base-model* keys
        (`wte.weight`, `h.0...`) with no `transformer.` prefix; `save_pretrained`
        output already has it. Either way we end up at `transformer.*`."""
        conv1d = (
            ".attn.c_attn.weight",
            ".attn.c_proj.weight",
            ".mlp.c_fc.weight",
            ".mlp.c_proj.weight",
        )
        out: dict[str, torch.Tensor] = {}
        for k, v in state.items():
            if k.endswith((".attn.bias", ".attn.masked_bias")):
                continue
            if k.endswith(conv1d):
                v = v.t().contiguous()  # [in, out] -> [out, in]
            if not k.startswith(("transformer.", "lm_head.")):
                k = "transformer." + k
            out[k] = v
        return out
