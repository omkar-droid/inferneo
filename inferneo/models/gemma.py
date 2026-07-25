"""Gemma (v1) — Google's Llama-family variant with three numeric deltas.

The attention block is Llama's unchanged (RoPE, GQA/MQA, an explicit ``head_dim``
that isn't ``hidden/num_heads``, no biases — all already config-driven). What
differs is everything around it:

    * RMSNorm scales by ``(1 + weight)`` and stays in float32 throughout.
    * The FFN is GeGLU — ``gelu_tanh(gate) * up`` instead of SwiGLU's SiLU.
    * The token embeddings are multiplied by ``sqrt(hidden_size)`` before layer 0.

Weights load through the shared ``fuse_state_dict`` (q/k/v -> qkv, gate/up ->
gate_up) and ``lm_head`` is tied to the embeddings.
"""

from __future__ import annotations

from torch import nn

from inferneo.models.layers import GemmaRMSNorm
from inferneo.models.llama import LlamaForCausalLM, LlamaMLP


class GemmaMLP(LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        # HF Gemma uses "gelu_pytorch_tanh": the tanh approximation of GELU.
        self.act_fn = nn.GELU(approximate="tanh")


class GemmaForCausalLM(LlamaForCausalLM):
    mlp_cls = GemmaMLP
    norm_cls = GemmaRMSNorm
    scale_embeddings = True
