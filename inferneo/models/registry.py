"""HF architecture string -> inferneo model class."""

from __future__ import annotations

from inferneo.models.llama import LlamaForCausalLM
from inferneo.models.qwen import Qwen2ForCausalLM, Qwen3ForCausalLM

# Adding a model family is a registry entry + (at most) a small attention subclass;
# the rest of the engine is model-agnostic.
# Mistral shares Llama's computation for contexts within its sliding window
# (4096 on v0.1); proper sliding-window attention is future work.
MODEL_REGISTRY: dict[str, type] = {
    "LlamaForCausalLM": LlamaForCausalLM,
    "MistralForCausalLM": LlamaForCausalLM,
    "Qwen2ForCausalLM": Qwen2ForCausalLM,
    "Qwen3ForCausalLM": Qwen3ForCausalLM,
}


def get_model_class(architectures: list[str] | None) -> type:
    for arch in architectures or []:
        if arch in MODEL_REGISTRY:
            return MODEL_REGISTRY[arch]
    raise ValueError(
        f"no inferneo implementation for architectures {architectures!r}; "
        f"supported: {sorted(MODEL_REGISTRY)}"
    )
