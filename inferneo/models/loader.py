"""Model construction + HF safetensors weight loading."""

from __future__ import annotations

import os
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig

from inferneo.attention.interface import AttentionBackend
from inferneo.config import ModelConfig
from inferneo.models.registry import get_model_class


def load_hf_config(model_config: ModelConfig):
    return AutoConfig.from_pretrained(
        model_config.model,
        revision=model_config.revision,
        trust_remote_code=model_config.trust_remote_code,
    )


def _weight_files(model_config: ModelConfig) -> list[Path]:
    if os.path.isdir(model_config.model):
        root = Path(model_config.model)
    else:
        root = Path(
            snapshot_download(
                model_config.model,
                revision=model_config.revision,
                allow_patterns=["*.safetensors", "*.json"],
            )
        )
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"no .safetensors weights found under {root}")
    return files


def is_multimodal(hf_config) -> bool:
    """A vision-language checkpoint nests the LLM's config under `text_config`."""
    return hasattr(hf_config, "text_config") and hasattr(hf_config, "vision_config")


def text_config(hf_config):
    return hf_config.text_config if is_multimodal(hf_config) else hf_config


def _strip_language_model(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """In a LLaVA checkpoint the Llama weights sit under a `language_model.` prefix
    (the layout differs across transformers versions). Map them onto the plain
    Llama names our model expects, and drop the vision/projector weights, which
    the vision tower loads separately."""
    out: dict[str, torch.Tensor] = {}
    for k, v in state.items():
        if "vision_tower" in k or "multi_modal_projector" in k:
            continue
        if k.startswith("model.language_model."):      # newer layout
            out["model." + k[len("model.language_model.") :]] = v
        elif k.startswith("language_model.model."):    # older layout
            out["model." + k[len("language_model.model.") :]] = v
        elif k.startswith("language_model.lm_head."):
            out["lm_head." + k[len("language_model.lm_head.") :]] = v
        else:
            out[k] = v
    return out


def load_model(
    model_config: ModelConfig,
    hf_config,
    backend: AttentionBackend,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.nn.Module:
    from safetensors.torch import load_file

    llm_config = text_config(hf_config)
    model_cls = get_model_class(getattr(llm_config, "architectures", None) or ["LlamaForCausalLM"])
    model = model_cls(llm_config, backend).to(dtype)

    state: dict[str, torch.Tensor] = {}
    for f in _weight_files(model_config):
        state.update(load_file(f))
    if is_multimodal(hf_config):
        state = _strip_language_model(state)
    if hasattr(model, "fuse_state_dict"):
        state = model.fuse_state_dict(state)
    missing, unexpected = model.load_state_dict(state, strict=False)

    tied = getattr(hf_config, "tie_word_embeddings", False)
    if tied and "lm_head.weight" in missing:
        model.tie_weights()
        missing.remove("lm_head.weight")
    unexpected = [u for u in unexpected if "rotary_emb.inv_freq" not in u]
    if missing or unexpected:
        raise RuntimeError(
            f"weight mismatch loading {model_config.model}: "
            f"missing={missing[:5]} unexpected={unexpected[:5]}"
        )
    return model.to(device).eval()
