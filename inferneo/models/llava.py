"""LLaVA: a CLIP vision tower + projector bolted onto our Llama engine.

LLaVA-1.5's language model *is* LlamaForCausalLM — which inferneo already runs
with paged KV, continuous batching, and CUDA graphs. So we don't implement a new
LLM; we only add the eyes:

    image ──► CLIP tower ──► projector ──► [576, hidden] image embeddings
                                                │
    "…<image>…" ──► token ids ──► embed_tokens ─┤  text embeddings
                                                ▼
                    splice image rows in where the <image> tokens sit
                                                ▼
                        one [seq, hidden] embedding sequence ──► the Llama engine

After the splice an image is *just rows in the sequence*: the KV cache, scheduler
and attention never learn it was ever a picture.

The vision tower runs once per image at prefill (~7.6 ms) and is ~1% of the work
for a typical answer, so it stays in PyTorch (which already calls cuDNN/cuBLAS) —
optimising it would be optimising 1%.
"""

from __future__ import annotations

import torch
from torch import nn


class LlavaVision(nn.Module):
    """CLIP vision tower + multimodal projector, loaded from HF LLaVA weights."""

    def __init__(self, hf_config, device: torch.device, dtype: torch.dtype):
        super().__init__()
        from transformers import CLIPVisionModel

        vision_cfg = hf_config.vision_config
        self.tower = CLIPVisionModel(vision_cfg).to(dtype)
        # LLaVA-1.5's projector is a 2-layer MLP: vision_hidden -> text_hidden.
        text_hidden = hf_config.text_config.hidden_size
        self.projector = nn.Sequential(
            nn.Linear(vision_cfg.hidden_size, text_hidden),
            nn.GELU(),
            nn.Linear(text_hidden, text_hidden),
        ).to(dtype)
        # LLaVA drops the CLS token and takes the penultimate layer's patches.
        self.feature_layer = getattr(hf_config, "vision_feature_layer", -2)
        self.select = getattr(hf_config, "vision_feature_select_strategy", "default")
        self.device_ = device
        self.dtype_ = dtype
        self.to(device).eval()

    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """[n_images, 3, H, W] -> [n_images, n_patches, text_hidden]."""
        pixel_values = pixel_values.to(self.device_, self.dtype_)
        out = self.tower(pixel_values, output_hidden_states=True)
        feats = out.hidden_states[self.feature_layer]
        if self.select == "default":
            feats = feats[:, 1:]  # drop CLS; keep the patch tokens
        return self.projector(feats)


def load_vision(model_config, hf_config, device, dtype) -> LlavaVision:
    """Build the vision tower and load its weights from the LLaVA checkpoint."""
    from safetensors.torch import load_file

    from inferneo.models.loader import _weight_files

    vision = LlavaVision(hf_config, device, dtype)

    state: dict[str, torch.Tensor] = {}
    for f in _weight_files(model_config):
        state.update(load_file(f))

    # The checkpoint nests the tower under `vision_tower.` (sometimes with a
    # further `vision_model.`), while CLIPVisionModel's own key names vary by
    # transformers version. Adapt to whatever *this* model actually wants.
    want = set(vision.tower.state_dict().keys())
    tower, proj = {}, {}
    for k, v in state.items():
        if "vision_tower." in k:
            name = k.split("vision_tower.", 1)[1]
            if name not in want and name.startswith("vision_model."):
                name = name[len("vision_model.") :]
            tower[name] = v
        elif "multi_modal_projector." in k:
            # linear_1/linear_2 -> our Sequential's 0/2
            tail = k.split("multi_modal_projector.", 1)[1]
            tail = tail.replace("linear_1.", "0.").replace("linear_2.", "2.")
            proj[tail] = v

    if not tower:
        raise RuntimeError("no vision-tower weights found in the checkpoint")
    # STRICT on purpose. Loading these non-strictly once left the tower on random
    # weights: shapes all lined up, nothing crashed, and the model confidently
    # described an image it had never seen. A weight that does not load is a bug,
    # not a warning.
    vision.tower.load_state_dict(tower, strict=True)
    vision.projector.load_state_dict(proj, strict=True)
    return vision


def splice_image_embeds(
    text_embeds: torch.Tensor,     # [seq, hidden] — embed_tokens(input_ids)
    input_ids: torch.Tensor,       # [seq]
    image_embeds: torch.Tensor,    # [n_images, n_patches, hidden]
    image_token_id: int,
) -> torch.Tensor:
    """Replace the embedding of every <image> placeholder token with the image's
    patch embeddings, in order. HF's processor already expands one <image> into
    n_patches placeholders, so the counts line up exactly."""
    mask = input_ids == image_token_id
    n_slots = int(mask.sum())
    flat = image_embeds.reshape(-1, image_embeds.shape[-1])
    if n_slots != flat.shape[0]:
        raise ValueError(
            f"{n_slots} <image> placeholder tokens but {flat.shape[0]} image "
            f"patch embeddings — the processor and model disagree"
        )
    out = text_embeds.clone()
    out[mask] = flat.to(out.dtype)
    return out
