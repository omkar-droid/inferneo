"""Turns request content (text / image / audio) into an ``EngineInput``.

This is the only place that knows about modalities. Add a new one here; the
engine below never changes.
"""

from __future__ import annotations

import base64
import io
from typing import Any

import torch

from inferneo.inputs.types import EngineInput


class UnsupportedModality(Exception):
    """Raised when a request needs a model we haven't loaded (e.g. audio without
    an ASR model). Surfaced to the caller as a 501 — never silently ignored."""


def _load_image(url: str):
    """Accepts a data: URI (base64) or an http(s) URL."""
    from PIL import Image

    if url.startswith("data:"):
        _, b64 = url.split(",", 1)
        return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
    if url.startswith(("http://", "https://")):
        import urllib.request

        # Many hosts (Wikipedia among them) 403 the default urllib user-agent.
        req = urllib.request.Request(url, headers={"User-Agent": "inferneo/0.1"})
        with urllib.request.urlopen(req, timeout=15) as r:  # noqa: S310
            return Image.open(io.BytesIO(r.read())).convert("RGB")
    raise ValueError(f"unsupported image url: {url[:40]}...")


class InputProcessor:
    """Text-only by default. If the model is multimodal, images light up too."""

    def __init__(self, tokenizer, vision=None, hf_processor=None, embedder=None,
                 image_token_id: int | None = None):
        self.tokenizer = tokenizer
        self.vision = vision              # LlavaVision, or None for a text model
        self.hf_processor = hf_processor  # HF LlavaProcessor (does the image resize/normalize)
        self.embedder = embedder          # model.embed — token ids -> embeddings
        self.image_token_id = image_token_id

    @property
    def supports_images(self) -> bool:
        return self.vision is not None

    # ------------------------------------------------------------------ #

    def from_text(self, text: str) -> EngineInput:
        return EngineInput(token_ids=self.tokenizer.encode(text), text=text)

    def from_chat(self, messages: list[dict[str, Any]]) -> EngineInput:
        """OpenAI chat messages, where `content` is either a string or a list of
        parts ({"type": "text"|"image_url"|"input_audio", ...})."""
        images, texts = [], []
        for msg in messages:
            content = msg.get("content")
            if isinstance(content, str):
                texts.append((msg["role"], content))
                continue
            for part in content or []:
                kind = part.get("type")
                if kind == "text":
                    texts.append((msg["role"], part.get("text", "")))
                elif kind == "image_url":
                    if not self.supports_images:
                        raise UnsupportedModality(
                            "this model has no vision tower; load a multimodal "
                            "model (e.g. llava-hf/llava-1.5-7b-hf) to send images"
                        )
                    images.append(_load_image(part["image_url"]["url"]))
                elif kind in ("input_audio", "audio_url"):
                    raise UnsupportedModality(
                        "audio input needs a speech model (e.g. Whisper), which is "
                        "not loaded; inferneo currently supports text and images"
                    )
                else:
                    raise ValueError(f"unknown content part type: {kind!r}")

        if not images:
            plain = [{"role": r, "content": c} for r, c in texts]
            return EngineInput(token_ids=self.tokenizer.apply_chat_template(plain))
        return self._with_images(texts, images)

    # ------------------------------------------------------------------ #

    @torch.inference_mode()
    def _with_images(self, texts, images) -> EngineInput:
        """Build the prompt with <image> placeholders, run the vision tower, and
        splice the patch embeddings into the text embedding sequence."""
        from inferneo.models.llava import splice_image_embeds

        prompt = "".join("<image>\n" for _ in images)
        prompt += " ".join(t for _, t in texts)
        prompt = f"USER: {prompt} ASSISTANT:"

        # HF's processor expands each <image> into n_patches placeholder tokens and
        # does the resize/normalise — reusing it keeps us bit-compatible with LLaVA.
        enc = self.hf_processor(images=images, text=prompt, return_tensors="pt")
        input_ids = enc["input_ids"][0]
        pixel_values = enc["pixel_values"]

        image_embeds = self.vision(pixel_values)                  # [n_img, patches, hidden]
        text_embeds = self.embedder(input_ids.to(self.vision.device_))
        embeds = splice_image_embeds(
            text_embeds, input_ids.to(self.vision.device_), image_embeds, self.image_token_id
        )
        return EngineInput(
            token_ids=input_ids.tolist(),
            prompt_embeds=embeds,
            text=prompt,
        )
