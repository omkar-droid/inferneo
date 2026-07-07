"""Thin tokenizer wrapper around HF tokenizers.

Phase 1 scope: encode/decode for the offline path. Incremental (streaming)
detokenization and chat templates arrive with the server.
"""

from __future__ import annotations

from transformers import AutoTokenizer

from inferneo.config import ModelConfig


class TokenizerWrapper:
    def __init__(self, model_config: ModelConfig):
        self._tok = AutoTokenizer.from_pretrained(
            model_config.model,
            revision=model_config.revision,
            trust_remote_code=model_config.trust_remote_code,
        )

    @property
    def eos_token_id(self) -> int | None:
        return self._tok.eos_token_id

    def encode(self, text: str) -> list[int]:
        return self._tok(text).input_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)
