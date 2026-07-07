"""Tokenizer wrapper around HF tokenizers: encode/decode, chat templates, and
incremental (streaming) detokenization."""

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

    @property
    def hf(self):
        return self._tok

    def encode(self, text: str) -> list[int]:
        return self._tok(text).input_ids

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return self._tok.decode(token_ids, skip_special_tokens=skip_special_tokens)

    def has_chat_template(self) -> bool:
        return self._tok.chat_template is not None

    def apply_chat_template(self, messages: list[dict]) -> list[int]:
        """Render chat messages to prompt token ids (adds the generation prompt).

        Renders to text first, then encodes without adding extra special tokens
        (the template already includes them) — portable across transformers
        versions, whose ``tokenize=True`` return type has changed.
        """
        text = self._tok.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False
        )
        return self._tok(text, add_special_tokens=False).input_ids

    def incremental_detokenizer(self) -> IncrementalDetokenizer:
        return IncrementalDetokenizer(self._tok)


class IncrementalDetokenizer:
    """Turns a growing token-id stream into text deltas.

    Decoding token-by-token is wrong (many tokens are sub-word or multi-byte
    fragments). This keeps a small window of recent tokens and emits the newly
    stable text, holding back a trailing fragment that isn't valid UTF-8 yet
    (a partial multi-byte character) until the next token completes it.
    """

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._tokens: list[int] = []
        self._text = ""

    def decode(self, all_token_ids: list[int]) -> str:
        """Feed the full output-token list so far; return the new text delta."""
        self._tokens = all_token_ids
        full = self._tok.decode(all_token_ids, skip_special_tokens=True)
        # Hold back an incomplete trailing UTF-8 char (shows as U+FFFD).
        if full.endswith("�"):
            return ""
        delta = full[len(self._text) :]
        self._text = full
        return delta

    @property
    def text(self) -> str:
        return self._text
