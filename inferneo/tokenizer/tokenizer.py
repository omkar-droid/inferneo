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
    fragments), but re-decoding the whole sequence every step is O(n^2). This
    uses the prefix/read-offset window (as in vLLM/HF): each step decodes only a
    couple of trailing tokens — the token(s) just added plus one of context for
    correct spacing — and emits the newly stable suffix, holding back an
    incomplete trailing UTF-8 char (U+FFFD) until the next token completes it.
    """

    def __init__(self, tokenizer):
        self._tok = tokenizer
        self._tokens: list[int] = []
        self._text = ""
        self._prefix_offset = 0
        self._read_offset = 0

    def decode(self, all_token_ids: list[int]) -> str:
        """Feed the full output-token list so far; return the new text delta."""
        self._tokens = all_token_ids
        prefix = self._tok.decode(
            all_token_ids[self._prefix_offset : self._read_offset],
            skip_special_tokens=True,
        )
        whole = self._tok.decode(
            all_token_ids[self._prefix_offset :], skip_special_tokens=True
        )
        if len(whole) <= len(prefix) or whole.endswith("�"):
            return ""  # nothing new yet, or a partial multi-byte char
        delta = whole[len(prefix) :]
        self._prefix_offset = self._read_offset
        self._read_offset = len(all_token_ids)
        self._text += delta
        return delta

    @property
    def text(self) -> str:
        return self._text
