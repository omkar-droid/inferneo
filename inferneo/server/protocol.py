"""OpenAI-compatible request/response schemas (the subset inferneo serves)."""

from __future__ import annotations

import time
import uuid

from pydantic import BaseModel, Field

from inferneo.sampling_params import SamplingParams


def _rid(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex}"


class _SamplingFields(BaseModel):
    max_tokens: int | None = 128
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = -1
    min_p: float = 0.0
    presence_penalty: float = 0.0
    frequency_penalty: float = 0.0
    repetition_penalty: float = 1.0
    seed: int | None = None
    stop: str | list[str] | None = None
    ignore_eos: bool = False
    logprobs: int | None = None
    stream: bool = False

    def to_sampling_params(self) -> SamplingParams:
        stop = [self.stop] if isinstance(self.stop, str) else (self.stop or [])
        return SamplingParams(
            max_tokens=self.max_tokens or 128,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            presence_penalty=self.presence_penalty,
            frequency_penalty=self.frequency_penalty,
            repetition_penalty=self.repetition_penalty,
            seed=self.seed,
            stop=stop,
            ignore_eos=self.ignore_eos,
            logprobs=self.logprobs,
        )


class CompletionRequest(_SamplingFields):
    model: str
    prompt: str | list[str]


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionRequest(_SamplingFields):
    model: str
    messages: list[ChatMessage]


# ---- responses ----


class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class CompletionChoice(BaseModel):
    index: int
    text: str
    finish_reason: str | None = None


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _rid("cmpl"))
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: UsageInfo


class ChatMessageOut(BaseModel):
    role: str = "assistant"
    content: str


class ChatChoice(BaseModel):
    index: int
    message: ChatMessageOut
    finish_reason: str | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: _rid("chatcmpl"))
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[ChatChoice]
    usage: UsageInfo


# ---- streaming chunk shapes ----


class CompletionStreamChoice(BaseModel):
    index: int
    text: str
    finish_reason: str | None = None


class CompletionStreamResponse(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionStreamChoice]


class DeltaMessage(BaseModel):
    role: str | None = None
    content: str | None = None


class ChatStreamChoice(BaseModel):
    index: int
    delta: DeltaMessage
    finish_reason: str | None = None


class ChatCompletionStreamResponse(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatStreamChoice]


class ModelCard(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "inferneo"


class ModelList(BaseModel):
    object: str = "list"
    data: list[ModelCard]
