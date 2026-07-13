"""FastAPI OpenAI-compatible server.

Endpoints: /v1/completions, /v1/chat/completions (both streaming + not),
/v1/models, /health. Streaming uses Server-Sent Events with the OpenAI
``data: {...}\\n\\n`` framing and a terminal ``data: [DONE]``.
"""

from __future__ import annotations

import os
import time
import uuid
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse

from inferneo.engine.async_engine import AsyncEngine
from inferneo.server.protocol import (
    ChatChoice,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    ChatMessageOut,
    ChatStreamChoice,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    CompletionStreamResponse,
    DeltaMessage,
    ModelCard,
    ModelList,
    UsageInfo,
)

# Checking the client connection costs an extra event-loop hop (Starlette spins up
# an anyio cancel scope), and doing it per token dominated the per-token cost once
# many streams ran concurrently. Sampling it every N tokens is enough: a client that
# vanishes is also caught when the generator closes, which aborts the request.
_DISCONNECT_CHECK_EVERY = int(os.environ.get("INFERNEO_DISCONNECT_EVERY", "16"))


async def _dropped(raw, tick: int) -> bool:
    return tick % _DISCONNECT_CHECK_EVERY == 0 and await raw.is_disconnected()


def _check_length(engine: AsyncEngine, prompt_ids: list[int]) -> None:
    """Reject an over-long prompt here, with a 400, rather than letting the engine
    raise for it later — by then a streaming response has already begun, and the
    caller would see a 500 for what is really their mistake."""
    limit = engine.max_model_len
    if len(prompt_ids) >= limit:
        raise HTTPException(
            status_code=400,
            detail=f"prompt is {len(prompt_ids)} tokens; max_model_len is {limit}",
        )


def build_app(engine: AsyncEngine) -> FastAPI:
    app = FastAPI(title="inferneo")

    @app.on_event("startup")
    async def _startup() -> None:
        engine.start()

    @app.on_event("shutdown")
    async def _shutdown() -> None:
        engine.shutdown()

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.get("/v1/models")
    async def models() -> ModelList:
        return ModelList(data=[ModelCard(id=engine.model_name)])

    # ---------------- completions ----------------

    @app.post("/v1/completions")
    async def completions(req: CompletionRequest, raw: Request):
        prompt = req.prompt[0] if isinstance(req.prompt, list) else req.prompt
        params = req.to_sampling_params()
        request_id = f"cmpl-{uuid.uuid4().hex}"
        prompt_ids = engine.tokenizer.encode(prompt)
        _check_length(engine, prompt_ids)

        if req.stream:
            return StreamingResponse(
                _stream_completion(engine, prompt, params, request_id, req.model, raw),
                media_type="text/event-stream",
            )

        text, finish, n_out = await _collect(engine, prompt, params, request_id, raw)
        return CompletionResponse(
            model=req.model,
            choices=[CompletionChoice(index=0, text=text, finish_reason=finish)],
            usage=_usage(len(prompt_ids), n_out),
        )

    # ---------------- chat completions ----------------

    @app.post("/v1/chat/completions")
    async def chat_completions(req: ChatCompletionRequest, raw: Request):
        messages = [{"role": m.role, "content": m.content} for m in req.messages]
        if engine.tokenizer.has_chat_template():
            prompt_ids = engine.tokenizer.apply_chat_template(messages)
        else:
            joined = "\n".join(f"{m['role']}: {m['content']}" for m in messages)
            prompt_ids = engine.tokenizer.encode(joined + "\nassistant:")
        params = req.to_sampling_params()
        request_id = f"chatcmpl-{uuid.uuid4().hex}"
        _check_length(engine, prompt_ids)

        if req.stream:
            return StreamingResponse(
                _stream_chat(engine, prompt_ids, params, request_id, req.model, raw),
                media_type="text/event-stream",
            )

        text, finish, n_out = await _collect(engine, prompt_ids, params, request_id, raw)
        return ChatCompletionResponse(
            id=request_id,
            model=req.model,
            choices=[
                ChatChoice(
                    index=0,
                    message=ChatMessageOut(content=text),
                    finish_reason=finish,
                )
            ],
            usage=_usage(len(prompt_ids), n_out),
        )

    return app


# ---------------- shared helpers ----------------


def _usage(prompt_tokens: int, completion_tokens: int) -> UsageInfo:
    return UsageInfo(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )


async def _collect(engine, prompt, params, request_id, raw) -> tuple[str, str | None, int]:
    """Non-streaming: run to completion, detokenize once."""
    detok = engine.tokenizer.incremental_detokenizer()
    finish, n_out, tick = None, 0, 0
    async for out in engine.generate(prompt, params, request_id):
        tick += 1
        if await _dropped(raw, tick):
            await engine.abort(request_id)
            break
        comp = out.outputs[0]
        detok.decode(comp.token_ids)
        n_out = len(comp.token_ids)
        finish = comp.finish_reason
    text = detok.text
    for s in params.stop:
        i = text.find(s)
        if i != -1:
            text, finish = text[:i], "stop"
    return text, finish, n_out


async def _stream_completion(
    engine, prompt, params, request_id, model, raw
) -> AsyncIterator[str]:
    created = int(time.time())
    detok = engine.tokenizer.incremental_detokenizer()
    tick = 0
    async for out in engine.generate(prompt, params, request_id):
        tick += 1
        if await _dropped(raw, tick):
            await engine.abort(request_id)
            return
        comp = out.outputs[0]
        delta = detok.decode(comp.token_ids)
        if delta:
            chunk = CompletionStreamResponse(
                id=request_id, created=created, model=model,
                choices=[CompletionStreamChoice(index=0, text=delta)],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
        if comp.finish_reason:
            final = CompletionStreamResponse(
                id=request_id, created=created, model=model,
                choices=[CompletionStreamChoice(index=0, text="", finish_reason=comp.finish_reason)],
            )
            yield f"data: {final.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


async def _stream_chat(
    engine, prompt_ids, params, request_id, model, raw
) -> AsyncIterator[str]:
    created = int(time.time())
    detok = engine.tokenizer.incremental_detokenizer()
    # First chunk carries the assistant role.
    first = ChatCompletionStreamResponse(
        id=request_id, created=created, model=model,
        choices=[ChatStreamChoice(index=0, delta=DeltaMessage(role="assistant"))],
    )
    yield f"data: {first.model_dump_json()}\n\n"
    tick = 0
    async for out in engine.generate(prompt_ids, params, request_id):
        tick += 1
        if await _dropped(raw, tick):
            await engine.abort(request_id)
            return
        comp = out.outputs[0]
        delta = detok.decode(comp.token_ids)
        if delta:
            chunk = ChatCompletionStreamResponse(
                id=request_id, created=created, model=model,
                choices=[ChatStreamChoice(index=0, delta=DeltaMessage(content=delta))],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"
        if comp.finish_reason:
            final = ChatCompletionStreamResponse(
                id=request_id, created=created, model=model,
                choices=[ChatStreamChoice(index=0, delta=DeltaMessage(), finish_reason=comp.finish_reason)],
            )
            yield f"data: {final.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"
