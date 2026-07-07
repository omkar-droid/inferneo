"""End-to-end OpenAI API tests: boot the server on the tiny model (CPU) and
exercise completions, chat, and SSE streaming through the real HTTP stack."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture(scope="module")
def client(tiny_model_dir):
    from inferneo.engine.async_engine import AsyncEngine
    from inferneo.server.api_server import build_app

    engine = AsyncEngine.from_model(
        tiny_model_dir, device="cpu", dtype="float32", num_blocks=128,
        max_num_seqs=8, max_num_batched_tokens=256,
    )
    with TestClient(build_app(engine)) as c:
        yield c


def test_health(client):
    assert client.get("/health").json()["status"] == "ok"


def test_models(client, tiny_model_dir):
    data = client.get("/v1/models").json()
    assert data["data"][0]["id"] == tiny_model_dir


def test_completion_nonstreaming(client):
    r = client.post("/v1/completions", json={
        "model": "tiny", "prompt": "hello world",
        "max_tokens": 8, "temperature": 0,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["choices"][0]["text"]  # non-empty
    assert body["choices"][0]["finish_reason"] == "length"
    assert body["usage"]["completion_tokens"] == 8


def test_completion_streaming(client):
    with client.stream("POST", "/v1/completions", json={
        "model": "tiny", "prompt": "hello", "max_tokens": 6, "temperature": 0,
        "stream": True,
    }) as r:
        assert r.status_code == 200
        chunks, done = [], False
        for line in r.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                done = True
                break
            chunks.append(json.loads(payload))
    assert done
    assert any(c["choices"][0].get("finish_reason") == "length" for c in chunks)


def test_chat_streaming(client):
    with client.stream("POST", "/v1/chat/completions", json={
        "model": "tiny",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 6, "temperature": 0, "stream": True,
    }) as r:
        assert r.status_code == 200
        first_role = None
        saw_done = False
        for line in r.iter_lines():
            if not line.startswith("data: "):
                continue
            payload = line[len("data: "):]
            if payload == "[DONE]":
                saw_done = True
                break
            chunk = json.loads(payload)
            delta = chunk["choices"][0]["delta"]
            if first_role is None and delta.get("role"):
                first_role = delta["role"]
    assert first_role == "assistant"
    assert saw_done


def test_deterministic_matches_offline(client, tiny_model_dir):
    """Greedy over HTTP must equal the offline LLM path token-for-text."""
    from inferneo import LLM, SamplingParams

    r = client.post("/v1/completions", json={
        "model": "tiny", "prompt": "the quick brown", "max_tokens": 10,
        "temperature": 0, "ignore_eos": True,
    })
    api_text = r.json()["choices"][0]["text"]

    llm = LLM(tiny_model_dir, device="cpu", dtype="float32", num_blocks=128)
    offline = llm.generate(["the quick brown"],
                           SamplingParams(max_tokens=10, temperature=0, ignore_eos=True))
    assert api_text == offline[0].outputs[0].text
