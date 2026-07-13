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


@pytest.mark.parametrize(
    "bad,expected",
    [
        ({"temperature": "hot"}, 422),   # wrong type
        ({"temperature": -5}, 422),      # out of range
        ({"top_p": 9}, 422),
        ({"top_k": 0}, 422),             # 0 is not a valid top_k (-1 disables)
        ({"min_p": 2}, 422),
        ({"max_tokens": 0}, 422),
        ({"presence_penalty": 9}, 422),
    ],
)
def test_invalid_params_rejected_cleanly(client, bad, expected):
    """Bad client input must return 4xx with a reason — never a 500, which would
    tell the caller *our* server broke when in fact their request was wrong."""
    r = client.post("/v1/completions", json={"model": "t", "prompt": "hi", **bad})
    assert r.status_code == expected, r.text


def test_bad_request_does_not_kill_the_engine(client):
    """Regression: add_request() used to raise inside the engine thread, which
    tripped _fail_all() and killed the engine for *every* request. One bad prompt
    must never take the server down."""
    long_prompt = "word " * 5000
    r = client.post("/v1/completions", json={"model": "t", "prompt": long_prompt, "max_tokens": 4})
    assert r.status_code == 400  # rejected, with a reason

    # the engine must still serve everyone else
    ok = client.post("/v1/completions", json={"model": "t", "prompt": "hello",
                                              "max_tokens": 5, "temperature": 0})
    assert ok.status_code == 200
    assert ok.json()["choices"][0]["text"]


def test_chat_accepts_multimodal_content_parts(client):
    """OpenAI's list-of-parts content must be accepted (text parts work on a
    text model); the shape itself must not be rejected."""
    r = client.post("/v1/chat/completions", json={
        "model": "t",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "max_tokens": 4, "temperature": 0,
    })
    assert r.status_code == 200, r.text
    assert r.json()["choices"][0]["message"]["content"] is not None


def test_image_on_a_text_model_is_honestly_rejected(client):
    """A text-only model has no vision tower. It must say so with a 501 — never
    silently ignore the image and answer as if it had seen it."""
    r = client.post("/v1/chat/completions", json={
        "model": "t",
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": "what is this?"},
            {"type": "image_url", "image_url": {"url": "https://example.com/x.png"}},
        ]}],
        "max_tokens": 4,
    })
    assert r.status_code == 501
    assert "vision" in r.json()["detail"].lower()


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
