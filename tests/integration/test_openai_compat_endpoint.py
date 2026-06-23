import json

import pytest


def _deltas(sse_text: str) -> list:
    out = []
    for line in sse_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


@pytest.mark.integration
def test_chat_completions_non_stream(client):
    r = client.post("/v1/chat/completions", json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "chat.completion"
    assert body["id"].startswith("chatcmpl-")
    assert body["choices"][0]["message"]["content"] == "Hello from mock backend"
    assert body["choices"][0]["finish_reason"] == "stop"
    assert "usage" in body


@pytest.mark.integration
def test_chat_completions_uses_model_override(client):
    r = client.post("/v1/chat/completions", json={
        "model": "my-custom-model",
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.json()["model"] == "my-custom-model"


@pytest.mark.integration
def test_chat_completions_ignores_unsupported_params(client):
    r = client.post("/v1/chat/completions", json={
        "messages": [{"role": "user", "content": "hi"}],
        "temperature": 0.7, "max_tokens": 100, "top_p": 0.9,
    })
    assert r.status_code == 200


@pytest.mark.integration
def test_chat_completions_stream(client):
    r = client.post("/v1/chat/completions", json={
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": True,
    })
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert r.text.strip().endswith("data: [DONE]")

    deltas = _deltas(r.text)
    assert deltas[0]["choices"][0]["delta"] == {"role": "assistant"}
    content = "".join(
        d["choices"][0]["delta"].get("content", "") for d in deltas
    )
    assert content == "Hello from mock stream"
    assert deltas[-1]["choices"][0]["finish_reason"] == "stop"
    assert all(d["object"] == "chat.completion.chunk" for d in deltas)


@pytest.mark.integration
def test_chat_completions_ignores_unknown_fields(client):
    # OpenAI clients send fields we don't model (e.g. "user", "n"); ignore them.
    r = client.post("/v1/chat/completions", json={
        "model": "x", "user": "abc", "n": 1,
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 200


@pytest.mark.integration
def test_v1_models_list(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = [m["id"] for m in body["data"]]
    assert ids == ["llama3.2:3b", "mistral:7b"]
    assert all(m["object"] == "model" for m in body["data"])
