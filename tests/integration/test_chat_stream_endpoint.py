import json

import pytest


def _tokens(sse_text: str) -> list[str]:
    out = []
    for line in sse_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        obj = json.loads(payload)
        if "token" in obj:
            out.append(obj["token"])
    return out


@pytest.mark.integration
def test_stream_emits_tokens_and_done(client):
    r = client.post("/chat/stream", json={"message": "hi"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert r.text.strip().endswith("data: [DONE]")
    assert _tokens(r.text) == ["Hello", " from", " mock", " stream"]


@pytest.mark.integration
def test_stream_persists_full_response(client):
    client.post("/chat/stream", json={"message": "hi", "session_id": "s1"})
    ctx = client.get("/context", params={"session": "s1"}).json()
    contents = [m["content"] for m in ctx["messages"]]
    assert "hi" in contents
    assert "Hello from mock stream" in contents  # joined tokens persisted


@pytest.mark.integration
def test_stream_respects_x_caller_id(client):
    r = client.post("/chat/stream", json={"message": "hi"}, headers={"X-Caller-ID": "streamer"})
    assert r.status_code == 200
    ctx = client.get("/context", params={"session": "streamer"}).json()
    assert ctx["total"] >= 2


@pytest.mark.integration
def test_stream_backend_unavailable_returns_503(client):
    r = client.post("/chat/stream", json={"message": "hi", "backend": "nonexistent"})
    assert r.status_code == 503
