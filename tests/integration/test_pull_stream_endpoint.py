import json

import pytest

from freeaiagent import pull as pull_mod


def _events(sse_text: str) -> list:
    out = []
    for line in sse_text.splitlines():
        if not line.startswith("data:"):
            continue
        payload = line[len("data:"):].strip()
        if payload == "[DONE]":
            continue
        out.append(json.loads(payload))
    return out


class _FakeBackend:
    """Stand-in backend whose download() drives the real ProgressEmitter."""

    def __init__(self, chunks, *, raise_exc=None):
        self.chunks = chunks            # list of (done, total, phase)
        self.raise_exc = raise_exc
        self.download_called_with = None

    def download(self, force=False, on_chunk=None):
        self.download_called_with = force
        for done, total, phase in self.chunks:
            if on_chunk:
                on_chunk(done, total, phase)
        if self.raise_exc:
            raise self.raise_exc
        return "/home/user/.freeaiagent/models/model.gguf"


def _patch_target(monkeypatch, backend, label="my-model"):
    def fake_resolve(target, port=8080):
        return pull_mod.PullTarget(backend, label, 4.7, 8, is_catalog_name=True)
    monkeypatch.setattr("freeaiagent.pull.resolve_target", fake_resolve)


@pytest.mark.integration
def test_pull_stream_emits_start_progress_done(client, monkeypatch):
    backend = _FakeBackend([(50, 100, "model"), (100, 100, "model")])
    _patch_target(monkeypatch, backend)

    r = client.post("/pull/stream", json={"model": "qwen2.5-7b"})
    assert r.status_code == 200
    assert "text/event-stream" in r.headers["content-type"]
    assert r.text.strip().endswith("data: [DONE]")

    evs = _events(r.text)
    types = [e["type"] for e in evs]
    assert types[0] == "start"
    assert "progress" in types
    assert evs[-1]["type"] == "done"
    assert evs[-1]["path"].endswith("model.gguf")


@pytest.mark.integration
def test_pull_stream_engine_then_model_phases(client, monkeypatch):
    backend = _FakeBackend([(100, 100, "engine"), (100, 100, "model")])
    _patch_target(monkeypatch, backend)

    r = client.post("/pull/stream", json={"model": "qwen2.5-7b"})
    evs = _events(r.text)
    phases = [e["phase"] for e in evs if e["type"] == "start"]
    assert phases == ["engine", "model"]


@pytest.mark.integration
def test_pull_stream_forwards_force(client, monkeypatch):
    backend = _FakeBackend([(100, 100, "model")])
    _patch_target(monkeypatch, backend)

    client.post("/pull/stream", json={"model": "qwen2.5-7b", "force": True})
    assert backend.download_called_with is True


@pytest.mark.integration
def test_pull_stream_error_event_on_failure(client, monkeypatch):
    backend = _FakeBackend([(10, 100, "model")], raise_exc=OSError("boom"))
    _patch_target(monkeypatch, backend)

    r = client.post("/pull/stream", json={"model": "qwen2.5-7b"})
    evs = _events(r.text)
    assert evs[-1] == {"type": "error", "message": "boom"}
    assert r.text.strip().endswith("data: [DONE]")


@pytest.mark.integration
def test_pull_stream_unknown_model_returns_400(client):
    r = client.post("/pull/stream", json={"model": "no-such-model"})
    assert r.status_code == 400
    assert "Unknown model" in r.json()["detail"]


@pytest.mark.integration
def test_pull_stream_409_when_download_in_progress(client, monkeypatch):
    backend = _FakeBackend([(100, 100, "model")])
    _patch_target(monkeypatch, backend)

    from freeaiagent.endpoints import models as models_ep
    models_ep._download_lock.acquire()
    try:
        r = client.post("/pull/stream", json={"model": "qwen2.5-7b"})
        assert r.status_code == 409
    finally:
        models_ep._download_lock.release()
