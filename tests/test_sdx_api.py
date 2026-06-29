"""Endpoint tests for SDX-related API paths.

Uses the FastAPI TestClient with a mocked SDXBackend so llama-cpp-python
is never loaded. Also tests pull.resolve_target for SDX catalog entries
and the /models/catalog SDX listing.
"""
from __future__ import annotations

import base64
import json
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock


# ── Fixtures ───────────────────────────────────────────────────────────────


@pytest.fixture
def sdx_mock_backend():
    """A mock that looks like SDXBackend from the endpoint's perspective."""
    backend = MagicMock()
    backend.extract_vision = AsyncMock(return_value="A blue sky with clouds")

    async def fake_stream(messages, model, **kwargs):
        for tok in ["The", " sky", " is", " blue"]:
            yield tok

    backend.stream = fake_stream
    backend.chat = AsyncMock(return_value="The sky is blue")
    backend.available_models = AsyncMock(return_value=["sdx-standard"])
    backend.is_available = AsyncMock(return_value=True)
    return backend


@pytest.fixture
def client_sdx(sdx_mock_backend):
    """TestClient with router.resolve patched to return the SDX mock backend."""
    from freeaiagent.main import app
    from starlette.testclient import TestClient

    async def fake_resolve(**kwargs):
        return sdx_mock_backend, "sdx-standard", 0

    with patch("freeaiagent.endpoints.chat.router.resolve", side_effect=fake_resolve):
        with TestClient(app, raise_server_exceptions=True) as c:
            yield c


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    monkeypatch.setattr("freeaiagent.context.DB_FILE", tmp_path / "context.db")
    monkeypatch.setattr("freeaiagent.config.CONFIG_DIR", tmp_path)
    monkeypatch.setattr("freeaiagent.config.CONFIG_FILE", tmp_path / "config.json")


# ── /chat with SDX backend ─────────────────────────────────────────────────


class TestChatEndpointSDX:
    def test_chat_no_image(self, client_sdx, isolated):
        resp = client_sdx.post("/chat", json={"message": "Hello", "session_id": "s1"})
        assert resp.status_code == 200
        data = resp.json()
        assert "response" in data

    def test_chat_with_image_calls_extract_vision(self, client_sdx, isolated, sdx_mock_backend, tmp_path):
        img_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20  # minimal fake JPEG header
        b64 = base64.b64encode(img_bytes).decode()
        resp = client_sdx.post("/chat", json={
            "message": "What's in the image?",
            "session_id": "s2",
            "image": b64,
        })
        assert resp.status_code == 200
        sdx_mock_backend.extract_vision.assert_called_once()
        call_args = sdx_mock_backend.extract_vision.call_args
        assert call_args[1].get("model") == "sdx-standard" or len(call_args[0]) >= 2

    def test_chat_image_injects_sdx_annotation_in_messages(
        self, client_sdx, isolated, sdx_mock_backend
    ):
        img_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20
        b64 = base64.b64encode(img_bytes).decode()
        resp = client_sdx.post("/chat", json={
            "message": "Describe it",
            "session_id": "s3",
            "image": b64,
        })
        assert resp.status_code == 200
        # SDXBackend.chat was called with messages containing [SDX-Image] injection
        call_msgs = sdx_mock_backend.chat.call_args[0][0]
        sdx_msgs = [m for m in call_msgs if m.get("content", "").startswith("[SDX-Image]:")]
        assert len(sdx_msgs) == 1, "Expected exactly one [SDX-Image] system message"

    def test_chat_no_image_skips_extract_vision(self, client_sdx, isolated, sdx_mock_backend):
        resp = client_sdx.post("/chat", json={"message": "No image here", "session_id": "s4"})
        assert resp.status_code == 200
        sdx_mock_backend.extract_vision.assert_not_called()

    def test_chat_image_data_uri_accepted(self, client_sdx, isolated, sdx_mock_backend):
        img_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20
        b64 = base64.b64encode(img_bytes).decode()
        data_uri = f"data:image/jpeg;base64,{b64}"
        resp = client_sdx.post("/chat", json={
            "message": "Describe",
            "session_id": "s5",
            "image": data_uri,
        })
        assert resp.status_code == 200
        sdx_mock_backend.extract_vision.assert_called_once()


# ── /chat/stream with SDX backend ─────────────────────────────────────────


class TestStreamEndpointSDX:
    def _parse_sse(self, text: str) -> list[dict]:
        events = []
        for line in text.splitlines():
            if line.startswith("data: "):
                raw = line[6:].strip()
                if raw == "[DONE]":
                    events.append({"type": "DONE"})
                else:
                    try:
                        events.append(json.loads(raw))
                    except json.JSONDecodeError:
                        pass
        return events

    def test_stream_yields_tokens(self, client_sdx, isolated):
        resp = client_sdx.post("/chat/stream", json={"message": "Hello", "session_id": "ss1"})
        assert resp.status_code == 200
        events = self._parse_sse(resp.text)
        tokens = [e["token"] for e in events if "token" in e]
        assert tokens == ["The", " sky", " is", " blue"]

    def test_stream_with_image_emits_analyzing_event(self, client_sdx, isolated, sdx_mock_backend):
        img_bytes = b"\xff\xd8\xff\xe0" + b"\x00" * 20
        b64 = base64.b64encode(img_bytes).decode()
        resp = client_sdx.post("/chat/stream", json={
            "message": "What is this?",
            "session_id": "ss2",
            "image": b64,
        })
        assert resp.status_code == 200
        events = self._parse_sse(resp.text)
        analyzing = [e for e in events if e.get("analyzing")]
        assert len(analyzing) == 1
        assert "message" in analyzing[0]

    def test_stream_ends_with_done(self, client_sdx, isolated):
        resp = client_sdx.post("/chat/stream", json={"message": "Hi", "session_id": "ss3"})
        assert resp.status_code == 200
        events = self._parse_sse(resp.text)
        assert events[-1] == {"type": "DONE"}

    def test_stream_no_image_no_analyzing_event(self, client_sdx, isolated):
        resp = client_sdx.post("/chat/stream", json={"message": "Hello", "session_id": "ss4"})
        events = self._parse_sse(resp.text)
        assert not any(e.get("analyzing") for e in events)


# ── pull.resolve_target for SDX entries ───────────────────────────────────


class TestPullTargetSDX:
    def test_resolve_sdx_standard(self):
        from freeaiagent.pull import resolve_target
        from freeaiagent.sdx.catalog import SDX_CATALOG

        pt = resolve_target("sdx-standard")
        assert pt.label == SDX_CATALOG["sdx-standard"]["display"]
        assert pt.phase_labels is not None
        assert "text_model" in pt.phase_labels
        assert "vision_model" in pt.phase_labels

    def test_resolve_sdx_nano(self):
        from freeaiagent.pull import resolve_target
        pt = resolve_target("sdx-nano")
        assert "nano" in pt.label.lower() or pt.label == "SDX Nano"

    def test_resolve_sdx_all_tiers(self):
        from freeaiagent.pull import resolve_target
        for name in ("sdx-nano", "sdx-mini", "sdx-standard", "sdx-plus", "sdx-max"):
            pt = resolve_target(name)
            assert pt.phase_labels is not None

    def test_resolve_unknown_raises(self):
        from freeaiagent.pull import resolve_target
        with pytest.raises(ValueError, match="Unknown model"):
            resolve_target("sdx-nonexistent")

    def test_resolve_regular_model_unchanged(self):
        from freeaiagent.pull import resolve_target
        pt = resolve_target("llama-3.2-3b")
        assert pt.phase_labels is None  # regular models have no phase_labels


# ── /models/catalog SDX listing ────────────────────────────────────────────


class TestModelsCatalogSDX:
    def test_catalog_includes_sdx_entries(self, client_sdx, isolated):
        with patch("freeaiagent.sdx.catalog.is_installed", return_value=False):
            resp = client_sdx.get("/models/catalog")
        assert resp.status_code == 200
        models = resp.json()["models"]
        sdx_models = [m for m in models if m["kind"] == "sdx"]
        assert len(sdx_models) == 5

    def test_sdx_entries_have_required_fields(self, client_sdx, isolated):
        with patch("freeaiagent.sdx.catalog.is_installed", return_value=False):
            resp = client_sdx.get("/models/catalog")
        models = resp.json()["models"]
        for m in [m for m in models if m["kind"] == "sdx"]:
            assert "name" in m
            assert "display" in m
            assert "size_gb" in m
            assert "min_ram_gb" in m
            assert "installed" in m

    def test_sdx_installed_flag_true(self, client_sdx, isolated):
        with patch("freeaiagent.sdx.catalog.is_installed", return_value=True):
            resp = client_sdx.get("/models/catalog")
        models = resp.json()["models"]
        sdx_models = [m for m in models if m["kind"] == "sdx"]
        assert all(m["installed"] for m in sdx_models)

    def test_sdx_installed_flag_false(self, client_sdx, isolated):
        with patch("freeaiagent.sdx.catalog.is_installed", return_value=False):
            resp = client_sdx.get("/models/catalog")
        models = resp.json()["models"]
        sdx_models = [m for m in models if m["kind"] == "sdx"]
        assert all(not m["installed"] for m in sdx_models)


# ── SDXDownloadHelper ─────────────────────────────────────────────────────


class TestSDXDownloadHelper:
    @pytest.fixture
    def helper(self, tmp_path):
        from freeaiagent.sdx.downloader import SDXDownloadHelper
        from freeaiagent.sdx.catalog import SDX_CATALOG
        entry = SDX_CATALOG["sdx-nano"]
        with patch("freeaiagent.sdx.downloader.SDX_DIR", tmp_path):
            return SDXDownloadHelper("sdx-nano", entry), tmp_path

    def test_skip_if_already_exists(self, tmp_path):
        from freeaiagent.sdx.downloader import SDXDownloadHelper
        from freeaiagent.sdx.catalog import SDX_CATALOG
        entry = SDX_CATALOG["sdx-nano"]

        dest = tmp_path / "sdx-nano"
        dest.mkdir()
        # Pre-create the files so the downloader skips them
        (dest / "text.gguf").write_bytes(b"fake")
        (dest / "vision.gguf").write_bytes(b"fake")

        helper = SDXDownloadHelper("sdx-nano", entry)
        helper._dest = dest  # override to tmp_path

        chunks = []
        def on_chunk(done, total, phase):
            chunks.append((done, total, phase))

        # No HTTP request should be made — patch requests.get to fail if called
        with patch("requests.get", side_effect=RuntimeError("should not be called")):
            helper._fetch("http://example.com/file.gguf", dest / "text.gguf",
                         phase="text_model", force=False, on_chunk=on_chunk)
        # Callback is called with file size
        assert any(p == "text_model" for _, _, p in chunks)

    def test_phase_labels_for_all_models(self):
        from freeaiagent.sdx.downloader import _sdx_phase_labels
        from freeaiagent.sdx.catalog import SDX_CATALOG
        for name, entry in SDX_CATALOG.items():
            labels = _sdx_phase_labels(entry)
            assert "text_model" in labels
            assert "vision_model" in labels
            assert "mmproj" in labels
