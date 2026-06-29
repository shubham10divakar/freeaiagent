"""Integration tests for SDXEngine and SDXBackend with mocked sub-runners.

llama-cpp-python is never imported: VisionRunner and TextRunner are patched
before any SDX code loads.
"""
from __future__ import annotations

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

from freeaiagent.sdx.context_builder import ContextBuilder


# ── ContextBuilder is tested standalone (no mocking needed) ────────────────


class TestContextBuilderIntegration:
    def test_empty_messages_no_crash(self):
        cb = ContextBuilder([])
        out = cb.build()
        assert "Assistant:" in out

    def test_single_user_message(self):
        msgs = [{"role": "user", "content": "Hello"}]
        out = ContextBuilder(msgs).build()
        assert "User: Hello" in out
        assert "Assistant:" in out

    def test_large_budget_keeps_all_history(self):
        msgs = []
        for i in range(10):
            msgs.append({"role": "user", "content": f"User turn {i}"})
            msgs.append({"role": "assistant", "content": f"Assistant reply {i}"})
        msgs.append({"role": "user", "content": "final"})
        out = ContextBuilder(msgs, token_budget=32768).build()
        for i in range(10):
            assert f"User turn {i}" in out

    def test_tight_budget_keeps_only_recent(self):
        msgs = []
        for i in range(8):
            msgs.append({"role": "user", "content": f"message {i}" + " x" * 100})
            msgs.append({"role": "assistant", "content": f"reply {i}" + " x" * 100})
        msgs.append({"role": "user", "content": "latest"})
        out = ContextBuilder(msgs, token_budget=512, min_turns_keep=1).build()
        assert "latest" in out
        # Not all early messages can fit
        early_count = sum(1 for i in range(8) if f"message {i}" in out)
        assert early_count < 8


# ── SDXEngine with mocked runners ─────────────────────────────────────────


@pytest.fixture
def mock_text_runner():
    runner = MagicMock()
    runner.is_loaded.return_value = True
    runner._last_ctx = []

    async def fake_generate(ctx):
        runner._last_ctx.append(ctx)
        for tok in ["Hello", " ", "world"]:
            yield tok

    runner.generate = fake_generate
    runner.load = MagicMock()
    runner.stop = MagicMock()
    runner.unload = MagicMock()
    return runner


@pytest.fixture
def mock_vision_runner():
    runner = MagicMock()
    runner.is_loaded.return_value = False
    runner.extract = AsyncMock(return_value="A red cat sitting on a white sofa")
    runner.load = MagicMock()
    runner.unload = MagicMock()
    return runner


@pytest.fixture
def sdx_engine(mock_text_runner, mock_vision_runner):
    with patch("freeaiagent.sdx.engine.TextRunner", return_value=mock_text_runner), \
         patch("freeaiagent.sdx.engine.VisionRunner", return_value=mock_vision_runner):
        from freeaiagent.sdx.engine import SDXEngine
        engine = SDXEngine(
            text_model_path="/fake/text.gguf",
            vision_model_path="/fake/vision.gguf",
            mmproj_path=None,
            token_budget=4096,
            n_ctx=4096,
        )
        engine._text = mock_text_runner
        engine._vision = mock_vision_runner
        return engine


class TestSDXEngine:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, sdx_engine):
        msgs = [{"role": "user", "content": "Hi"}]
        tokens = []
        async for tok in sdx_engine.stream(msgs):
            tokens.append(tok)
        assert tokens == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_extract_vision_calls_runner(self, sdx_engine, mock_vision_runner):
        desc = await sdx_engine.extract_vision("/tmp/img.jpg", "What is this?")
        mock_vision_runner.extract.assert_called_once_with("/tmp/img.jpg", "What is this?")
        assert desc == "A red cat sitting on a white sofa"

    @pytest.mark.asyncio
    async def test_stream_builds_flat_context(self, sdx_engine, mock_text_runner):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi!"},
            {"role": "user", "content": "Bye"},
        ]
        tokens = []
        async for tok in sdx_engine.stream(msgs):
            tokens.append(tok)
        # generate() was called with a context string that includes both turns
        assert mock_text_runner._last_ctx, "generate() was never called"
        ctx = mock_text_runner._last_ctx[-1]
        assert "Hello" in ctx
        assert "Hi!" in ctx
        assert "Bye" in ctx

    def test_is_loaded(self, sdx_engine, mock_text_runner):
        mock_text_runner.is_loaded.return_value = True
        assert sdx_engine.is_loaded()

    def test_unload_delegates(self, sdx_engine, mock_text_runner, mock_vision_runner):
        sdx_engine.unload()
        mock_text_runner.unload.assert_called_once()
        mock_vision_runner.unload.assert_called_once()

    def test_stop_delegates(self, sdx_engine, mock_text_runner):
        sdx_engine.stop()
        mock_text_runner.stop.assert_called_once()

    @pytest.mark.asyncio
    async def test_load_calls_both_runners(self, sdx_engine, mock_text_runner, mock_vision_runner):
        await sdx_engine.load()
        mock_text_runner.load.assert_called_once()
        mock_vision_runner.load.assert_called_once()


# ── SDXBackend with mocked catalog + engine ────────────────────────────────


@pytest.fixture
def fake_sdx_catalog(tmp_path):
    model_dir = tmp_path / "sdx-standard"
    model_dir.mkdir()
    (model_dir / "text.gguf").touch()
    (model_dir / "vision.gguf").touch()
    return tmp_path


@pytest.fixture
def sdx_backend(fake_sdx_catalog, mock_text_runner, mock_vision_runner):
    from freeaiagent.backends.sdx_backend import SDXBackend, _ENGINES
    _ENGINES.clear()

    catalog_patch = {
        "sdx-standard": {
            "display": "SDX Standard",
            "token_budget": 4096,
            "files": {
                "text": {"url": "...", "size_gb": 2.0},
                "vision": {"url": "...", "size_gb": 2.4},
            },
        }
    }
    with patch("freeaiagent.backends.sdx_backend.SDX_CATALOG", catalog_patch), \
         patch("freeaiagent.backends.sdx_backend.SDX_DIR", fake_sdx_catalog), \
         patch("freeaiagent.backends.sdx_backend.is_installed", return_value=True), \
         patch("freeaiagent.backends.sdx_backend.model_paths", return_value={
             "text": str(fake_sdx_catalog / "sdx-standard" / "text.gguf"),
             "vision": str(fake_sdx_catalog / "sdx-standard" / "vision.gguf"),
             "mmproj": None,
         }), \
         patch("freeaiagent.sdx.engine.TextRunner", return_value=mock_text_runner), \
         patch("freeaiagent.sdx.engine.VisionRunner", return_value=mock_vision_runner):
        backend = SDXBackend({"model": "sdx-standard", "n_gpu_layers": 0})
        # pre-wire the engine so it doesn't try to build paths
        from freeaiagent.sdx.engine import SDXEngine
        engine = SDXEngine.__new__(SDXEngine)
        engine._text = mock_text_runner
        engine._vision = mock_vision_runner
        engine._token_budget = 4096
        _ENGINES["sdx-standard"] = engine
        yield backend
    _ENGINES.clear()


class TestSDXBackend:
    @pytest.mark.asyncio
    async def test_stream_yields_tokens(self, sdx_backend):
        msgs = [{"role": "user", "content": "Hello"}]
        tokens = [t async for t in sdx_backend.stream(msgs, "sdx-standard")]
        assert tokens == ["Hello", " ", "world"]

    @pytest.mark.asyncio
    async def test_chat_joins_tokens(self, sdx_backend):
        msgs = [{"role": "user", "content": "Hello"}]
        response = await sdx_backend.chat(msgs, "sdx-standard")
        assert response == "Hello world"

    @pytest.mark.asyncio
    async def test_available_models(self, sdx_backend):
        with patch("freeaiagent.backends.sdx_backend.is_installed", return_value=True), \
             patch("freeaiagent.backends.sdx_backend.SDX_CATALOG", {"sdx-standard": {}}):
            models = await sdx_backend.available_models()
            assert "sdx-standard" in models

    @pytest.mark.asyncio
    async def test_is_available_true(self, sdx_backend):
        with patch("freeaiagent.backends.sdx_backend.is_installed", return_value=True), \
             patch("freeaiagent.backends.sdx_backend.SDX_CATALOG", {"sdx-standard": {}}):
            assert await sdx_backend.is_available()

    @pytest.mark.asyncio
    async def test_is_available_false_when_no_models(self):
        from freeaiagent.backends.sdx_backend import SDXBackend, _ENGINES
        _ENGINES.clear()
        backend = SDXBackend({"model": "sdx-standard"})
        with patch("freeaiagent.backends.sdx_backend.is_installed", return_value=False), \
             patch("freeaiagent.backends.sdx_backend.SDX_CATALOG", {"sdx-standard": {}}):
            assert not await backend.is_available()

    @pytest.mark.asyncio
    async def test_extract_vision_delegates_to_engine(self, sdx_backend, mock_vision_runner):
        desc = await sdx_backend.extract_vision("/img.jpg", "what is it?", model="sdx-standard")
        mock_vision_runner.extract.assert_called_once_with("/img.jpg", "what is it?")
        assert "cat" in desc


# ── SDX catalog ────────────────────────────────────────────────────────────


class TestSDXCatalog:
    def test_all_five_tiers_present(self):
        from freeaiagent.sdx.catalog import SDX_CATALOG
        expected = {"sdx-nano", "sdx-mini", "sdx-standard", "sdx-plus", "sdx-max"}
        assert expected == set(SDX_CATALOG)

    def test_each_entry_has_required_keys(self):
        from freeaiagent.sdx.catalog import SDX_CATALOG
        required = {"display", "kind", "tier", "min_ram_gb", "size_gb", "token_budget",
                    "description", "files"}
        for name, entry in SDX_CATALOG.items():
            missing = required - entry.keys()
            assert not missing, f"sdx catalog entry '{name}' missing keys: {missing}"

    def test_files_structure(self):
        from freeaiagent.sdx.catalog import SDX_CATALOG
        for name, entry in SDX_CATALOG.items():
            files = entry["files"]
            assert "text" in files, f"{name}: missing files.text"
            assert "vision" in files, f"{name}: missing files.vision"
            assert "url" in files["text"], f"{name}: missing files.text.url"
            assert "url" in files["vision"], f"{name}: missing files.vision.url"

    def test_is_installed_false_by_default(self, tmp_path):
        from freeaiagent.sdx.catalog import is_installed
        with patch("freeaiagent.sdx.catalog.SDX_DIR", tmp_path):
            assert not is_installed("sdx-standard")

    def test_is_installed_true_when_text_gguf_exists(self, tmp_path):
        from freeaiagent.sdx.catalog import is_installed
        (tmp_path / "sdx-standard").mkdir()
        (tmp_path / "sdx-standard" / "text.gguf").touch()
        with patch("freeaiagent.sdx.catalog.SDX_DIR", tmp_path):
            assert is_installed("sdx-standard")

    def test_model_paths_no_mmproj(self, tmp_path):
        from freeaiagent.sdx.catalog import model_paths
        base = tmp_path / "sdx-nano"
        base.mkdir()
        (base / "text.gguf").touch()
        (base / "vision.gguf").touch()
        with patch("freeaiagent.sdx.catalog.SDX_DIR", tmp_path):
            paths = model_paths("sdx-nano")
        assert paths["text"].endswith("text.gguf")
        assert paths["vision"].endswith("vision.gguf")
        assert paths["mmproj"] is None

    def test_model_paths_with_mmproj(self, tmp_path):
        from freeaiagent.sdx.catalog import model_paths
        base = tmp_path / "sdx-standard"
        base.mkdir()
        (base / "text.gguf").touch()
        (base / "vision.gguf").touch()
        (base / "mmproj.gguf").touch()
        with patch("freeaiagent.sdx.catalog.SDX_DIR", tmp_path):
            paths = model_paths("sdx-standard")
        assert paths["mmproj"] is not None
        assert paths["mmproj"].endswith("mmproj.gguf")
