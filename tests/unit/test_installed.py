import pytest

from freeaiagent import installed


@pytest.fixture
def temp_model_dirs(tmp_path, monkeypatch):
    """Point the llamafile backend dirs at empty temp dirs."""
    llama = tmp_path / "llamafile"
    models = tmp_path / "models"
    llama.mkdir(); models.mkdir()
    monkeypatch.setattr("freeaiagent.backends.llamafile.LLAMAFILE_DIR", llama)
    monkeypatch.setattr("freeaiagent.backends.llamafile.MODELS_DIR", models)
    return {"llamafile": llama, "models": models}


# ── installed_files ──────────────────────────────────────────────────────────

@pytest.mark.unit
def test_installed_files_lists_and_skips_part(temp_model_dirs):
    (temp_model_dirs["llamafile"] / "a.llamafile").write_bytes(b"x" * 2048)
    (temp_model_dirs["models"] / "b.gguf").write_bytes(b"y" * 1024)
    (temp_model_dirs["models"] / "c.gguf.part").write_bytes(b"z" * 10)

    files = installed.installed_files()
    names = {f["name"] for f in files}
    assert names == {"a.llamafile", "b.gguf"}
    kinds = {f["name"]: f["kind"] for f in files}
    assert kinds == {"a.llamafile": "llamafile", "b.gguf": "gguf"}


@pytest.mark.unit
def test_installed_files_empty(temp_model_dirs):
    assert installed.installed_files() == []


# ── resolve_path ─────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_resolve_by_catalog_name(temp_model_dirs):
    from freeaiagent.backends.llamafile import LlamafileBackend
    p = LlamafileBackend(model="llama-3.2-1b")._bin()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"x")
    assert installed.resolve_path("llama-3.2-1b") == p


@pytest.mark.unit
def test_resolve_by_filename(temp_model_dirs):
    f = temp_model_dirs["models"] / "Qwen2.5-7B.gguf"
    f.write_bytes(b"y")
    assert installed.resolve_path("Qwen2.5-7B.gguf") == f


@pytest.mark.unit
def test_resolve_missing_returns_none(temp_model_dirs):
    assert installed.resolve_path("nope.gguf") is None


@pytest.mark.unit
@pytest.mark.parametrize("bad", ["../etc/passwd", "a/b", "a\\b", "..", ""])
def test_resolve_rejects_unsafe_names(temp_model_dirs, bad):
    assert installed.resolve_path(bad) is None


# ── delete ───────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_delete_removes_file_and_reports_size(temp_model_dirs):
    f = temp_model_dirs["models"] / "b.gguf"
    f.write_bytes(b"y" * (2 * 1024 * 1024))
    result = installed.delete("b.gguf")
    assert not f.exists()
    assert result["deleted"] == "b.gguf"
    assert result["freed_mb"] == 2.0


@pytest.mark.unit
def test_delete_missing_raises(temp_model_dirs):
    with pytest.raises(FileNotFoundError):
        installed.delete("nope.gguf")


@pytest.mark.unit
def test_delete_unsafe_raises(temp_model_dirs):
    with pytest.raises(ValueError):
        installed.delete("../secret")
