import pytest

from freeaiagent import catalog


@pytest.fixture
def temp_model_dirs(tmp_path, monkeypatch):
    """Point both the backend and the models endpoint at empty temp dirs."""
    llama = tmp_path / "llamafile"
    models = tmp_path / "models"
    engine = tmp_path / "engine"
    llama.mkdir(); models.mkdir(); engine.mkdir()
    for mod in ("freeaiagent.backends.llamafile", "freeaiagent.endpoints.models"):
        monkeypatch.setattr(f"{mod}.LLAMAFILE_DIR", llama, raising=False)
        monkeypatch.setattr(f"{mod}.MODELS_DIR", models, raising=False)
    monkeypatch.setattr("freeaiagent.backends.llamafile.ENGINE_DIR", engine, raising=False)
    return {"llamafile": llama, "models": models, "engine": engine}


# ── /models/catalog ──────────────────────────────────────────────────────────

@pytest.mark.integration
def test_models_catalog_lists_all_uninstalled(client, temp_model_dirs):
    data = client.get("/models/catalog").json()
    names = [m["name"] for m in data["models"]]
    assert names == catalog.names()
    assert all(m["installed"] is False for m in data["models"])
    one = data["models"][0]
    assert {"display", "kind", "size_gb", "min_ram_gb", "tier", "description"} <= set(one)


@pytest.mark.integration
def test_models_catalog_marks_installed(client, temp_model_dirs):
    from freeaiagent.backends.llamafile import LlamafileBackend
    backend = LlamafileBackend(model="llama-3.2-1b")
    bin_path = backend._bin()
    bin_path.parent.mkdir(parents=True, exist_ok=True)
    bin_path.touch()

    data = client.get("/models/catalog").json()
    flags = {m["name"]: m["installed"] for m in data["models"]}
    assert flags["llama-3.2-1b"] is True
    assert flags["gemma-2-2b"] is False


# ── /models/installed ────────────────────────────────────────────────────────

@pytest.mark.integration
def test_models_installed_lists_files(client, temp_model_dirs):
    (temp_model_dirs["llamafile"] / "Llama-3.2-1B.llamafile").write_bytes(b"x" * 2048)
    (temp_model_dirs["models"] / "Qwen2.5-7B.gguf").write_bytes(b"y" * 1024)
    # .part files (in-flight downloads) must be ignored
    (temp_model_dirs["models"] / "half.gguf.part").write_bytes(b"z" * 10)

    data = client.get("/models/installed").json()
    names = {m["name"] for m in data["models"]}
    assert names == {"Llama-3.2-1B.llamafile", "Qwen2.5-7B.gguf"}
    kinds = {m["name"]: m["kind"] for m in data["models"]}
    assert kinds["Qwen2.5-7B.gguf"] == "gguf"
    assert all("size_mb" in m for m in data["models"])


@pytest.mark.integration
def test_models_installed_empty(client, temp_model_dirs):
    data = client.get("/models/installed").json()
    assert data == {"models": []}


# ── DELETE /models/installed/{name} ──────────────────────────────────────────

@pytest.mark.integration
def test_delete_installed_by_filename(client, temp_model_dirs):
    f = temp_model_dirs["models"] / "Qwen2.5-7B.gguf"
    f.write_bytes(b"y" * (1024 * 1024))

    r = client.delete("/models/installed/Qwen2.5-7B.gguf")
    assert r.status_code == 200
    body = r.json()
    assert body["deleted"] == "Qwen2.5-7B.gguf"
    assert body["freed_mb"] == 1.0
    assert not f.exists()
    assert client.get("/models/installed").json() == {"models": []}


@pytest.mark.integration
def test_delete_installed_missing_404(client, temp_model_dirs):
    r = client.delete("/models/installed/nope.gguf")
    assert r.status_code == 404


@pytest.mark.integration
def test_delete_installed_unsafe_name_rejected(client, temp_model_dirs):
    # A name containing ".." (but no slash, so it reaches the handler as one
    # path param) is rejected with 400 before any filesystem access.
    r = client.delete("/models/installed/..foo")
    assert r.status_code == 400


# ── /config ──────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_config_get_returns_config(client):
    data = client.get("/config").json()
    assert "default_backend" in data
    assert "backends" in data


@pytest.mark.integration
def test_config_set_persists(client):
    r = client.post("/config/set", json={"key": "default_backend", "value": "groq"})
    assert r.status_code == 200
    assert r.json() == {"key": "default_backend", "value": "groq"}
    assert client.get("/config").json()["default_backend"] == "groq"


@pytest.mark.integration
def test_config_set_nested_key(client):
    client.post("/config/set", json={"key": "backends.groq.api_key", "value": "gsk_test"})
    assert client.get("/config").json()["backends"]["groq"]["api_key"] == "gsk_test"
