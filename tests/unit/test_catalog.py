import pytest

from freeaiagent import catalog
from freeaiagent.backends.llamafile import LlamafileBackend


@pytest.mark.unit
def test_default_model_is_in_catalog():
    assert catalog.DEFAULT_MODEL in catalog.CATALOG
    assert catalog.get(catalog.DEFAULT_MODEL) is not None


@pytest.mark.unit
def test_get_unknown_returns_none():
    assert catalog.get("does-not-exist") is None
    assert catalog.url_for("does-not-exist") is None


@pytest.mark.unit
def test_every_entry_has_required_fields():
    required = {"display", "url", "size_gb", "min_ram_gb", "tier", "description"}
    for name, entry in catalog.all_entries():
        assert required <= entry.keys(), f"{name} missing fields"
        assert entry["url"].startswith("https://")
        assert entry["url"].endswith(".llamafile")
        assert entry["size_gb"] < 4.0, f"{name} fused file must stay under the 4 GB Windows cap"


@pytest.mark.unit
def test_url_for_matches_entry():
    name = catalog.names()[0]
    assert catalog.url_for(name) == catalog.get(name)["url"]


@pytest.mark.unit
def test_backend_resolves_catalog_model_url():
    b = LlamafileBackend(model="gemma-2-2b")
    assert b.download_url == catalog.url_for("gemma-2-2b")


@pytest.mark.unit
def test_backend_explicit_url_overrides_catalog():
    url = "https://example.com/custom.llamafile"
    b = LlamafileBackend(model="gemma-2-2b", download_url=url)
    assert b.download_url == url


@pytest.mark.unit
def test_backend_unknown_model_falls_back_to_default_url():
    b = LlamafileBackend(model="nonexistent-model")
    assert b.download_url == catalog.url_for(catalog.DEFAULT_MODEL)
