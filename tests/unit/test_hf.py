import pytest
from unittest.mock import patch, MagicMock

from freeaiagent import hf


def _resp(json_data):
    r = MagicMock()
    r.raise_for_status = MagicMock()
    r.json.return_value = json_data
    return r


@pytest.mark.unit
def test_search_models_maps_fields():
    data = [
        {"id": "bartowski/Qwen2.5-7B-Instruct-GGUF", "downloads": 1200000, "likes": 42},
        {"id": "Qwen/Qwen2.5-7B-Instruct-GGUF", "downloads": 800000, "likes": 30},
    ]
    with patch("freeaiagent.hf.httpx.get", return_value=_resp(data)) as g:
        out = hf.search_models("qwen2.5")
    assert out[0]["id"] == "bartowski/Qwen2.5-7B-Instruct-GGUF"
    assert out[0]["downloads"] == 1200000
    # gguf filter + downloads sort requested
    params = g.call_args[1]["params"]
    assert params["filter"] == "gguf"
    assert params["sort"] == "downloads"


@pytest.mark.unit
def test_list_gguf_files_filters_and_sorts():
    tree = [
        {"type": "file", "path": "README.md", "size": 1000},
        {"type": "file", "path": "Qwen2.5-7B-Instruct-Q6_K.gguf", "size": 6300000000},
        {"type": "file", "path": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "size": 4700000000},
        {"type": "directory", "path": "subdir"},
    ]
    with patch("freeaiagent.hf.httpx.get", return_value=_resp(tree)):
        files = hf.list_gguf_files("bartowski/Qwen2.5-7B-Instruct-GGUF")
    names = [f["path"] for f in files]
    assert names == [
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "Qwen2.5-7B-Instruct-Q6_K.gguf",
    ]
    assert all(not n.endswith(".md") for n in names)


@pytest.mark.unit
def test_resolve_url():
    assert hf.resolve_url("owner/name", "model-Q4.gguf") == (
        "https://huggingface.co/owner/name/resolve/main/model-Q4.gguf"
    )


@pytest.mark.unit
def test_parse_hf_ref_splits_repo_and_file():
    repo, fname = hf.parse_hf_ref("hf:bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf")
    assert repo == "bartowski/Qwen2.5-7B-Instruct-GGUF"
    assert fname == "Qwen2.5-7B-Instruct-Q4_K_M.gguf"


@pytest.mark.unit
def test_parse_hf_ref_rejects_short_ref():
    with pytest.raises(ValueError):
        hf.parse_hf_ref("hf:owner/name")
