import pytest
from unittest.mock import patch
from typer.testing import CliRunner

from freeaiagent.cli import app

runner = CliRunner()


@pytest.mark.unit
def test_search_term_lists_repos():
    repos = [{"id": "bartowski/Qwen2.5-7B-Instruct-GGUF", "downloads": 1200000, "likes": 42}]
    with patch("freeaiagent.hf.search_models", return_value=repos) as m:
        result = runner.invoke(app, ["search", "qwen2.5"])
    assert result.exit_code == 0
    assert "bartowski/Qwen2.5-7B-Instruct-GGUF" in result.output
    m.assert_called_once()


@pytest.mark.unit
def test_search_repo_lists_files():
    files = [{"path": "Qwen2.5-7B-Instruct-Q4_K_M.gguf", "size": 4700000000}]
    with patch("freeaiagent.hf.list_gguf_files", return_value=files) as m:
        result = runner.invoke(app, ["search", "bartowski/Qwen2.5-7B-Instruct-GGUF"])
    assert result.exit_code == 0
    assert "Qwen2.5-7B-Instruct-Q4_K_M.gguf" in result.output
    assert "pull hf:bartowski/Qwen2.5-7B-Instruct-GGUF/<filename>" in result.output
    m.assert_called_once_with("bartowski/Qwen2.5-7B-Instruct-GGUF")


@pytest.mark.unit
def test_pull_hf_ref_builds_correct_url(isolated_config, tmp_path):
    captured = {}

    def fake_download(self, force=False):
        captured["url"] = self.download_url
        return tmp_path / "x"

    # avoid touching the real filesystem path check
    with patch("freeaiagent.backends.llamafile.LlamafileBackend._bin", return_value=tmp_path / "missing.gguf"), \
         patch("freeaiagent.backends.llamafile.LlamafileBackend.download", new=fake_download):
        result = runner.invoke(
            app,
            ["pull", "hf:bartowski/Qwen2.5-7B-Instruct-GGUF/Qwen2.5-7B-Instruct-Q4_K_M.gguf"],
        )
    assert result.exit_code == 0, result.output
    assert captured["url"] == (
        "https://huggingface.co/bartowski/Qwen2.5-7B-Instruct-GGUF/resolve/main/"
        "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
    )


@pytest.mark.unit
def test_pull_invalid_hf_ref_errors(isolated_config):
    result = runner.invoke(app, ["pull", "hf:owner/name"])
    assert result.exit_code == 1
    assert "Invalid reference" in result.output
