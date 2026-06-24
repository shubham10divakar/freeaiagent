import pytest
from typer.testing import CliRunner

from freeaiagent.cli import app

runner = CliRunner()


@pytest.fixture
def temp_model_dirs(tmp_path, monkeypatch):
    llama = tmp_path / "llamafile"
    models = tmp_path / "models"
    llama.mkdir(); models.mkdir()
    monkeypatch.setattr("freeaiagent.backends.llamafile.LLAMAFILE_DIR", llama)
    monkeypatch.setattr("freeaiagent.backends.llamafile.MODELS_DIR", models)
    return {"llamafile": llama, "models": models}


@pytest.mark.unit
def test_rm_deletes_with_yes(temp_model_dirs):
    f = temp_model_dirs["models"] / "b.gguf"
    f.write_bytes(b"y" * (1024 * 1024))
    result = runner.invoke(app, ["rm", "b.gguf", "--yes"])
    assert result.exit_code == 0, result.output
    assert "freed" in result.output
    assert not f.exists()


@pytest.mark.unit
def test_rm_confirmation_abort_keeps_file(temp_model_dirs):
    f = temp_model_dirs["models"] / "b.gguf"
    f.write_bytes(b"y" * 1024)
    result = runner.invoke(app, ["rm", "b.gguf"], input="n\n")
    assert result.exit_code != 0
    assert f.exists()


@pytest.mark.unit
def test_rm_missing_lists_installed(temp_model_dirs):
    (temp_model_dirs["models"] / "present.gguf").write_bytes(b"y" * 1024)
    result = runner.invoke(app, ["rm", "absent.gguf"])
    assert result.exit_code == 1
    assert "No installed model named 'absent.gguf'" in result.output
    assert "present.gguf" in result.output
