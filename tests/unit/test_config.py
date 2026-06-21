import json
import pytest
import freeaiagent.config as cfg


@pytest.mark.unit
def test_load_creates_defaults_on_first_run(isolated_config):
    config = cfg.load()
    assert config["default_backend"] == "ollama"
    assert config["port"] == 7731
    assert (isolated_config / "config.json").exists()


@pytest.mark.unit
def test_load_merges_with_defaults(isolated_config):
    # write partial config to disk — missing keys should be filled from DEFAULTS
    (isolated_config / "config.json").write_text(json.dumps({"port": 9999}))
    config = cfg.load()
    assert config["port"] == 9999
    assert config["default_backend"] == "ollama"  # merged from defaults


@pytest.mark.unit
def test_save_and_reload(isolated_config):
    data = {**cfg.DEFAULTS, "port": 1234}
    cfg.save(data)
    loaded = cfg.load()
    assert loaded["port"] == 1234


@pytest.mark.unit
def test_set_value_top_level(isolated_config):
    cfg.set_value("port", 8080)
    assert cfg.load()["port"] == 8080


@pytest.mark.unit
def test_set_value_dotted_key(isolated_config):
    cfg.set_value("backends.groq.api_key", "gsk_test123")
    assert cfg.load()["backends"]["groq"]["api_key"] == "gsk_test123"


@pytest.mark.unit
def test_set_value_creates_nested_keys(isolated_config):
    cfg.set_value("backends.newprovider.url", "http://custom:9000")
    assert cfg.load()["backends"]["newprovider"]["url"] == "http://custom:9000"
