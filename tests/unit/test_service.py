import sys

import pytest
from typer.testing import CliRunner

from freeaiagent import service
from freeaiagent.cli import app

runner = CliRunner()


# ── content generators ───────────────────────────────────────────────────────

@pytest.mark.unit
def test_start_args_uses_current_interpreter():
    assert service.start_args() == [sys.executable, "-m", "freeaiagent", "start"]


@pytest.mark.unit
def test_systemd_unit_content():
    unit = service.systemd_unit_content()
    assert "[Unit]" in unit and "[Service]" in unit and "[Install]" in unit
    assert "-m freeaiagent start" in unit
    assert "WantedBy=default.target" in unit


@pytest.mark.unit
def test_launchd_plist_content():
    plist = service.launchd_plist_content()
    assert "com.freeaiagent" in plist
    assert "<key>RunAtLoad</key>" in plist
    assert "<string>-m</string>" in plist and "<string>start</string>" in plist


@pytest.mark.unit
def test_windows_run_command_quotes_interpreter():
    cmd = service.windows_run_command()
    assert cmd == f'"{sys.executable}" -m freeaiagent start'


# ── dispatch ─────────────────────────────────────────────────────────────────

@pytest.mark.unit
@pytest.mark.parametrize("system", ["Linux", "Darwin", "Windows"])
def test_install_dispatches_per_platform(system, monkeypatch):
    called = []
    monkeypatch.setattr(service.platform, "system", lambda: system)
    monkeypatch.setitem(
        service._DISPATCH, system,
        (lambda: called.append("install"), lambda: None, lambda: "x"),
    )
    service.install()
    assert called == ["install"]


@pytest.mark.unit
def test_unsupported_platform_raises(monkeypatch):
    monkeypatch.setattr(service.platform, "system", lambda: "Plan9")
    with pytest.raises(RuntimeError, match="Unsupported platform"):
        service.install()


@pytest.mark.unit
def test_status_returns_value(monkeypatch):
    monkeypatch.setattr(service.platform, "system", lambda: "Linux")
    monkeypatch.setitem(
        service._DISPATCH, "Linux",
        (lambda: None, lambda: None, lambda: "running"),
    )
    assert service.status() == "running"


# ── Linux filesystem path (no systemctl) ─────────────────────────────────────

@pytest.mark.unit
def test_install_linux_writes_unit(tmp_path, monkeypatch):
    unit = tmp_path / "systemd" / "user" / "freeaiagent.service"
    monkeypatch.setattr(service, "SYSTEMD_UNIT_PATH", unit)
    monkeypatch.setattr(service.subprocess, "run", lambda *a, **k: None)
    service._install_linux()
    assert unit.exists()
    assert "ExecStart=" in unit.read_text()


# ── CLI wiring ───────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_cli_install_success(monkeypatch):
    monkeypatch.setattr("freeaiagent.service.install", lambda: None)
    result = runner.invoke(app, ["install"])
    assert result.exit_code == 0
    assert "Installed" in result.output


@pytest.mark.unit
def test_cli_install_failure_exits_1(monkeypatch):
    def boom():
        raise RuntimeError("nope")
    monkeypatch.setattr("freeaiagent.service.install", boom)
    result = runner.invoke(app, ["install"])
    assert result.exit_code == 1
    assert "Install failed" in result.output


@pytest.mark.unit
def test_cli_service_status(monkeypatch):
    monkeypatch.setattr("freeaiagent.service.status", lambda: "running")
    result = runner.invoke(app, ["service", "status"])
    assert result.exit_code == 0
    assert "running" in result.output
