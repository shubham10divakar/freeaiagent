import os

import pytest

from freeaiagent import server as srv


@pytest.fixture
def lock_in_tmp(tmp_path, monkeypatch):
    monkeypatch.setattr(srv, "BASE_DIR", tmp_path)
    monkeypatch.setattr(srv, "LOCK_FILE", tmp_path / "server.json")
    return tmp_path / "server.json"


@pytest.mark.unit
def test_write_then_read_lock(lock_in_tmp):
    srv.write_lock(8123)
    lock = srv.read_lock()
    assert lock["port"] == 8123
    assert lock["pid"] == os.getpid()
    assert "started_at" in lock


@pytest.mark.unit
def test_remove_lock_is_idempotent(lock_in_tmp):
    srv.write_lock(7731)
    srv.remove_lock()
    assert srv.read_lock() is None
    srv.remove_lock()  # no error on a missing file


@pytest.mark.unit
def test_read_lock_none_when_absent(lock_in_tmp):
    assert srv.read_lock() is None


@pytest.mark.unit
def test_read_lock_none_when_corrupt(lock_in_tmp):
    lock_in_tmp.write_text("{not json")
    assert srv.read_lock() is None


@pytest.mark.unit
def test_pid_alive_for_self():
    assert srv.pid_alive(os.getpid()) is True


@pytest.mark.unit
def test_pid_alive_false_for_bogus():
    assert srv.pid_alive(-1) is False
    assert srv.pid_alive(0) is False


@pytest.mark.unit
def test_discover_port_uses_lock_when_alive(lock_in_tmp, monkeypatch):
    srv.write_lock(9090)
    monkeypatch.setattr(srv, "pid_alive", lambda pid: True)
    assert srv.discover_port() == 9090


@pytest.mark.unit
def test_discover_port_falls_back_on_dead_pid(lock_in_tmp, monkeypatch):
    srv.write_lock(9090)
    monkeypatch.setattr(srv, "pid_alive", lambda pid: False)
    assert srv.discover_port(default=7731) == 7731


@pytest.mark.unit
def test_discover_port_default_when_no_lock(lock_in_tmp):
    assert srv.discover_port(default=7731) == 7731
