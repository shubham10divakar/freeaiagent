"""Server lock file for zero-config discovery.

The server writes ``~/.freeaiagent/server.json`` on start and removes it on a
clean exit. Apps using the SDK read it to find the running port instead of
hardcoding 7731, so they survive port changes with no config on their side.
A stale lock (PID no longer alive) is treated as "not running".
"""
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

BASE_DIR = Path.home() / ".freeaiagent"
LOCK_FILE = BASE_DIR / "server.json"

DEFAULT_PORT = 7731


def write_lock(port: int) -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    LOCK_FILE.write_text(json.dumps({
        "port": port,
        "pid": os.getpid(),
        "started_at": datetime.now(timezone.utc).isoformat(),
    }))


def remove_lock() -> None:
    try:
        LOCK_FILE.unlink()
    except FileNotFoundError:
        pass


def read_lock() -> Optional[dict]:
    try:
        return json.loads(LOCK_FILE.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def pid_alive(pid: int) -> bool:
    """Best-effort liveness check that never kills the target process."""
    if not pid or pid <= 0:
        return False
    if platform.system() == "Windows":
        # os.kill on Windows TERMINATES the process for any signal, so use the
        # Win32 API directly to merely probe existence.
        import ctypes
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        STILL_ACTIVE = 259
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
        if not handle:
            return False
        try:
            code = ctypes.c_ulong()
            kernel32.GetExitCodeProcess(handle, ctypes.byref(code))
            return code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True  # exists, owned by another user
    except OSError:
        return False
    return True


def discover_port(default: int = DEFAULT_PORT) -> int:
    """Return the running server's port from the lock file, else ``default``.

    Only trusts the lock when its PID is still alive; a stale lock falls back to
    the default so the caller can decide to (re)start.
    """
    lock = read_lock()
    if lock and pid_alive(lock.get("pid", -1)):
        return lock.get("port", default)
    return default
