"""Install freeaiagent to start automatically, so apps can assume it's up.

Per-user, no-admin install on every platform:
  - Linux:   systemd *user* unit (~/.config/systemd/user/freeaiagent.service)
  - macOS:   launchd LaunchAgent (~/Library/LaunchAgents/com.freeaiagent.plist)
  - Windows: HKCU\\...\\Run registry value (starts at login)

After install, use Client(auto_start=False) and treat the agent like a local
database. The content generators are pure so they can be unit-tested without
touching the system.
"""
import platform
import subprocess
import sys
from pathlib import Path
from typing import List

LABEL = "freeaiagent"
LAUNCHD_LABEL = "com.freeaiagent"

SYSTEMD_UNIT_PATH = Path.home() / ".config" / "systemd" / "user" / "freeaiagent.service"
LAUNCHD_PLIST_PATH = Path.home() / "Library" / "LaunchAgents" / "com.freeaiagent.plist"
WIN_RUN_KEY = r"Software\Microsoft\Windows\CurrentVersion\Run"


def start_args() -> List[str]:
    """The argv that launches the server with the current interpreter."""
    return [sys.executable, "-m", "freeaiagent", "start"]


# ── content generators (pure) ────────────────────────────────────────────────

def systemd_unit_content() -> str:
    exec_start = " ".join(start_args())
    return (
        "[Unit]\n"
        "Description=freeaiagent local AI agent service\n"
        "After=network.target\n\n"
        "[Service]\n"
        f"ExecStart={exec_start}\n"
        "Restart=on-failure\n\n"
        "[Install]\n"
        "WantedBy=default.target\n"
    )


def launchd_plist_content() -> str:
    args = "".join(f"        <string>{a}</string>\n" for a in start_args())
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" '
        '"http://www.apple.com/DTDs/PropertyList-1.0.dtd">\n'
        '<plist version="1.0">\n'
        "<dict>\n"
        f"    <key>Label</key>\n    <string>{LAUNCHD_LABEL}</string>\n"
        "    <key>ProgramArguments</key>\n    <array>\n"
        f"{args}"
        "    </array>\n"
        "    <key>RunAtLoad</key>\n    <true/>\n"
        "    <key>KeepAlive</key>\n    <true/>\n"
        "</dict>\n"
        "</plist>\n"
    )


def windows_run_command() -> str:
    exe, *rest = start_args()
    return " ".join([f'"{exe}"', *rest])  # quote the interpreter path (may have spaces)


# ── Linux ────────────────────────────────────────────────────────────────────

def _install_linux() -> None:
    SYSTEMD_UNIT_PATH.parent.mkdir(parents=True, exist_ok=True)
    SYSTEMD_UNIT_PATH.write_text(systemd_unit_content())
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
    subprocess.run(["systemctl", "--user", "enable", "--now", LABEL], check=True)


def _uninstall_linux() -> None:
    subprocess.run(["systemctl", "--user", "disable", "--now", LABEL], check=False)
    SYSTEMD_UNIT_PATH.unlink(missing_ok=True)
    subprocess.run(["systemctl", "--user", "daemon-reload"], check=False)


def _status_linux() -> str:
    r = subprocess.run(
        ["systemctl", "--user", "is-active", LABEL],
        capture_output=True, text=True,
    )
    return r.stdout.strip() or "unknown"


# ── macOS ────────────────────────────────────────────────────────────────────

def _install_macos() -> None:
    LAUNCHD_PLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    LAUNCHD_PLIST_PATH.write_text(launchd_plist_content())
    subprocess.run(["launchctl", "unload", str(LAUNCHD_PLIST_PATH)], check=False)
    subprocess.run(["launchctl", "load", "-w", str(LAUNCHD_PLIST_PATH)], check=True)


def _uninstall_macos() -> None:
    subprocess.run(["launchctl", "unload", str(LAUNCHD_PLIST_PATH)], check=False)
    LAUNCHD_PLIST_PATH.unlink(missing_ok=True)


def _status_macos() -> str:
    r = subprocess.run(["launchctl", "list"], capture_output=True, text=True)
    return "running" if LAUNCHD_LABEL in r.stdout else "not installed"


# ── Windows ──────────────────────────────────────────────────────────────────

def _install_windows() -> None:
    import winreg
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, WIN_RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
        winreg.SetValueEx(key, LABEL, 0, winreg.REG_SZ, windows_run_command())


def _uninstall_windows() -> None:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, WIN_RUN_KEY, 0, winreg.KEY_SET_VALUE) as key:
            winreg.DeleteValue(key, LABEL)
    except FileNotFoundError:
        pass


def _status_windows() -> str:
    import winreg
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, WIN_RUN_KEY) as key:
            winreg.QueryValueEx(key, LABEL)
        return "installed (starts at login)"
    except FileNotFoundError:
        return "not installed"


# ── dispatch ─────────────────────────────────────────────────────────────────

_DISPATCH = {
    "Linux":   (_install_linux, _uninstall_linux, _status_linux),
    "Darwin":  (_install_macos, _uninstall_macos, _status_macos),
    "Windows": (_install_windows, _uninstall_windows, _status_windows),
}


def _for_platform():
    funcs = _DISPATCH.get(platform.system())
    if funcs is None:
        raise RuntimeError(f"Unsupported platform: {platform.system()}")
    return funcs


def install() -> None:
    _for_platform()[0]()


def uninstall() -> None:
    _for_platform()[1]()


def status() -> str:
    return _for_platform()[2]()
