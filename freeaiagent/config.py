import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".freeaiagent"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS: dict = {
    "default_backend": "ollama",
    "default_model": "llama3.2:3b",
    "port": 7731,
    "backends": {
        "ollama": {"base_url": "http://localhost:11434"},
        "groq": {"api_key": ""},
    },
    "fallback_order": ["ollama", "groq"],
}


def load() -> dict:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    if not CONFIG_FILE.exists():
        save(DEFAULTS)
        return DEFAULTS.copy()
    with open(CONFIG_FILE) as f:
        on_disk = json.load(f)
    # shallow merge so new default keys appear without losing user values
    merged = {**DEFAULTS, **on_disk}
    merged["backends"] = {**DEFAULTS["backends"], **on_disk.get("backends", {})}
    return merged


def save(config: dict) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def set_value(dotted_key: str, value: Any) -> None:
    config = load()
    parts = dotted_key.split(".")
    target = config
    for part in parts[:-1]:
        target = target.setdefault(part, {})
    target[parts[-1]] = value
    save(config)
