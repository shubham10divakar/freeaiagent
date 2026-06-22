import json
from pathlib import Path
from typing import Any

from . import catalog

CONFIG_DIR = Path.home() / ".freeaiagent"
CONFIG_FILE = CONFIG_DIR / "config.json"

DEFAULTS: dict = {
    "default_backend": "llamafile",
    "default_model": "llama-3.2-3b",  # catalog name; see `freeaiagent models --available`
    "port": 7731,
    "max_messages": 0,  # 0 = unlimited; set to e.g. 20 to keep last 20 messages
    "backends": {
        # Local backend: run `freeaiagent pull` once (~2.3 GB), then it starts automatically.
        # Set auto_download=true to fetch on first request instead of via `pull`.
        "llamafile": {"type": "llamafile", "port": 8080, "auto_download": False},
        # Ollama: install from https://ollama.com, then: ollama pull llama3.2:3b
        "ollama":    {"base_url": "http://localhost:11434"},
        # Groq: free API key at https://console.groq.com
        "groq":      {"api_key": ""},
        # Any OpenAI-compatible server (LM Studio, LocalAI, Gemini, OpenRouter, etc.)
        # "lmstudio":  {"type": "openai_compat", "base_url": "http://localhost:1234"},
    },
    "fallback_order": ["llamafile", "ollama", "groq"],
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
    # migrate legacy default_model strings (e.g. "Llama-3.2-3B-Instruct") to catalog names
    merged["default_model"] = catalog.normalize(merged.get("default_model"))
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
