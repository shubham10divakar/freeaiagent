"""Curated catalog of local models (Phase 1: self-contained fused llamafiles).

Each entry maps a short, stable catalog name to a downloadable llamafile plus
metadata used for `freeaiagent models --available` and disk/RAM warnings.

Only models whose *fused* llamafile is under ~4 GB are listed here, because a
single-file llamafile larger than 4 GB will not run on Windows (PE format
limit). Larger models (7B+) are deferred to the engine/weights split — see
MULTI_MODEL_DESIGN.md. Until then, pull a bigger model by URL:
    freeaiagent pull <url-to-a-llamafile>
"""
from typing import Optional

_HF = "https://huggingface.co/Mozilla"      # fused llamafiles
_BK = "https://huggingface.co/bartowski"    # external GGUF weights

# name -> entry. Two kinds:
#   "llamafile" — self-contained fused executable; runs directly. Kept < 4 GB so
#                 it works on Windows (PE format cap on single-file executables).
#   "gguf"      — external weights run via the downloaded llamafile *engine*
#                 binary; no 4 GB cap, so 7B+ models work everywhere.
# All URLs verified to exist.
CATALOG: dict[str, dict] = {
    "llama-3.2-1b": {
        "display": "Llama 3.2 1B Instruct",
        "kind": "llamafile",
        "url": f"{_HF}/Llama-3.2-1B-Instruct-llamafile/resolve/main/Llama-3.2-1B-Instruct-Q6_K.llamafile",
        "size_gb": 1.3,
        "min_ram_gb": 2,
        "tier": "low",
        "description": "Fastest. Good for classify / extract / tag.",
    },
    "gemma-2-2b": {
        "display": "Gemma 2 2B Instruct",
        "kind": "llamafile",
        "url": f"{_HF}/gemma-2-2b-it-llamafile/resolve/main/gemma-2-2b-it.Q4_K_M.llamafile",
        "size_gb": 2.0,
        "min_ram_gb": 4,
        "tier": "mid",
        "description": "Concise; strong at short summaries.",
    },
    "llama-3.2-3b": {
        "display": "Llama 3.2 3B Instruct",
        "kind": "llamafile",
        "url": f"{_HF}/Llama-3.2-3B-Instruct-llamafile/resolve/main/Llama-3.2-3B-Instruct-Q4_K_M.llamafile",
        "size_gb": 2.3,
        "min_ram_gb": 4,
        "tier": "mid",
        "description": "Balanced default. Handles light reasoning / Q&A.",
    },
    "phi-3-mini": {
        "display": "Phi-3 Mini 4k Instruct",
        "kind": "llamafile",
        "url": f"{_HF}/Phi-3-mini-4k-instruct-llamafile/resolve/main/Phi-3-mini-4k-instruct.Q4_K_M.llamafile",
        "size_gb": 2.4,
        "min_ram_gb": 4,
        "tier": "mid",
        "description": "Strong reasoning per byte.",
    },
    # GGUF (engine-run). One-time ~305 MB engine download applies to these.
    "qwen2.5-7b": {
        "display": "Qwen2.5 7B Instruct",
        "kind": "gguf",
        "url": f"{_BK}/Qwen2.5-7B-Instruct-GGUF/resolve/main/Qwen2.5-7B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.7,
        "min_ram_gb": 8,
        "tier": "high",
        "description": "Strong reasoning, Q&A, summaries.",
    },
    "llama-3.1-8b": {
        "display": "Llama 3.1 8B Instruct",
        "kind": "gguf",
        "url": f"{_BK}/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
        "size_gb": 4.9,
        "min_ram_gb": 8,
        "tier": "high",
        "description": "Broad capability, 128k context.",
    },
    "qwen2.5-14b": {
        "display": "Qwen2.5 14B Instruct",
        "kind": "gguf",
        "url": f"{_BK}/Qwen2.5-14B-Instruct-GGUF/resolve/main/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "size_gb": 9.0,
        "min_ram_gb": 16,
        "tier": "max",
        "description": "Strongest local option.",
    },
}

DEFAULT_MODEL = "llama-3.2-3b"

# Map pre-catalog default_model strings to catalog names (config migration).
LEGACY_ALIASES = {
    "Llama-3.2-3B-Instruct": "llama-3.2-3b",
    "Llama-3.2-1B-Instruct": "llama-3.2-1b",
}


def normalize(name: Optional[str]) -> Optional[str]:
    """Translate a legacy model string to its catalog name (no-op otherwise)."""
    if name is None:
        return None
    return LEGACY_ALIASES.get(name, name)


def get(name: str) -> Optional[dict]:
    """Return the catalog entry for a name, or None if unknown."""
    return CATALOG.get(name)


def names() -> list[str]:
    return list(CATALOG)


def all_entries() -> list[tuple[str, dict]]:
    """(name, entry) pairs in catalog order (low tier first)."""
    return list(CATALOG.items())


def url_for(name: str) -> Optional[str]:
    entry = CATALOG.get(name)
    return entry["url"] if entry else None


def kind_of(name: str) -> Optional[str]:
    entry = CATALOG.get(name)
    return entry.get("kind", "llamafile") if entry else None
