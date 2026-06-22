"""Live HuggingFace discovery for GGUF models.

Thin wrappers over the public HuggingFace REST API (no key needed for public
repos). Used by `freeaiagent search` and `freeaiagent pull hf:<repo>/<file>`.
"""
from typing import List, Dict

import httpx2 as httpx

HF_API = "https://huggingface.co/api"
HF_HOST = "https://huggingface.co"


def search_models(query: str, limit: int = 20) -> List[Dict]:
    """Search GGUF repos, most-downloaded first."""
    r = httpx.get(
        f"{HF_API}/models",
        params={
            "search": query,
            "filter": "gguf",
            "sort": "downloads",
            "direction": -1,
            "limit": limit,
        },
        timeout=15.0,
    )
    r.raise_for_status()
    return [
        {
            "id": m["id"],
            "downloads": m.get("downloads", 0),
            "likes": m.get("likes", 0),
        }
        for m in r.json()
    ]


def list_gguf_files(repo: str) -> List[Dict]:
    """List the .gguf files in a repo with their sizes (bytes)."""
    r = httpx.get(f"{HF_API}/models/{repo}/tree/main", params={"recursive": "true"}, timeout=15.0)
    r.raise_for_status()
    files = [
        {"path": item["path"], "size": item.get("size", 0)}
        for item in r.json()
        if item.get("type") == "file" and item.get("path", "").endswith(".gguf")
    ]
    return sorted(files, key=lambda f: f["path"])


def resolve_url(repo: str, filename: str) -> str:
    """Direct download URL for a file in a repo (main revision)."""
    return f"{HF_HOST}/{repo}/resolve/main/{filename}"


def parse_hf_ref(ref: str) -> tuple[str, str]:
    """Parse `hf:<owner>/<name>/<path/to/file.gguf>` into (repo, filename).

    Raises ValueError if it doesn't contain at least owner/name/file.
    """
    spec = ref[3:] if ref.startswith("hf:") else ref
    parts = [p for p in spec.split("/") if p]
    if len(parts) < 3:
        raise ValueError("expected hf:<owner>/<name>/<file.gguf>")
    return "/".join(parts[:2]), "/".join(parts[2:])
