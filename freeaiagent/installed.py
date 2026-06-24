"""Disk-side helpers for locally installed model files.

Listing, resolving, and deleting downloaded models is needed by three callers
that must agree exactly: the ``freeaiagent rm`` CLI command (runs offline, no
server), the ``/models/installed`` endpoints (for the SDK), and tests. Keeping
the filesystem logic here makes them behave identically — the CLI deletes the
same file the endpoint would.
"""
from pathlib import Path
from typing import List, Optional

from . import catalog
from .backends.llamafile import LlamafileBackend

_MB = 1024 * 1024


def _dirs() -> tuple:
    # (directory, kind) pairs; resolved lazily so test monkeypatching of the
    # module-level DIR constants is honoured on every call.
    from .backends import llamafile
    return (
        (llamafile.LLAMAFILE_DIR, "llamafile"),
        (llamafile.MODELS_DIR, "gguf"),
    )


def installed_files() -> List[dict]:
    """List model files actually present on disk (skips in-flight ``.part``)."""
    out: List[dict] = []
    for directory, kind in _dirs():
        if not directory.exists():
            continue
        for f in sorted(directory.iterdir()):
            if not f.is_file() or f.name.endswith(".part"):
                continue
            out.append({
                "name": f.name,
                "path": str(f),
                "size_mb": round(f.stat().st_size / _MB, 1),
                "kind": kind,
            })
    return out


def _is_safe_name(name: str) -> bool:
    """Reject path traversal — only bare filenames / catalog names are allowed."""
    return bool(name) and "/" not in name and "\\" not in name and ".." not in name


def resolve_path(name: str) -> Optional[Path]:
    """Resolve a catalog name or on-disk filename to an installed model file.

    Catalog names (e.g. ``llama-3.2-3b``) resolve via the backend's expected
    artifact path; anything else is treated as a literal filename in one of the
    model directories. Returns ``None`` if nothing matches or the name is unsafe.
    """
    if not _is_safe_name(name):
        return None
    if catalog.get(name):
        p = LlamafileBackend(model=name)._bin()
        if p.exists():
            return p
    for directory, _kind in _dirs():
        cand = directory / name
        if cand.exists() and cand.is_file():
            return cand
    return None


def delete(name: str) -> dict:
    """Delete an installed model file, returning what was freed.

    Raises ``ValueError`` for an unsafe name and ``FileNotFoundError`` when no
    matching model is installed. The shared engine binary is never touched.
    """
    if not _is_safe_name(name):
        raise ValueError(f"Invalid model name: {name!r}")
    path = resolve_path(name)
    if path is None:
        raise FileNotFoundError(f"No installed model named '{name}'.")
    freed_mb = round(path.stat().st_size / _MB, 1)
    path.unlink()
    return {"deleted": path.name, "path": str(path), "freed_mb": freed_mb}
