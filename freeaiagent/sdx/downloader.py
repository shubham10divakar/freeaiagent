"""SDX bundle downloader — fetch text + vision + optional mmproj files."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Callable, Optional

from .catalog import SDX_CATALOG, SDX_DIR

_MB = 1024 * 1024
_TIMEOUT = 30
_CHUNK = 65536


class SDXDownloadHelper:
    """Download an SDX bundle (text.gguf + vision.gguf + optional mmproj.gguf).

    Matches the ``download(force, on_chunk)`` interface of ``LlamafileBackend``
    so it can be dropped straight into a ``PullTarget`` and used by models.py.
    """

    def __init__(self, model_id: str, entry: dict) -> None:
        self._model_id = model_id
        self._entry = entry
        self._dest = SDX_DIR / model_id

    def download(
        self, force: bool = False, on_chunk: Optional[Callable] = None
    ) -> Path:
        """Download all bundle files to ``~/.freeaiagent/sdx/<model_id>/``.

        Returns the destination directory. Files already on disk are skipped
        unless ``force`` is True. Partial downloads are resumed via HTTP Range.
        """
        self._dest.mkdir(parents=True, exist_ok=True)
        files_cfg = self._entry["files"]

        self._fetch(
            files_cfg["text"]["url"],
            self._dest / "text.gguf",
            phase="text_model",
            force=force,
            on_chunk=on_chunk,
        )
        self._fetch(
            files_cfg["vision"]["url"],
            self._dest / "vision.gguf",
            phase="vision_model",
            force=force,
            on_chunk=on_chunk,
        )
        mmproj_cfg = (files_cfg["vision"].get("mmproj") or {})
        if mmproj_cfg.get("url"):
            self._fetch(
                mmproj_cfg["url"],
                self._dest / "mmproj.gguf",
                phase="mmproj",
                force=force,
                on_chunk=on_chunk,
            )

        return self._dest

    def _fetch(
        self,
        url: str,
        dest: Path,
        phase: str,
        force: bool,
        on_chunk: Optional[Callable],
    ) -> None:
        import requests

        if dest.exists() and not force:
            size = dest.stat().st_size
            if on_chunk:
                on_chunk(size, size, phase)
            return

        tmp = dest.with_suffix(".part")
        existing = tmp.stat().st_size if tmp.exists() else 0

        headers = {"Range": f"bytes={existing}-"} if existing else {}
        r = requests.get(url, stream=True, headers=headers, timeout=_TIMEOUT)
        r.raise_for_status()

        # Determine total size (handles both fresh and resumed downloads)
        cr = r.headers.get("Content-Range", "")
        if cr and "/" in cr:
            total = int(cr.split("/")[-1])
        else:
            cl = r.headers.get("Content-Length", "0")
            total = int(cl) + existing if cl else 0

        done = existing
        mode = "ab" if existing else "wb"
        with open(tmp, mode) as fh:
            for chunk in r.iter_content(chunk_size=_CHUNK):
                if chunk:
                    fh.write(chunk)
                    done += len(chunk)
                    if on_chunk:
                        on_chunk(done, total, phase)

        tmp.rename(dest)


def _sdx_phase_labels(entry: dict) -> dict:
    display = entry["display"]
    return {
        "text_model": f"{display} · text model",
        "vision_model": f"{display} · vision model",
        "mmproj": f"{display} · vision projector",
    }
