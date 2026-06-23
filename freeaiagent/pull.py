"""Shared model-pull logic.

Resolving a pull target (catalog name, `hf:` ref, or URL) into a
``LlamafileBackend`` is needed by both the ``freeaiagent pull`` CLI command and
the ``/pull/stream`` SSE endpoint — keeping it here makes them behave
identically. ``ProgressEmitter`` shapes the backend's raw per-chunk callback
into the SSE event dicts the endpoint and SDK consume.
"""
import time
from dataclasses import dataclass
from typing import Callable, Optional

from . import catalog
from .backends.llamafile import LlamafileBackend

_MB = 1024 * 1024


@dataclass
class PullTarget:
    backend: LlamafileBackend
    label: str
    size_gb: Optional[float]
    min_ram_gb: Optional[int]
    is_catalog_name: bool


def resolve_target(target: str, port: int = 8080) -> PullTarget:
    """Resolve a model spec to a backend plus display metadata.

    ``target`` is a catalog name (e.g. ``llama-3.2-3b``), an
    ``hf:<repo>/<file.gguf>`` ref, or a direct ``http(s)`` URL. Raises
    ``ValueError`` for a malformed hf ref or an unknown catalog name.
    """
    if target.startswith("hf:"):
        from . import hf
        try:
            repo, fname = hf.parse_hf_ref(target)
        except ValueError as e:
            raise ValueError(f"Invalid reference: {e}")
        backend = LlamafileBackend(port=port, download_url=hf.resolve_url(repo, fname))
        return PullTarget(backend, fname, None, None, is_catalog_name=False)

    if target.startswith(("http://", "https://")):
        backend = LlamafileBackend(port=port, download_url=target)
        return PullTarget(backend, target.rsplit("/", 1)[-1], None, None, is_catalog_name=False)

    entry = catalog.get(target)
    if entry is None:
        raise ValueError(
            f"Unknown model '{target}'.\n"
            f"See available models with: freeaiagent models --available\n"
            f"Or pass a direct llamafile/GGUF URL."
        )
    backend = LlamafileBackend(port=port, model=target)
    return PullTarget(
        backend, entry["display"], entry["size_gb"], entry["min_ram_gb"],
        is_catalog_name=True,
    )


class ProgressEmitter:
    """Turn raw ``(done, total, phase)`` chunk callbacks into SSE event dicts.

    Emits one ``start`` event the first time a phase is seen, then ``progress``
    events throttled to ``min_interval`` seconds (the final chunk of a phase
    always emits). ``emit`` receives each event dict.
    """

    def __init__(
        self,
        labels: dict,
        emit: Callable[[dict], None],
        min_interval: float = 0.25,
        _clock: Callable[[], float] = time.monotonic,
    ):
        self.labels = labels
        self.emit = emit
        self.min_interval = min_interval
        self._clock = _clock
        self._phase: Optional[str] = None
        self._phase_start = 0.0
        self._last_emit = 0.0

    def __call__(self, done: int, total: int, phase: str) -> None:
        now = self._clock()
        if phase != self._phase:
            self._phase = phase
            self._phase_start = now
            self._last_emit = float("-inf")  # ensure the first progress isn't throttled
            self.emit({
                "type": "start",
                "phase": phase,
                "label": self.labels.get(phase, phase),
                "total_mb": round(total / _MB, 1) if total else None,
            })
        is_final = bool(total) and done >= total
        if now - self._last_emit < self.min_interval and not is_final:
            return
        self._last_emit = now
        elapsed = max(now - self._phase_start, 1e-6)
        self.emit({
            "type": "progress",
            "phase": phase,
            "pct": round(done * 100 / total, 1) if total else 0,
            "downloaded_mb": round(done / _MB, 1),
            "total_mb": round(total / _MB, 1) if total else None,
            "speed_mbps": round((done / _MB) / elapsed, 1),
        })
