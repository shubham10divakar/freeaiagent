import hashlib

import pytest
from unittest.mock import patch

from freeaiagent.backends.llamafile import (
    IntegrityError,
    LlamafileBackend,
    _resp_total,
)


class FakeResp:
    """Minimal urlopen replacement; records the Range header it was given."""

    def __init__(self, req, *, status=200, body=b"", headers=None):
        self.req = req
        self.status = status
        self.headers = headers or {"Content-Length": str(len(body))}
        self._chunks = [body] if body else []

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def read(self, _n):
        return self._chunks.pop(0) if self._chunks else b""


def _patch_urlopen(fn):
    return patch("freeaiagent.backends.llamafile.urllib.request.urlopen", side_effect=fn)


@pytest.fixture
def backend():
    return LlamafileBackend(model="llama-3.2-3b")


# ── _resp_total ──────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_resp_total_plain_200():
    r = FakeResp(None, status=200, headers={"Content-Length": "100"})
    assert _resp_total(r, 0, 200) == 100


@pytest.mark.unit
def test_resp_total_206_uses_content_range():
    r = FakeResp(None, status=206, headers={"Content-Length": "40", "Content-Range": "bytes 60-99/100"})
    assert _resp_total(r, 60, 206) == 100


@pytest.mark.unit
def test_resp_total_206_without_range_adds_existing():
    r = FakeResp(None, status=206, headers={"Content-Length": "40"})
    assert _resp_total(r, 60, 206) == 100


# ── resume ───────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_resumes_from_partial(backend, tmp_path):
    dest = tmp_path / "m.bin"
    (tmp_path / "m.bin.part").write_bytes(b"AAA")  # 3 bytes already there

    seen = {}

    def fake(req):
        seen["range"] = req.headers.get("Range")
        return FakeResp(req, status=206, body=b"BB",
                        headers={"Content-Length": "2", "Content-Range": "bytes 3-4/5"})

    events = []
    with _patch_urlopen(fake):
        backend._download_file("http://x/m.bin", dest, on_chunk=lambda d, t, p: events.append((d, t)))

    assert seen["range"] == "bytes=3-"
    assert dest.read_bytes() == b"AAABB"
    assert events[0] == (3, 5)     # resume baseline reported
    assert events[-1] == (5, 5)    # completion
    assert not (tmp_path / "m.bin.part").exists()


@pytest.mark.unit
def test_restarts_when_server_ignores_range(backend, tmp_path):
    dest = tmp_path / "m.bin"
    (tmp_path / "m.bin.part").write_bytes(b"OLD")  # stale partial

    def fake(req):
        # Status 200 = server ignored our Range header; must restart cleanly.
        return FakeResp(req, status=200, body=b"NEW", headers={"Content-Length": "3"})

    with _patch_urlopen(fake):
        backend._download_file("http://x/m.bin", dest)

    assert dest.read_bytes() == b"NEW"


@pytest.mark.unit
def test_partial_kept_on_network_error(backend, tmp_path):
    dest = tmp_path / "m.bin"

    class Boom(FakeResp):
        def read(self, _n):
            raise ConnectionResetError("boom")

    def fake(req):
        return Boom(req, status=200, headers={"Content-Length": "10"})

    with _patch_urlopen(fake):
        with pytest.raises(ConnectionResetError):
            backend._download_file("http://x/m.bin", dest)

    # .part is preserved (even if empty) and dest never created, so a retry resumes.
    assert not dest.exists()


# ── force ────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_force_clears_stale_part_and_redownloads(backend, tmp_path):
    dest = tmp_path / "m.bin"
    dest.write_bytes(b"old")
    (tmp_path / "m.bin.part").write_bytes(b"STALE")

    seen = {}

    def fake(req):
        seen["range"] = req.headers.get("Range")
        return FakeResp(req, status=200, body=b"FRESH", headers={"Content-Length": "5"})

    with _patch_urlopen(fake):
        backend._download_file("http://x/m.bin", dest, force=True)

    assert seen["range"] is None        # no resume after a forced re-download
    assert dest.read_bytes() == b"FRESH"


# ── sha256 verification ──────────────────────────────────────────────────────

@pytest.mark.unit
def test_sha256_match_passes(backend, tmp_path):
    dest = tmp_path / "m.bin"
    body = b"hello world"
    digest = hashlib.sha256(body).hexdigest()

    with _patch_urlopen(lambda req: FakeResp(req, body=body)):
        backend._download_file("http://x/m.bin", dest, sha256=digest)

    assert dest.read_bytes() == body


@pytest.mark.unit
def test_sha256_mismatch_raises_and_removes_files(backend, tmp_path):
    dest = tmp_path / "m.bin"

    with _patch_urlopen(lambda req: FakeResp(req, body=b"hello")):
        with pytest.raises(IntegrityError):
            backend._download_file("http://x/m.bin", dest, sha256="00" * 32)

    assert not dest.exists()
    assert not (tmp_path / "m.bin.part").exists()


@pytest.mark.unit
def test_download_uses_catalog_sha256(tmp_path, monkeypatch):
    """download() threads a catalog-pinned checksum through to verification."""
    from freeaiagent import catalog

    backend = LlamafileBackend(model="llama-3.2-3b")
    dest = tmp_path / "model.llamafile"
    monkeypatch.setattr(backend, "_bin", lambda: dest)
    # Pin a checksum that the downloaded bytes won't match.
    entry = dict(catalog.CATALOG["llama-3.2-3b"], sha256="ab" * 32)
    monkeypatch.setitem(catalog.CATALOG, "llama-3.2-3b", entry)

    with _patch_urlopen(lambda req: FakeResp(req, body=b"wrong")):
        with pytest.raises(IntegrityError):
            backend.download()
