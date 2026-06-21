"""
Smoke tests — real server, real HTTP, real LLM backend.

Requirements:
  - Ollama running on localhost:11434 with at least one model pulled, OR
  - GROQ_API_KEY set in environment

Run with:
  pytest tests/smoke/ -m smoke -v
"""
import time
import threading
import pytest
import httpx2 as httpx
import uvicorn

BASE = "http://localhost:17731"  # dedicated port so it never clashes with a real instance


class _Server(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        from freeaiagent.main import app
        self.config = uvicorn.Config(app, host="127.0.0.1", port=17731, log_level="error")
        self.server = uvicorn.Server(self.config)

    def run(self):
        self.server.run()

    def stop(self):
        self.server.should_exit = True


def _wait_ready(timeout: float = 10.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{BASE}/health", timeout=1.0)
            if r.status_code == 200:
                return
        except Exception:
            time.sleep(0.2)
    raise RuntimeError("freeaiagent server did not start in time")


@pytest.fixture(scope="module")
def live_server():
    srv = _Server()
    srv.start()
    _wait_ready()
    yield BASE
    srv.stop()


@pytest.fixture(scope="module", autouse=True)
def require_backend(live_server):
    """Skip entire smoke module if no backend is available."""
    r = httpx.get(f"{live_server}/health", timeout=5.0)
    if r.json().get("status") != "ok":
        pytest.skip("No LLM backend available — skipping smoke tests")


# ---------------------------------------------------------------------------
# Smoke tests
# ---------------------------------------------------------------------------

@pytest.mark.smoke
def test_smoke_health(live_server):
    r = httpx.get(f"{live_server}/health", timeout=5.0)
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


@pytest.mark.smoke
def test_smoke_models(live_server):
    r = httpx.get(f"{live_server}/models", timeout=10.0)
    assert r.status_code == 200
    assert len(r.json()["models"]) > 0


@pytest.mark.smoke
def test_smoke_chat(live_server):
    # Clear first so context is predictable
    httpx.delete(f"{live_server}/context", timeout=5.0)
    r = httpx.post(
        f"{live_server}/chat",
        json={"message": "Reply with exactly the word: PONG"},
        timeout=60.0,
    )
    assert r.status_code == 200
    data = r.json()
    assert data["response"]
    assert data["context_length"] == 2


@pytest.mark.smoke
def test_smoke_context_persists(live_server):
    httpx.delete(f"{live_server}/context", timeout=5.0)
    httpx.post(f"{live_server}/chat", json={"message": "Remember: MAGPIE"}, timeout=60.0)
    r = httpx.get(f"{live_server}/context", timeout=5.0)
    data = r.json()
    assert data["total"] == 2
    assert any("MAGPIE" in m["content"] for m in data["messages"])


@pytest.mark.smoke
def test_smoke_task_no_context_side_effect(live_server):
    httpx.delete(f"{live_server}/context", timeout=5.0)
    httpx.post(f"{live_server}/task", json={"task": "Say hello"}, timeout=60.0)
    r = httpx.get(f"{live_server}/context", timeout=5.0)
    assert r.json()["total"] == 0  # task must not write to context


@pytest.mark.smoke
def test_smoke_task_returns_result(live_server):
    r = httpx.post(
        f"{live_server}/task",
        json={"task": "What is 2 + 2? Reply with just the number."},
        timeout=60.0,
    )
    assert r.status_code == 200
    assert r.json()["result"]


@pytest.mark.smoke
def test_smoke_context_clear(live_server):
    httpx.post(f"{live_server}/chat", json={"message": "hi"}, timeout=60.0)
    r = httpx.delete(f"{live_server}/context", timeout=5.0)
    assert r.status_code == 200
    assert r.json()["cleared"] >= 2
    assert httpx.get(f"{live_server}/context", timeout=5.0).json()["total"] == 0
