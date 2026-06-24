import pytest
from fastapi.testclient import TestClient


@pytest.mark.integration
def test_register_list_delete_tool(client):
    r = client.post("/tools/register", json={
        "name": "weather", "description": "Get weather",
        "endpoint": "http://localhost:9/weather",
    })
    assert r.status_code == 201
    assert r.json()["name"] == "weather"

    listed = client.get("/tools").json()["tools"]
    assert any(t["name"] == "weather" for t in listed)

    assert client.delete("/tools/weather").status_code == 200
    assert client.get("/tools").json()["tools"] == []


@pytest.mark.integration
def test_delete_unknown_tool_returns_404(client):
    assert client.delete("/tools/nope").status_code == 404


@pytest.mark.integration
def test_chat_with_tools_runs_loop(isolated_config, isolated_db, isolated_tools, monkeypatch):
    from freeaiagent import tools

    tools.register("weather", "Get weather", "http://x")

    class FakeToolBackend:
        def __init__(self):
            self.n = 0

        async def chat(self, messages, model):
            return "plain (tools not used)"

        async def chat_completion(self, messages, model, tools=None):
            self.n += 1
            if self.n == 1:
                assert tools, "tools spec should be passed on the first round"
                return {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "function": {"name": "weather", "arguments": "{}"}}]}
            return {"role": "assistant", "content": "sunny answer"}

    fake = FakeToolBackend()

    async def _resolve(override_model=None, override_backend=None, override_max_messages=None):
        return fake, "test-model", 0

    async def _fake_exec(name, args):
        return "SUNNY"

    monkeypatch.setattr("freeaiagent.router.resolve", _resolve)
    monkeypatch.setattr(tools, "execute", _fake_exec)

    from freeaiagent.main import app
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "weather?", "tools": True})

    assert r.status_code == 200
    assert r.json()["response"] == "sunny answer"
    assert fake.n == 2


@pytest.mark.integration
def test_chat_without_tools_flag_ignores_registry(isolated_config, isolated_db, isolated_tools, monkeypatch):
    from freeaiagent import tools

    tools.register("weather", "Get weather", "http://x")

    class Backend:
        async def chat(self, messages, model):
            return "plain answer"

        async def chat_completion(self, messages, model, tools=None):
            raise AssertionError("chat_completion should not be called when tools=false")

    async def _resolve(override_model=None, override_backend=None, override_max_messages=None):
        return Backend(), "test-model", 0

    monkeypatch.setattr("freeaiagent.router.resolve", _resolve)

    from freeaiagent.main import app
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "hi"})  # tools defaults to False

    assert r.status_code == 200
    assert r.json()["response"] == "plain answer"
