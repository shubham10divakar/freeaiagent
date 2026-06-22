import pytest

from freeaiagent import tools


@pytest.mark.unit
def test_register_get_and_list(isolated_tools):
    tools.register(
        "weather", "Get weather", "http://x/weather",
        {"type": "object", "properties": {"city": {"type": "string"}}},
    )
    assert tools.get("weather")["endpoint"] == "http://x/weather"
    assert [t["name"] for t in tools.all_tools()] == ["weather"]


@pytest.mark.unit
def test_unregister(isolated_tools):
    tools.register("a", "d", "http://x")
    assert tools.unregister("a") is True
    assert tools.unregister("a") is False
    assert tools.all_tools() == []


@pytest.mark.unit
def test_openai_spec_shape(isolated_tools):
    tools.register(
        "weather", "Get weather", "http://x",
        {"type": "object", "properties": {"city": {"type": "string"}}},
    )
    spec = tools.openai_spec()
    assert spec[0]["type"] == "function"
    assert spec[0]["function"]["name"] == "weather"
    assert spec[0]["function"]["parameters"]["properties"]["city"]["type"] == "string"


@pytest.mark.unit
def test_register_defaults_empty_parameters(isolated_tools):
    rec = tools.register("noargs", "d", "http://x")
    assert rec["parameters"] == {"type": "object", "properties": {}}


class _ToolThenAnswerBackend:
    def __init__(self):
        self.calls = 0

    async def chat(self, messages, model):
        return "plain"

    async def chat_completion(self, messages, model, tools=None):
        self.calls += 1
        if self.calls == 1:
            return {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {"id": "c1", "function": {"name": "weather", "arguments": '{"city":"Paris"}'}}
                ],
            }
        return {"role": "assistant", "content": "It is sunny in Paris."}


@pytest.mark.unit
async def test_run_executes_tool_then_answers(isolated_tools, monkeypatch):
    tools.register("weather", "Get weather", "http://x")

    async def fake_execute(name, args):
        assert name == "weather"
        assert args == {"city": "Paris"}
        return "SUNNY"

    monkeypatch.setattr(tools, "execute", fake_execute)

    backend = _ToolThenAnswerBackend()
    messages = [{"role": "user", "content": "weather in Paris?"}]
    result = await tools.run(backend, "m", messages)

    assert result == "It is sunny in Paris."
    assert backend.calls == 2
    assert any(m.get("role") == "tool" and m.get("content") == "SUNNY" for m in messages)


@pytest.mark.unit
async def test_run_without_registered_tools_falls_back_to_chat(isolated_tools):
    class B:
        async def chat(self, messages, model):
            return "plain answer"

    result = await tools.run(B(), "m", [{"role": "user", "content": "x"}])
    assert result == "plain answer"


@pytest.mark.unit
async def test_run_stops_after_max_rounds(isolated_tools, monkeypatch):
    tools.register("loop", "always calls", "http://x")

    async def fake_execute(name, args):
        return "ok"

    monkeypatch.setattr(tools, "execute", fake_execute)

    class AlwaysCalls:
        async def chat(self, messages, model):
            return "plain"

        async def chat_completion(self, messages, model, tools=None):
            return {"role": "assistant", "content": "still going",
                    "tool_calls": [{"id": "c", "function": {"name": "loop", "arguments": "{}"}}]}

    result = await tools.run(AlwaysCalls(), "m", [{"role": "user", "content": "x"}], max_rounds=3)
    assert result == "still going"  # last content returned when loop doesn't converge
