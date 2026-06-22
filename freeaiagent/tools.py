"""Tool/function-calling registry and execution loop.

Apps register an HTTP tool once; thereafter the model may call it mid-chat.
A registered tool is `{name, description, endpoint, parameters}` where
`parameters` is an OpenAI/JSON-Schema parameter object. When the model emits
tool calls, we POST the arguments to the tool's `endpoint` and feed the result
back, looping until the model produces a final answer.

Note: tool calling requires a backend/model that supports the OpenAI tool
protocol (Groq, most hosted providers, some local models). Backends without
support fall back to answering normally (see BaseBackend.chat_completion).
"""
import json
from pathlib import Path
from typing import List, Dict, Optional

import httpx2 as httpx

TOOLS_FILE = Path.home() / ".freeaiagent" / "tools.json"

_EMPTY_PARAMS = {"type": "object", "properties": {}}


def _load() -> dict:
    if not TOOLS_FILE.exists():
        return {}
    try:
        return json.loads(TOOLS_FILE.read_text())
    except Exception:
        return {}


def _save(data: dict) -> None:
    TOOLS_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOOLS_FILE.write_text(json.dumps(data, indent=2))


def register(name: str, description: str, endpoint: str, parameters: Optional[dict] = None) -> dict:
    tools = _load()
    record = {
        "name": name,
        "description": description,
        "endpoint": endpoint,
        "parameters": parameters or _EMPTY_PARAMS,
    }
    tools[name] = record
    _save(tools)
    return record


def all_tools() -> List[dict]:
    return list(_load().values())


def get(name: str) -> Optional[dict]:
    return _load().get(name)


def unregister(name: str) -> bool:
    tools = _load()
    if name in tools:
        del tools[name]
        _save(tools)
        return True
    return False


def openai_spec() -> List[dict]:
    """Registered tools in OpenAI `tools=[...]` format."""
    return [
        {
            "type": "function",
            "function": {
                "name": t["name"],
                "description": t["description"],
                "parameters": t.get("parameters", _EMPTY_PARAMS),
            },
        }
        for t in all_tools()
    ]


async def execute(name: str, arguments: dict) -> str:
    """Invoke a registered tool by POSTing arguments to its endpoint."""
    tool = get(name)
    if tool is None:
        return f"[error: tool '{name}' is not registered]"
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            r = await client.post(tool["endpoint"], json=arguments)
            r.raise_for_status()
            return r.text
    except Exception as e:
        return f"[error calling tool '{name}': {e}]"


async def run(backend, model: str, messages: List[Dict], max_rounds: int = 4) -> str:
    """Run the tool-call loop. Returns the model's final text answer.

    `messages` is mutated in place with the assistant tool-call turns and tool
    results. If no tools are registered, falls back to a plain chat.
    """
    spec = openai_spec()
    if not spec:
        return await backend.chat(messages, model)

    for _ in range(max_rounds):
        message = await backend.chat_completion(messages, model, tools=spec)
        messages.append(message)
        tool_calls = message.get("tool_calls")
        if not tool_calls:
            return message.get("content") or ""
        for call in tool_calls:
            fn = call.get("function", {})
            name = fn.get("name", "")
            try:
                args = json.loads(fn.get("arguments") or "{}")
            except json.JSONDecodeError:
                args = {}
            result = await execute(name, args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id", ""),
                "name": name,
                "content": result,
            })

    # ran out of rounds — return the most recent assistant text we have
    for message in reversed(messages):
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"]
    return "[tool loop did not converge]"
