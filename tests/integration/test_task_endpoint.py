import pytest


@pytest.mark.integration
def test_task_returns_result(client):
    r = client.post("/task", json={"task": "summarize something"})
    assert r.status_code == 200
    data = r.json()
    assert "result" in data
    assert data["result"] == "Hello from mock backend"
    assert "model" in data


@pytest.mark.integration
def test_task_does_not_affect_context(client):
    client.post("/task", json={"task": "one-shot task"})
    r = client.get("/context")
    assert r.json()["total"] == 0  # task must never touch context


@pytest.mark.integration
def test_task_with_input_combined(client, patched_router):
    client.post("/task", json={"task": "summarize", "input": "some long text"})
    messages = patched_router.chat.call_args[0][0]
    user_msg = next(m for m in messages if m["role"] == "user")
    assert "summarize" in user_msg["content"]
    assert "some long text" in user_msg["content"]


@pytest.mark.integration
def test_task_uses_default_system_prompt(client, patched_router):
    client.post("/task", json={"task": "do something"})
    messages = patched_router.chat.call_args[0][0]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert system_msg["content"]  # non-empty default


@pytest.mark.integration
def test_task_custom_system_prompt(client, patched_router):
    client.post("/task", json={"task": "x", "system": "custom system"})
    messages = patched_router.chat.call_args[0][0]
    system_msg = next(m for m in messages if m["role"] == "system")
    assert system_msg["content"] == "custom system"


@pytest.mark.integration
def test_task_model_override(client, patched_router):
    client.post("/task", json={"task": "x", "model": "mistral:7b"})
    _, model = patched_router.chat.call_args[0]
    assert model == "mistral:7b"
