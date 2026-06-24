import pytest


@pytest.mark.integration
def test_chat_returns_response(client):
    r = client.post("/chat", json={"message": "hello"})
    assert r.status_code == 200
    data = r.json()
    assert "response" in data
    assert data["response"] == "Hello from mock backend"
    assert data["model"] == "Llama-3.2-1B-Instruct"
    assert "backend" in data
    assert data["context_length"] == 2  # user + assistant appended


@pytest.mark.integration
def test_chat_builds_up_context(client):
    client.post("/chat", json={"message": "first"})
    r = client.post("/chat", json={"message": "second"})
    assert r.json()["context_length"] == 4  # 2 turns × 2 messages each


@pytest.mark.integration
def test_chat_passes_system_prompt(client, patched_router):
    client.post("/chat", json={"message": "hi", "system": "You are a pirate."})
    call_args = patched_router.chat.call_args
    messages = call_args[0][0]
    assert messages[0] == {"role": "system", "content": "You are a pirate."}


@pytest.mark.integration
def test_chat_without_system_has_no_system_message(client, patched_router):
    client.post("/chat", json={"message": "hi"})
    messages = patched_router.chat.call_args[0][0]
    roles = [m["role"] for m in messages]
    assert "system" not in roles


@pytest.mark.integration
def test_chat_history_sent_to_backend(client, patched_router):
    client.post("/chat", json={"message": "turn one"})
    client.post("/chat", json={"message": "turn two"})
    # second call's messages should include prior turn
    messages = patched_router.chat.call_args[0][0]
    contents = [m["content"] for m in messages]
    assert "turn one" in contents
    assert "Hello from mock backend" in contents
    assert "turn two" in contents


@pytest.mark.integration
def test_chat_model_override(client, patched_router):
    r = client.post("/chat", json={"message": "hi", "model": "mistral:7b"})
    assert r.status_code == 200
    assert r.json()["model"] == "mistral:7b"


@pytest.mark.integration
def test_chat_backend_override_unknown_returns_503(client):
    r = client.post("/chat", json={"message": "hi", "backend": "nonexistent"})
    assert r.status_code == 503


@pytest.mark.integration
def test_chat_uses_x_caller_id_header_as_session(client):
    r = client.post("/chat", json={"message": "hi"}, headers={"X-Caller-ID": "magpie"})
    assert r.status_code == 200
    assert r.json()["session_id"] == "magpie"


@pytest.mark.integration
def test_chat_body_session_id_overrides_header(client):
    r = client.post(
        "/chat",
        json={"message": "hi", "session_id": "explicit"},
        headers={"X-Caller-ID": "magpie"},
    )
    assert r.json()["session_id"] == "explicit"


@pytest.mark.integration
def test_chat_callers_have_separate_context(client):
    client.post("/chat", json={"message": "a1"}, headers={"X-Caller-ID": "app-a"})
    client.post("/chat", json={"message": "a2"}, headers={"X-Caller-ID": "app-a"})
    r_b = client.post("/chat", json={"message": "b1"}, headers={"X-Caller-ID": "app-b"})
    # app-b is a fresh thread: just its own user + assistant
    assert r_b.json()["context_length"] == 2
    assert r_b.json()["session_id"] == "app-b"


@pytest.mark.integration
def test_chat_respects_max_messages_window(isolated_config, isolated_db, patched_router):
    import json
    from freeaiagent import config as cfg

    # set a window of 2 messages
    cfg.set_value("max_messages", 2)

    from freeaiagent.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        c.post("/chat", json={"message": "turn one"})
        c.post("/chat", json={"message": "turn two"})
        c.post("/chat", json={"message": "turn three"})

    # last call to backend should have seen at most 2 prior messages + the new one
    last_call_messages = patched_router.chat.call_args[0][0]
    # with window=2, only last 2 stored messages sent; total sent = 2 + current = 3
    assert len(last_call_messages) <= 3


@pytest.mark.integration
def test_chat_ensemble_returns_votes(client):
    r = client.post("/chat", json={"message": "hi", "ensemble": ["m1", "m2"]})
    assert r.status_code == 200
    data = r.json()
    assert "ensemble_votes" in data
    assert len(data["ensemble_votes"]) == 2
    assert {v["model"] for v in data["ensemble_votes"]} == {"m1", "m2"}
    assert data["response"] == "Hello from mock backend"


@pytest.mark.integration
def test_chat_single_ensemble_model_is_normal_chat(client):
    # < 2 models => no ensemble, no votes key
    r = client.post("/chat", json={"message": "hi", "ensemble": ["m1"]})
    assert r.status_code == 200
    assert "ensemble_votes" not in r.json()


@pytest.mark.integration
def test_chat_summarize_strategy_compresses_history(isolated_config, isolated_db, patched_router):
    from freeaiagent import config as cfg, context
    from freeaiagent.summarize import SUMMARY_PREFIX

    cfg.set_value("context_strategy", "summarize")
    cfg.set_value("summarize_threshold", 2)
    cfg.set_value("summarize_batch", 2)

    for i in range(2):
        context.append("user", f"u{i}", session_id="default")
        context.append("assistant", f"a{i}", session_id="default")

    from freeaiagent.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        r = c.post("/chat", json={"message": "new question"})

    assert r.status_code == 200
    msgs = context.all_messages(session_id="default")
    assert msgs[0]["role"] == "system"
    assert msgs[0]["content"].startswith(SUMMARY_PREFIX)


@pytest.mark.integration
def test_chat_per_call_max_messages_override(isolated_config, isolated_db, patched_router):
    # No global window set (unlimited); the per-call override should still trim.
    from freeaiagent.main import app
    from fastapi.testclient import TestClient
    with TestClient(app) as c:
        c.post("/chat", json={"message": "turn one"})
        c.post("/chat", json={"message": "turn two"})
        c.post("/chat", json={"message": "turn three", "max_messages": 2})

    last_call_messages = patched_router.chat.call_args[0][0]
    assert len(last_call_messages) <= 3
