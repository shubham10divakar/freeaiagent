import pytest


@pytest.mark.integration
def test_get_context_empty(client):
    r = client.get("/context")
    assert r.status_code == 200
    data = r.json()
    assert data["messages"] == []
    assert data["total"] == 0


@pytest.mark.integration
def test_get_context_after_chat(client):
    client.post("/chat", json={"message": "hello"})
    r = client.get("/context")
    data = r.json()
    assert data["total"] == 2
    assert data["messages"][0]["role"] == "user"
    assert data["messages"][0]["content"] == "hello"
    assert data["messages"][1]["role"] == "assistant"


@pytest.mark.integration
def test_delete_context_clears_messages(client):
    client.post("/chat", json={"message": "hello"})
    client.post("/chat", json={"message": "world"})
    r = client.delete("/context")
    assert r.status_code == 200
    assert r.json()["cleared"] == 4

    r = client.get("/context")
    assert r.json()["total"] == 0


@pytest.mark.integration
def test_delete_context_on_empty_is_safe(client):
    r = client.delete("/context")
    assert r.status_code == 200
    assert r.json()["cleared"] == 0


@pytest.mark.integration
def test_context_messages_have_timestamps(client):
    client.post("/chat", json={"message": "ts test"})
    msgs = client.get("/context").json()["messages"]
    for m in msgs:
        assert "timestamp" in m
        assert m["timestamp"]
