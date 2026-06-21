import pytest


@pytest.mark.integration
def test_list_sessions_empty(client):
    r = client.get("/sessions")
    assert r.status_code == 200
    assert r.json()["sessions"] == []


@pytest.mark.integration
def test_session_created_on_first_chat(client):
    client.post("/chat", json={"message": "hello", "session_id": "s1"})
    sessions = client.get("/sessions").json()["sessions"]
    assert len(sessions) == 1
    assert sessions[0]["id"] == "s1"


@pytest.mark.integration
def test_session_title_set_from_first_message(client):
    client.post("/chat", json={"message": "What is the meaning of life?", "session_id": "s1"})
    s = client.get("/sessions").json()["sessions"][0]
    assert s["title"] == "What is the meaning of life?"


@pytest.mark.integration
def test_long_first_message_truncated_to_60(client):
    long_msg = "A" * 80
    client.post("/chat", json={"message": long_msg, "session_id": "s1"})
    s = client.get("/sessions").json()["sessions"][0]
    assert len(s["title"]) <= 60


@pytest.mark.integration
def test_chat_response_includes_session_id(client):
    r = client.post("/chat", json={"message": "hi", "session_id": "my-sess"})
    assert r.json()["session_id"] == "my-sess"


@pytest.mark.integration
def test_separate_sessions_have_separate_context(client):
    client.post("/chat", json={"message": "I am session A", "session_id": "a"})
    client.post("/chat", json={"message": "I am session B", "session_id": "b"})
    a_msgs = client.get("/context", params={"session": "a"}).json()["messages"]
    b_msgs = client.get("/context", params={"session": "b"}).json()["messages"]
    assert a_msgs[0]["content"] == "I am session A"
    assert b_msgs[0]["content"] == "I am session B"
    assert len(a_msgs) == 2  # user + assistant
    assert len(b_msgs) == 2


@pytest.mark.integration
def test_create_session_explicit(client):
    r = client.post("/sessions", json={"id": "proj-x", "title": "Project X"})
    assert r.status_code == 201
    assert r.json()["id"] == "proj-x"
    assert r.json()["title"] == "Project X"


@pytest.mark.integration
def test_create_session_auto_id(client):
    r = client.post("/sessions", json={})
    assert r.status_code == 201
    assert r.json()["id"]  # UUID generated


@pytest.mark.integration
def test_rename_session(client):
    client.post("/sessions", json={"id": "s1", "title": "Old"})
    r = client.patch("/sessions/s1", json={"title": "New Title"})
    assert r.status_code == 200
    assert r.json()["title"] == "New Title"
    sessions = client.get("/sessions").json()["sessions"]
    assert any(s["title"] == "New Title" for s in sessions)


@pytest.mark.integration
def test_rename_nonexistent_session_returns_404(client):
    r = client.patch("/sessions/ghost", json={"title": "whatever"})
    assert r.status_code == 404


@pytest.mark.integration
def test_delete_session(client):
    client.post("/sessions", json={"id": "s1"})
    r = client.delete("/sessions/s1")
    assert r.status_code == 200
    assert r.json()["deleted"] == "s1"
    sessions = client.get("/sessions").json()["sessions"]
    assert not any(s["id"] == "s1" for s in sessions)


@pytest.mark.integration
def test_delete_session_removes_its_messages(client):
    client.post("/chat", json={"message": "hi", "session_id": "s1"})
    client.delete("/sessions/s1")
    msgs = client.get("/context", params={"session": "s1"}).json()["messages"]
    assert msgs == []


@pytest.mark.integration
def test_delete_nonexistent_session_returns_404(client):
    r = client.delete("/sessions/ghost")
    assert r.status_code == 404


@pytest.mark.integration
def test_multiple_sessions_listed_newest_first(client):
    client.post("/chat", json={"message": "first", "session_id": "old"})
    client.post("/chat", json={"message": "second", "session_id": "new"})
    sessions = client.get("/sessions").json()["sessions"]
    ids = [s["id"] for s in sessions]
    assert ids.index("new") < ids.index("old")


@pytest.mark.integration
def test_message_count_increments(client):
    client.post("/chat", json={"message": "one", "session_id": "s1"})
    client.post("/chat", json={"message": "two", "session_id": "s1"})
    s = next(s for s in client.get("/sessions").json()["sessions"] if s["id"] == "s1")
    assert s["message_count"] == 4  # 2 user + 2 assistant


@pytest.mark.integration
def test_default_session_used_when_no_session_id(client):
    client.post("/chat", json={"message": "hi"})
    r = client.get("/context")  # no ?session= → defaults to "default"
    assert r.json()["total"] == 2
