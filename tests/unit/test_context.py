import pytest
import freeaiagent.context as ctx


@pytest.mark.unit
def test_empty_db_returns_no_messages(isolated_db):
    assert ctx.all_messages() == []
    assert ctx.count() == 0


@pytest.mark.unit
def test_append_and_retrieve(isolated_db):
    ctx.append("user", "hello")
    ctx.append("assistant", "hi there")
    msgs = ctx.all_messages()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[0]["content"] == "hello"
    assert msgs[1]["role"] == "assistant"


@pytest.mark.unit
def test_as_llm_messages_strips_timestamp(isolated_db):
    ctx.append("user", "test")
    llm_msgs = ctx.as_llm_messages()
    assert llm_msgs == [{"role": "user", "content": "test"}]


@pytest.mark.unit
def test_count(isolated_db):
    ctx.append("user", "a")
    ctx.append("assistant", "b")
    ctx.append("user", "c")
    assert ctx.count() == 3


@pytest.mark.unit
def test_clear_returns_count_and_empties_db(isolated_db):
    ctx.append("user", "x")
    ctx.append("assistant", "y")
    n = ctx.clear()
    assert n == 2
    assert ctx.count() == 0
    assert ctx.all_messages() == []


@pytest.mark.unit
def test_clear_on_empty_db(isolated_db):
    assert ctx.clear() == 0


@pytest.mark.unit
def test_messages_preserve_order(isolated_db):
    for i in range(5):
        ctx.append("user", f"msg {i}")
    contents = [m["content"] for m in ctx.all_messages()]
    assert contents == [f"msg {i}" for i in range(5)]


@pytest.mark.unit
def test_timestamp_is_present(isolated_db):
    ctx.append("user", "timestamped")
    msg = ctx.all_messages()[0]
    assert "timestamp" in msg
    assert msg["timestamp"]  # non-empty
