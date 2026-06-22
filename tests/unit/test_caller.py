import pytest

from freeaiagent.caller import resolve_session, DEFAULT_SESSION


@pytest.mark.unit
def test_body_session_id_wins_over_header():
    assert resolve_session("explicit", "header-caller") == "explicit"


@pytest.mark.unit
def test_header_used_when_body_is_default():
    assert resolve_session("default", "magpie") == "magpie"


@pytest.mark.unit
def test_header_stripped():
    assert resolve_session("default", "  magpie  ") == "magpie"


@pytest.mark.unit
def test_falls_back_to_default():
    assert resolve_session("default", None) == DEFAULT_SESSION
    assert resolve_session("default", "") == DEFAULT_SESSION
    assert resolve_session(None, None) == DEFAULT_SESSION


@pytest.mark.unit
def test_blank_header_falls_back():
    assert resolve_session("default", "   ") == DEFAULT_SESSION
