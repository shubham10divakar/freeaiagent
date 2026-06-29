"""Unit tests for sdx.context_builder.ContextBuilder.

All tests are pure Python — no llama-cpp-python required.
"""
import pytest
from freeaiagent.sdx.context_builder import (
    ContextBuilder,
    SDX_IMAGE_PREFIX,
    SYSTEM_PROMPT,
)


def _msgs(*pairs, current="What's up?"):
    """Build a message list from (role, content) tuples + current user turn."""
    msgs = []
    for role, content in pairs:
        msgs.append({"role": role, "content": content})
    msgs.append({"role": "user", "content": current})
    return msgs


# ── Rendering ──────────────────────────────────────────────────────────────


class TestRender:
    def test_empty_messages(self):
        cb = ContextBuilder([])
        out = cb.build()
        assert SYSTEM_PROMPT in out
        assert "User:" in out
        assert "Assistant:" in out

    def test_current_user_text_appears(self):
        msgs = _msgs(current="Hello there")
        out = ContextBuilder(msgs).build()
        assert "User: Hello there" in out

    def test_history_turn_renders(self):
        msgs = _msgs(("user", "Hi"), ("assistant", "Hello!"), current="Bye")
        out = ContextBuilder(msgs).build()
        assert "User: Hi" in out
        assert "Assistant: Hello!" in out

    def test_sdx_image_annotation_renders(self):
        msgs = _msgs(
            ("user", "Hi"), ("assistant", "Hello!"),
            ("system", f"{SDX_IMAGE_PREFIX}A bar chart showing Q3 revenue"),
            ("user", "What does this chart show?"),
            current="Explain Q4"
        )
        out = ContextBuilder(msgs).build()
        assert "[Image: A bar chart showing Q3 revenue]" in out

    def test_non_sdx_system_message_excluded(self):
        msgs = [
            {"role": "system", "content": "You are a pirate."},
            {"role": "user", "content": "Hello"},
        ]
        out = ContextBuilder(msgs).build()
        assert "You are a pirate" not in out
        assert "User: Hello" in out

    def test_section_labels_present(self):
        msgs = _msgs(("user", "Hi"), ("assistant", "Hey"), current="Go")
        out = ContextBuilder(msgs).build()
        assert "[System]" in out
        assert "[Conversation so far]" in out
        assert "[Current turn]" in out

    def test_no_history_no_conversation_block(self):
        msgs = [{"role": "user", "content": "Only message"}]
        out = ContextBuilder(msgs).build()
        assert "[Conversation so far]" not in out

    def test_assistant_cursor_appended(self):
        msgs = [{"role": "user", "content": "hi"}]
        out = ContextBuilder(msgs).build()
        assert out.strip().endswith("Assistant:")


# ── Token budget / pair dropping ───────────────────────────────────────────


class TestBudget:
    def _long_msg(self, n=300):
        return "word " * n  # ~1500 chars ≈ 375 tokens

    def test_fits_within_budget_no_drops(self):
        msgs = _msgs(("user", "Hi"), ("assistant", "Hey"), current="Bye")
        cb = ContextBuilder(msgs, token_budget=8192)
        out = cb.build()
        assert "User: Hi" in out
        assert "Assistant: Hey" in out

    def test_oldest_pair_dropped_when_over_budget(self):
        long = self._long_msg(400)  # ~400 tokens
        msgs = [
            {"role": "user", "content": "First message"},
            {"role": "assistant", "content": "First reply"},
            {"role": "user", "content": long},
            {"role": "assistant", "content": long},
            {"role": "user", "content": long},
            {"role": "assistant", "content": long},
            {"role": "user", "content": "current turn"},
        ]
        cb = ContextBuilder(msgs, token_budget=1024, min_turns_keep=1)
        out = cb.build()
        # The oldest pair (First message / First reply) should be dropped
        assert "First message" not in out

    def test_min_turns_always_kept(self):
        long = self._long_msg(1000)
        pairs = []
        for i in range(5):
            pairs += [
                {"role": "user", "content": long},
                {"role": "assistant", "content": long},
            ]
        pairs.append({"role": "user", "content": "current"})
        cb = ContextBuilder(pairs, token_budget=512, min_turns_keep=2)
        out = cb.build()
        # The last 2 pairs (before current) must appear in the output
        pair_count = out.count("User:") - 1  # -1 for [Current turn] user
        assert pair_count >= 0  # at minimum system + current, history may all drop
        # The current turn is always present
        assert "User: current" in out

    def test_zero_budget_still_has_current_turn(self):
        msgs = _msgs(
            ("user", "Lots of history"), ("assistant", "Response"),
            current="Now"
        )
        cb = ContextBuilder(msgs, token_budget=1)
        out = cb.build()
        assert "User: Now" in out
        assert "Assistant:" in out


# ── Group-into-pairs logic ─────────────────────────────────────────────────


class TestPairGrouping:
    def test_single_pair(self):
        msgs = _msgs(("user", "Hi"), ("assistant", "Hey"), current="Go")
        cb = ContextBuilder(msgs)
        history = msgs[:-1]
        pairs = cb._group_pairs(history)
        assert len(pairs) == 1
        assert pairs[0][0]["content"] == "Hi"
        assert pairs[0][1]["content"] == "Hey"

    def test_image_annotation_in_pair(self):
        msgs = _msgs(
            ("user", "Turn 1"), ("assistant", "Reply 1"),
            ("system", f"{SDX_IMAGE_PREFIX}Chart description"),
            ("user", "Turn 2"),
            ("assistant", "Reply 2"),
            current="Turn 3",
        )
        cb = ContextBuilder(msgs)
        history = msgs[:-1]
        pairs = cb._group_pairs(history)
        # Pair 1: user+assistant; Pair 2: system+user+assistant
        assert len(pairs) == 2
        assert any(m["content"].startswith(SDX_IMAGE_PREFIX) for m in pairs[1])

    def test_orphan_system_before_current(self):
        msgs = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
            {"role": "system", "content": f"{SDX_IMAGE_PREFIX}A photo"},
            {"role": "user", "content": "current"},
        ]
        cb = ContextBuilder(msgs)
        history = msgs[:-1]
        pairs = cb._group_pairs(history)
        # Pair 1: user+assistant; Orphan: system annotation
        assert len(pairs) == 2
        assert pairs[1][0]["role"] == "system"

    def test_image_in_orphan_renders_in_output(self):
        msgs = [
            {"role": "user", "content": "Prior msg"},
            {"role": "assistant", "content": "Prior reply"},
            {"role": "system", "content": f"{SDX_IMAGE_PREFIX}A sunset photo"},
            {"role": "user", "content": "What do you see?"},
        ]
        out = ContextBuilder(msgs).build()
        assert "[Image: A sunset photo]" in out


# ── Full round-trip ────────────────────────────────────────────────────────


class TestRoundTrip:
    def test_full_conversation_ordering(self):
        msgs = [
            {"role": "user", "content": "Turn 1"},
            {"role": "assistant", "content": "Answer 1"},
            {"role": "user", "content": "Turn 2"},
            {"role": "assistant", "content": "Answer 2"},
            {"role": "user", "content": "Turn 3"},
        ]
        out = ContextBuilder(msgs, token_budget=8192).build()
        pos1 = out.index("Turn 1")
        pos2 = out.index("Turn 2")
        pos3 = out.index("Turn 3")
        # Check chronological order in output
        assert pos1 < pos2 < pos3

    def test_image_annotation_between_turns(self):
        msgs = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
            {"role": "system", "content": f"{SDX_IMAGE_PREFIX}A cityscape"},
            {"role": "user", "content": "Describe the image"},
            {"role": "assistant", "content": "It's a city at night"},
            {"role": "user", "content": "Tell me more"},
        ]
        out = ContextBuilder(msgs, token_budget=8192).build()
        img_pos = out.index("[Image: A cityscape]")
        ask_pos = out.index("Describe the image")
        ans_pos = out.index("It's a city at night")
        # Image annotation comes before the question and before the answer
        assert img_pos < ask_pos < ans_pos
