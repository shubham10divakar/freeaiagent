import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

DB_FILE = Path.home() / ".freeaiagent" / "context.db"


def _conn() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            role      TEXT NOT NULL,
            content   TEXT NOT NULL,
            timestamp TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def append(role: str, content: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (role, content, timestamp) VALUES (?, ?, ?)",
            (role, content, ts),
        )


def all_messages() -> List[Dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM messages ORDER BY id"
        ).fetchall()
    return [dict(r) for r in rows]


def as_llm_messages(max_messages: int = 0) -> List[Dict]:
    """Return messages in the format LLM backends expect.

    max_messages: keep only the last N messages (0 = all).
    Always keeps pairs intact — truncates from the oldest user turn.
    """
    msgs = [{"role": m["role"], "content": m["content"]} for m in all_messages()]
    if max_messages > 0 and len(msgs) > max_messages:
        msgs = msgs[-max_messages:]
    return msgs


def clear() -> int:
    with _conn() as conn:
        n = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
        conn.execute("DELETE FROM messages")
    return n


def count() -> int:
    with _conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
