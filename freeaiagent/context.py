import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Dict

DB_FILE = Path.home() / ".freeaiagent" / "context.db"


def _conn() -> sqlite3.Connection:
    DB_FILE.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_FILE))
    conn.row_factory = sqlite3.Row
    _ensure_schema(conn)
    return conn


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL DEFAULT 'default',
            role       TEXT NOT NULL,
            content    TEXT NOT NULL,
            timestamp  TEXT NOT NULL
        )
    """)
    # Migrate: existing installs have no session_id column yet
    cols = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
    if "session_id" not in cols:
        conn.execute("ALTER TABLE messages ADD COLUMN session_id TEXT NOT NULL DEFAULT 'default'")

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id            TEXT PRIMARY KEY,
            title         TEXT NOT NULL DEFAULT '',
            model         TEXT NOT NULL DEFAULT '',
            backend       TEXT NOT NULL DEFAULT '',
            created_at    TEXT NOT NULL,
            last_updated  TEXT NOT NULL,
            message_count INTEGER NOT NULL DEFAULT 0
        )
    """)
    conn.commit()


# ── Messages ─────────────────────────────────────────────────────────────────

def append(role: str, content: str, session_id: str = "default", model: str = "", backend: str = "") -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT INTO messages (session_id, role, content, timestamp) VALUES (?, ?, ?, ?)",
            (session_id, role, content, ts),
        )
        existing = conn.execute("SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
        if existing is None:
            title = content[:60] if role == "user" else ""
            conn.execute(
                "INSERT INTO sessions (id, title, model, backend, created_at, last_updated, message_count)"
                " VALUES (?, ?, ?, ?, ?, ?, 1)",
                (session_id, title, model, backend, ts, ts),
            )
        else:
            parts = ["last_updated = ?", "message_count = message_count + 1"]
            params: list = [ts]
            if not existing["title"] and role == "user":
                parts.append("title = ?")
                params.append(content[:60])
            if model:
                parts.append("model = ?")
                params.append(model)
            if backend:
                parts.append("backend = ?")
                params.append(backend)
            params.append(session_id)
            conn.execute(f"UPDATE sessions SET {', '.join(parts)} WHERE id = ?", params)
        conn.commit()


def all_messages(session_id: str = "default") -> List[Dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT role, content, timestamp FROM messages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def as_llm_messages(session_id: str = "default", max_messages: int = 0) -> List[Dict]:
    msgs = [{"role": m["role"], "content": m["content"]} for m in all_messages(session_id=session_id)]
    if max_messages > 0 and len(msgs) > max_messages:
        msgs = msgs[-max_messages:]
    return msgs


def clear(session_id: str = "default") -> int:
    with _conn() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        conn.execute("UPDATE sessions SET message_count = 0 WHERE id = ?", (session_id,))
        conn.commit()
    return n


def count(session_id: str = "default") -> int:
    with _conn() as conn:
        return conn.execute(
            "SELECT COUNT(*) FROM messages WHERE session_id = ?", (session_id,)
        ).fetchone()[0]


# ── Sessions ─────────────────────────────────────────────────────────────────

def all_sessions() -> List[Dict]:
    with _conn() as conn:
        rows = conn.execute(
            "SELECT id, title, model, backend, created_at, last_updated, message_count"
            " FROM sessions ORDER BY last_updated DESC"
        ).fetchall()
    return [
        {
            "id": r["id"],
            "title": r["title"] or "New Chat",
            "model": r["model"],
            "backend": r["backend"],
            "created_at": r["created_at"],
            "last_updated": r["last_updated"],
            "message_count": r["message_count"],
        }
        for r in rows
    ]


def create_session(session_id: str, title: str = "") -> Dict:
    ts = datetime.now(timezone.utc).isoformat()
    with _conn() as conn:
        conn.execute(
            "INSERT OR IGNORE INTO sessions (id, title, model, backend, created_at, last_updated, message_count)"
            " VALUES (?, ?, '', '', ?, ?, 0)",
            (session_id, title, ts, ts),
        )
        conn.commit()
        row = conn.execute("SELECT * FROM sessions WHERE id = ?", (session_id,)).fetchone()
    return dict(row)


def rename_session(session_id: str, title: str) -> bool:
    with _conn() as conn:
        cur = conn.execute("UPDATE sessions SET title = ? WHERE id = ?", (title, session_id))
        conn.commit()
    return cur.rowcount > 0


def delete_session(session_id: str) -> bool:
    with _conn() as conn:
        conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
        cur = conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
        conn.commit()
    return cur.rowcount > 0
