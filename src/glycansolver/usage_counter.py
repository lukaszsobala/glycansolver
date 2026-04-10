"""
Lightweight SQLite-backed usage counter for the Glycan Solver web app.

Tracks page visits, candidate searches, and solve jobs with timestamps.
All operations are thread-safe (SQLite serialised mode + per-call connections).
"""

from __future__ import annotations

import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path

_DB_PATH: Path | None = None
_lock = threading.Lock()


def _resolve_default_db_path() -> Path:
    env_path = os.environ.get("GLYCANSOLVER_USAGE_DB")
    if env_path:
        return Path(env_path).expanduser()
    return Path.cwd() / "usage.db"


def init(db_path: str | Path | None = None) -> None:
    """Initialise the counter database. Call once at app startup."""
    global _DB_PATH
    if db_path is None:
        db_path = _resolve_default_db_path()
    _DB_PATH = Path(db_path)
    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    with _connect() as con:
        con.execute(
            """
            CREATE TABLE IF NOT EXISTS events (
                id    INTEGER PRIMARY KEY AUTOINCREMENT,
                kind  TEXT    NOT NULL,
                ts    TEXT    NOT NULL
            )
            """
        )
        con.commit()


def _connect() -> sqlite3.Connection:
    if _DB_PATH is None:
        raise RuntimeError("usage_counter.init() has not been called")
    return sqlite3.connect(str(_DB_PATH), timeout=5)


def record(kind: str) -> None:
    """Record a single event of the given *kind* (e.g. 'visit', 'solve')."""
    ts = datetime.now(timezone.utc).isoformat()
    with _lock, _connect() as con:
        con.execute("INSERT INTO events (kind, ts) VALUES (?, ?)", (kind, ts))
        con.commit()


def stats() -> dict:
    """Return aggregate counts and recent activity."""
    with _lock, _connect() as con:
        cur = con.execute(
            "SELECT kind, COUNT(*) FROM events GROUP BY kind ORDER BY kind"
        )
        totals = {row[0]: row[1] for row in cur.fetchall()}

        cur = con.execute(
            "SELECT kind, ts FROM events ORDER BY id DESC LIMIT 20"
        )
        recent = [{"kind": r[0], "ts": r[1]} for r in cur.fetchall()]

    return {"totals": totals, "recent": recent}
