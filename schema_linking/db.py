"""Minimal SQLite database wrapper.

Extracted from InsightXpert (src/insightxpert/db.py), simplified to accept a
direct file path instead of relying on project-wide settings.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


class Database:
    """Read-only SQLite connection backed by a single .sqlite file."""

    def __init__(self, db_path: Path | str) -> None:
        db_path = Path(db_path)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file not found: {db_path}")
        self.db_id = db_path.stem
        self._conn = sqlite3.connect(
            f"file:{db_path}?mode=ro",
            uri=True,
            check_same_thread=False,
        )

    def execute(self, sql: str, params: tuple = ()) -> list[tuple]:
        cur = self._conn.cursor()
        cur.execute(sql, params)
        return cur.fetchall()

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Database:
        return self

    def __exit__(self, *_) -> None:
        self.close()
