from __future__ import annotations

import sqlite3

from whiskerscope.domain.ports import EventStorePort


class SQLiteStore(EventStorePort):
    def __init__(self, db_path: str = "whiskerscope.db") -> None:
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                cat_count INTEGER NOT NULL,
                max_confidence REAL NOT NULL,
                session_id TEXT NOT NULL
            )"""
        )
        self.conn.commit()

    def save_detection(self, timestamp: str, cat_count: int, max_confidence: float, session_id: str) -> None:
        self.conn.execute(
            "INSERT INTO detections (timestamp, cat_count, max_confidence, session_id) VALUES (?, ?, ?, ?)",
            (timestamp, cat_count, max_confidence, session_id),
        )
        self.conn.commit()

    def get_session_history(self, session_id: str, limit: int = 20) -> list[dict]:
        rows = self.conn.execute(
            "SELECT timestamp, cat_count, max_confidence FROM detections WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_daily_summary(self) -> dict:
        row = self.conn.execute(
            "SELECT COUNT(*) as events, COALESCE(SUM(cat_count), 0) as total_cats, COALESCE(MAX(max_confidence), 0) as best_confidence FROM detections WHERE date(timestamp) = date('now')"
        ).fetchone()
        return dict(row) if row else {"events": 0, "total_cats": 0, "best_confidence": 0}
