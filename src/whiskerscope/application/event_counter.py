from __future__ import annotations

import time
from datetime import datetime, timezone

from whiskerscope.domain.ports import EventStorePort


class EventCounter:
    def __init__(self, store: EventStorePort | None = None, session_id: str = "") -> None:
        self.total_detections: int = 0
        self.max_simultaneous: int = 0
        self._start_time: float = time.monotonic()
        self._store = store
        self._session_id = session_id

    def update(self, cat_count: int, max_confidence: float = 0.0) -> None:
        if cat_count > 0:
            self.total_detections += cat_count
            self.max_simultaneous = max(self.max_simultaneous, cat_count)
            if self._store:
                self._store.save_detection(
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    cat_count=cat_count,
                    max_confidence=max_confidence,
                    session_id=self._session_id,
                )

    @property
    def session_duration_secs(self) -> float:
        return time.monotonic() - self._start_time

    def session_history(self, limit: int = 20) -> list[dict]:
        if self._store:
            return self._store.get_session_history(self._session_id, limit)
        return []

    def daily_summary(self) -> dict:
        if self._store:
            return self._store.get_daily_summary()
        return {}

    def stats(self) -> dict[str, int | float]:
        return {
            "total_detections": self.total_detections,
            "max_simultaneous": self.max_simultaneous,
            "session_duration_secs": round(self.session_duration_secs, 1),
        }
