from __future__ import annotations

import time


class EventCounter:
    def __init__(self) -> None:
        self.total_detections: int = 0
        self.max_simultaneous: int = 0
        self._start_time: float = time.monotonic()

    def update(self, cat_count: int) -> None:
        if cat_count > 0:
            self.total_detections += cat_count
            self.max_simultaneous = max(self.max_simultaneous, cat_count)

    @property
    def session_duration_secs(self) -> float:
        return time.monotonic() - self._start_time

    def stats(self) -> dict[str, int | float]:
        return {
            "total_detections": self.total_detections,
            "max_simultaneous": self.max_simultaneous,
            "session_duration_secs": round(self.session_duration_secs, 1),
        }
