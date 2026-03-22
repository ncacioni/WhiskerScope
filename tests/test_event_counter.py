from whiskerscope.application.event_counter import EventCounter
from whiskerscope.domain.ports import EventStorePort


class FakeStore(EventStorePort):
    def __init__(self) -> None:
        self.saved: list[dict] = []

    def save_detection(self, timestamp: str, cat_count: int, max_confidence: float, session_id: str) -> None:
        self.saved.append({"timestamp": timestamp, "cat_count": cat_count, "max_confidence": max_confidence})

    def get_session_history(self, session_id: str, limit: int = 20) -> list[dict]:
        return self.saved[:limit]

    def get_daily_summary(self) -> dict:
        return {"events": len(self.saved), "total_cats": sum(d["cat_count"] for d in self.saved)}


def test_counter_tracks_stats():
    counter = EventCounter()
    counter.update(2)
    counter.update(0)
    counter.update(3)
    stats = counter.stats()
    assert stats["total_detections"] == 5
    assert stats["max_simultaneous"] == 3


def test_counter_with_store():
    store = FakeStore()
    counter = EventCounter(store=store, session_id="test")
    counter.update(2, max_confidence=0.9)
    counter.update(1, max_confidence=0.7)
    assert len(store.saved) == 2
    assert store.saved[0]["cat_count"] == 2


def test_session_history_with_store():
    store = FakeStore()
    counter = EventCounter(store=store, session_id="test")
    counter.update(1, max_confidence=0.8)
    history = counter.session_history()
    assert len(history) == 1


def test_session_history_without_store():
    counter = EventCounter()
    assert counter.session_history() == []


def test_daily_summary_without_store():
    counter = EventCounter()
    assert counter.daily_summary() == {}
