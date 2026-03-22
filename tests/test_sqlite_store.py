from whiskerscope.adapters.sqlite_store import SQLiteStore


def test_save_and_retrieve():
    store = SQLiteStore(":memory:")
    store.save_detection("2026-03-22T10:00:00", 2, 0.95, "sess1")
    store.save_detection("2026-03-22T10:01:00", 1, 0.80, "sess1")
    history = store.get_session_history("sess1")
    assert len(history) == 2
    assert history[0]["cat_count"] == 1  # most recent first


def test_session_history_limit():
    store = SQLiteStore(":memory:")
    for i in range(10):
        store.save_detection(f"2026-03-22T10:{i:02d}:00", 1, 0.5, "sess1")
    history = store.get_session_history("sess1", limit=3)
    assert len(history) == 3


def test_session_history_filters_by_session():
    store = SQLiteStore(":memory:")
    store.save_detection("2026-03-22T10:00:00", 2, 0.9, "sess1")
    store.save_detection("2026-03-22T10:01:00", 1, 0.8, "sess2")
    assert len(store.get_session_history("sess1")) == 1
    assert len(store.get_session_history("sess2")) == 1


def test_empty_history():
    store = SQLiteStore(":memory:")
    assert store.get_session_history("nonexistent") == []


def test_daily_summary():
    store = SQLiteStore(":memory:")
    store.save_detection("2026-03-22T10:00:00", 2, 0.95, "sess1")
    store.save_detection("2026-03-22T11:00:00", 3, 0.80, "sess1")
    summary = store.get_daily_summary()
    assert summary["events"] == 2
    assert summary["total_cats"] == 5
