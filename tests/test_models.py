from datetime import datetime, timezone

from whiskerscope.domain.models import BoundingBox, CatEvent, Detection, Frame


def test_bounding_box_dimensions():
    bbox = BoundingBox(x1=10.0, y1=20.0, x2=110.0, y2=80.0, confidence=0.9)
    assert bbox.width == 100.0
    assert bbox.height == 60.0


def test_bounding_box_is_frozen():
    bbox = BoundingBox(0, 0, 1, 1, 0.5)
    try:
        bbox.x1 = 99  # type: ignore[misc]
        assert False, "Should have raised"
    except AttributeError:
        pass


def test_cat_event_auto_count():
    now = datetime.now(timezone.utc)
    frame = Frame(data=None, timestamp=now, width=640, height=480)
    dets = [
        Detection("cat", BoundingBox(0, 0, 1, 1, 0.9), now),
        Detection("cat", BoundingBox(2, 2, 3, 3, 0.8), now),
    ]
    event = CatEvent(detections=dets, frame=frame)
    assert event.cat_count == 2


def test_cat_event_empty():
    now = datetime.now(timezone.utc)
    frame = Frame(data=None, timestamp=now, width=640, height=480)
    event = CatEvent(detections=[], frame=frame)
    assert event.cat_count == 0
