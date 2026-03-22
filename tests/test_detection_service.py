from datetime import datetime, timezone

from whiskerscope.application.detection_service import DetectionService
from whiskerscope.application.event_counter import EventCounter
from whiskerscope.domain.models import BoundingBox, Detection, Frame
from whiskerscope.domain.ports import CameraPort, DetectorPort


class FakeCamera(CameraPort):
    def __init__(self, frames: list[Frame | None]) -> None:
        self._frames = iter(frames)

    def read_frame(self) -> Frame | None:
        return next(self._frames, None)

    def release(self) -> None:
        pass


class FakeDetector(DetectorPort):
    def __init__(self, results: list[list[Detection]]) -> None:
        self._results = iter(results)

    def detect_cats(self, frame: Frame) -> list[Detection]:
        return next(self._results, [])


def _make_frame() -> Frame:
    return Frame(data=None, timestamp=datetime.now(timezone.utc), width=640, height=480)


def _make_detection() -> Detection:
    return Detection(
        label="cat",
        bbox=BoundingBox(10, 20, 100, 200, 0.95),
        timestamp=datetime.now(timezone.utc),
    )


def test_process_frame_with_detection():
    frame = _make_frame()
    det = _make_detection()
    service = DetectionService(FakeCamera([frame]), FakeDetector([[det]]))
    event = service.process_frame()
    assert event is not None
    assert event.cat_count == 1


def test_process_frame_no_frame():
    service = DetectionService(FakeCamera([None]), FakeDetector([]))
    event = service.process_frame()
    assert event is None


def test_process_frame_no_cats():
    frame = _make_frame()
    service = DetectionService(FakeCamera([frame]), FakeDetector([[]]))
    event = service.process_frame()
    assert event is not None
    assert event.cat_count == 0


def test_event_counter_tracks_stats():
    counter = EventCounter()
    counter.update(2)
    counter.update(0)
    counter.update(3)
    stats = counter.stats()
    assert stats["total_detections"] == 5
    assert stats["max_simultaneous"] == 3
    assert stats["session_duration_secs"] >= 0
