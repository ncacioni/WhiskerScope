from datetime import datetime, timezone

from whiskerscope.application.clip_recorder import ClipRecorderService
from whiskerscope.domain.models import BoundingBox, CatEvent, Detection, Frame
from whiskerscope.domain.ports import RecorderPort


class FakeRecorder(RecorderPort):
    def __init__(self) -> None:
        self._recording = False
        self.frames_written: int = 0
        self.clips_saved: list[str] = []

    def start_clip(self, frame: Frame) -> None:
        self._recording = True
        self.frames_written = 1

    def add_frame(self, frame: Frame) -> None:
        self.frames_written += 1

    def stop_clip(self) -> str:
        self._recording = False
        path = f"clip_{len(self.clips_saved)}.avi"
        self.clips_saved.append(path)
        return path

    def is_recording(self) -> bool:
        return self._recording


def _make_event(cat_count: int) -> CatEvent:
    now = datetime.now(timezone.utc)
    frame = Frame(data=None, timestamp=now, width=640, height=480)
    dets = [
        Detection("cat", BoundingBox(0, 0, 1, 1, 0.9), now)
        for _ in range(cat_count)
    ]
    return CatEvent(detections=dets, frame=frame)


def test_starts_recording_on_detection():
    rec = FakeRecorder()
    svc = ClipRecorderService(rec, cooldown_secs=1.0)
    svc.update(_make_event(1))
    assert rec.is_recording()


def test_no_recording_without_cats():
    rec = FakeRecorder()
    svc = ClipRecorderService(rec, cooldown_secs=1.0)
    svc.update(_make_event(0))
    assert not rec.is_recording()


def test_stops_after_cooldown(monkeypatch):
    rec = FakeRecorder()
    svc = ClipRecorderService(rec, cooldown_secs=0.0)
    svc.update(_make_event(1))
    assert rec.is_recording()
    # Simulate cooldown expired
    svc._last_detection_time = 0.0
    clip = svc.update(_make_event(0))
    assert clip is not None
    assert not rec.is_recording()
