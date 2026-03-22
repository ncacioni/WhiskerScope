from whiskerscope.application.detection_service import DetectionService
from whiskerscope.domain.models import Frame, Detection
from tests.conftest import FakeCamera, FakeDetector


def test_process_frame_with_detection(sample_frame, sample_detection):
    service = DetectionService(FakeCamera([sample_frame]), FakeDetector([[sample_detection]]))
    event = service.process_frame()
    assert event is not None
    assert event.cat_count == 1


def test_process_frame_no_frame():
    service = DetectionService(FakeCamera([None]), FakeDetector([]))
    event = service.process_frame()
    assert event is None


def test_process_frame_no_cats(sample_frame):
    service = DetectionService(FakeCamera([sample_frame]), FakeDetector([[]]))
    event = service.process_frame()
    assert event is not None
    assert event.cat_count == 0
