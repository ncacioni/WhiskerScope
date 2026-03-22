from __future__ import annotations

from whiskerscope.domain.models import CatEvent
from whiskerscope.domain.ports import CameraPort, DetectorPort


class DetectionService:
    def __init__(self, camera: CameraPort, detector: DetectorPort) -> None:
        self.camera = camera
        self.detector = detector

    def process_frame(self) -> CatEvent | None:
        frame = self.camera.read_frame()
        if frame is None:
            return None
        detections = self.detector.detect_cats(frame)
        return CatEvent(detections=detections, frame=frame)
