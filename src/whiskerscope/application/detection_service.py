from __future__ import annotations

import logging

from whiskerscope.domain.models import CatEvent
from whiskerscope.domain.ports import CameraPort, DetectorPort

logger = logging.getLogger(__name__)


class DetectionService:
    def __init__(self, camera: CameraPort, detector: DetectorPort) -> None:
        self.camera = camera
        self.detector = detector

    def process_frame(self) -> CatEvent | None:
        frame = self.camera.read_frame()
        if frame is None:
            return None
        try:
            detections = self.detector.detect_cats(frame)
        except Exception:
            logger.exception("Detection failed for frame")
            detections = []
        if detections:
            logger.info("Detected %d cat(s)", len(detections))
        return CatEvent(detections=detections, frame=frame)
