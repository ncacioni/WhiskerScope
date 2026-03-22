from __future__ import annotations

import logging
import time

from ultralytics import YOLO

from whiskerscope.domain.models import BoundingBox, Detection, Frame
from whiskerscope.domain.ports import DetectorPort

CAT_CLASS_ID = 15  # COCO class ID for "cat"
logger = logging.getLogger(__name__)


class YOLOv8Detector(DetectorPort):
    def __init__(self, model_name: str = "yolov8n.pt", confidence: float = 0.5) -> None:
        t0 = time.monotonic()
        self.model = YOLO(model_name)
        self.confidence = confidence
        logger.info("Model %s loaded in %.1fms", model_name, (time.monotonic() - t0) * 1000)

    def detect_cats(self, frame: Frame) -> list[Detection]:
        try:
            results = self.model(frame.data, conf=self.confidence, verbose=False)
        except Exception:
            logger.exception("Model inference failed")
            return []
        detections: list[Detection] = []
        for result in results:
            for box in result.boxes:
                if int(box.cls[0]) == CAT_CLASS_ID:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    detections.append(
                        Detection(
                            label="cat",
                            bbox=BoundingBox(x1, y1, x2, y2, float(box.conf[0])),
                            timestamp=frame.timestamp,
                        )
                    )
        return detections
