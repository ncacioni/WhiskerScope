from __future__ import annotations

import logging
import time

import cv2
import numpy as np
import onnxruntime as ort

from whiskerscope.domain.models import BoundingBox, Detection, Frame
from whiskerscope.domain.ports import DetectorPort

CAT_CLASS_ID = 15
logger = logging.getLogger(__name__)


class ONNXDetector(DetectorPort):
    def __init__(self, model_path: str = "yolov8n.onnx", confidence: float = 0.5) -> None:
        t0 = time.monotonic()
        self.session = ort.InferenceSession(model_path)
        self.confidence = confidence
        self.input_name = self.session.get_inputs()[0].name
        logger.info("ONNX model %s loaded in %.1fms", model_path, (time.monotonic() - t0) * 1000)

    def _preprocess(self, frame: Frame) -> np.ndarray:
        img = cv2.resize(frame.data, (640, 640))
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return np.expand_dims(img, axis=0)

    def detect_cats(self, frame: Frame) -> list[Detection]:
        try:
            input_tensor = self._preprocess(frame)
            outputs = self.session.run(None, {self.input_name: input_tensor})
        except Exception:
            logger.exception("ONNX inference failed")
            return []

        # YOLOv8 ONNX output: (1, 84, 8400) -> transpose to (8400, 84)
        output = outputs[0][0].T
        detections: list[Detection] = []
        scale_x = frame.width / 640
        scale_y = frame.height / 640

        for row in output:
            class_scores = row[4:]
            class_id = int(np.argmax(class_scores))
            conf = float(class_scores[class_id])
            if class_id != CAT_CLASS_ID or conf < self.confidence:
                continue
            cx, cy, w, h = row[:4]
            x1 = (cx - w / 2) * scale_x
            y1 = (cy - h / 2) * scale_y
            x2 = (cx + w / 2) * scale_x
            y2 = (cy + h / 2) * scale_y
            detections.append(
                Detection(
                    label="cat",
                    bbox=BoundingBox(float(x1), float(y1), float(x2), float(y2), conf),
                    timestamp=frame.timestamp,
                )
            )
        return detections
