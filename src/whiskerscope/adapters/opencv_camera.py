from __future__ import annotations

from datetime import datetime, timezone

import cv2

from whiskerscope.domain.models import Frame
from whiskerscope.domain.ports import CameraPort


class OpenCVCamera(CameraPort):
    def __init__(self, camera_index: int = 0) -> None:
        self.cap = cv2.VideoCapture(camera_index)

    def read_frame(self) -> Frame | None:
        ret, data = self.cap.read()
        if not ret:
            return None
        h, w = data.shape[:2]
        return Frame(data=data, timestamp=datetime.now(timezone.utc), width=w, height=h)

    def release(self) -> None:
        self.cap.release()
