from __future__ import annotations

import logging
import os
from datetime import datetime, timezone

import cv2

from whiskerscope.domain.models import Frame
from whiskerscope.domain.ports import RecorderPort

logger = logging.getLogger(__name__)


class OpenCVRecorder(RecorderPort):
    def __init__(self, output_dir: str = "clips", fps: float = 20.0) -> None:
        self.output_dir = output_dir
        self.fps = fps
        self._writer: cv2.VideoWriter | None = None
        self._current_path: str = ""
        os.makedirs(output_dir, exist_ok=True)

    def start_clip(self, frame: Frame) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._current_path = os.path.join(self.output_dir, f"cat_{ts}.avi")
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self._writer = cv2.VideoWriter(
            self._current_path, fourcc, self.fps, (frame.width, frame.height)
        )
        self._writer.write(frame.data)
        logger.info("Started clip: %s", self._current_path)

    def add_frame(self, frame: Frame) -> None:
        if self._writer is not None:
            try:
                self._writer.write(frame.data)
            except Exception:
                logger.exception("Failed to write frame to clip")

    def stop_clip(self) -> str:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
            logger.info("Saved clip: %s", self._current_path)
        return self._current_path

    def is_recording(self) -> bool:
        return self._writer is not None

    def __enter__(self) -> OpenCVRecorder:
        return self

    def __exit__(self, *exc: object) -> None:
        if self.is_recording():
            self.stop_clip()
