from __future__ import annotations

import os
from datetime import datetime, timezone

import cv2

from whiskerscope.domain.models import Frame
from whiskerscope.domain.ports import RecorderPort


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

    def add_frame(self, frame: Frame) -> None:
        if self._writer is not None:
            self._writer.write(frame.data)

    def stop_clip(self) -> str:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        return self._current_path

    def is_recording(self) -> bool:
        return self._writer is not None
