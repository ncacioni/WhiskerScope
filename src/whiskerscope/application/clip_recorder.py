from __future__ import annotations

import time

from whiskerscope.domain.models import CatEvent
from whiskerscope.domain.ports import RecorderPort


class ClipRecorderService:
    def __init__(self, recorder: RecorderPort, cooldown_secs: float = 3.0) -> None:
        self.recorder = recorder
        self.cooldown_secs = cooldown_secs
        self._last_detection_time: float = 0.0
        self.saved_clips: list[str] = []

    def update(self, event: CatEvent) -> str | None:
        now = time.monotonic()

        if event.cat_count > 0:
            self._last_detection_time = now
            if not self.recorder.is_recording():
                self.recorder.start_clip(event.frame)
            else:
                self.recorder.add_frame(event.frame)
            return None

        if self.recorder.is_recording():
            if now - self._last_detection_time >= self.cooldown_secs:
                clip_path = self.recorder.stop_clip()
                self.saved_clips.append(clip_path)
                return clip_path
            self.recorder.add_frame(event.frame)

        return None
