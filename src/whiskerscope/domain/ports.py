from __future__ import annotations

from abc import ABC, abstractmethod

from .models import Detection, Frame


class CameraPort(ABC):
    @abstractmethod
    def read_frame(self) -> Frame | None: ...

    @abstractmethod
    def release(self) -> None: ...


class DetectorPort(ABC):
    @abstractmethod
    def detect_cats(self, frame: Frame) -> list[Detection]: ...


class RecorderPort(ABC):
    @abstractmethod
    def start_clip(self, frame: Frame) -> None: ...

    @abstractmethod
    def add_frame(self, frame: Frame) -> None: ...

    @abstractmethod
    def stop_clip(self) -> str: ...

    @abstractmethod
    def is_recording(self) -> bool: ...


class EventStorePort(ABC):
    @abstractmethod
    def save_detection(self, timestamp: str, cat_count: int, max_confidence: float, session_id: str) -> None: ...

    @abstractmethod
    def get_session_history(self, session_id: str, limit: int = 20) -> list[dict]: ...

    @abstractmethod
    def get_daily_summary(self) -> dict: ...
