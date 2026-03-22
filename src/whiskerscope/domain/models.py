from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass(frozen=True)
class BoundingBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1


@dataclass(frozen=True)
class Detection:
    label: str
    bbox: BoundingBox
    timestamp: datetime


@dataclass
class Frame:
    data: Any  # numpy array at runtime — no numpy import in domain
    timestamp: datetime
    width: int
    height: int


@dataclass
class CatEvent:
    detections: list[Detection]
    frame: Frame
    cat_count: int = field(init=False)

    def __post_init__(self) -> None:
        self.cat_count = len(self.detections)
