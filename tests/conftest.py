from __future__ import annotations

from datetime import datetime, timezone

import pytest

from whiskerscope.config import WhiskerscopeConfig
from whiskerscope.domain.models import BoundingBox, Detection, Frame
from whiskerscope.domain.ports import CameraPort, DetectorPort


class FakeCamera(CameraPort):
    def __init__(self, frames: list[Frame | None]) -> None:
        self._frames = iter(frames)

    def read_frame(self) -> Frame | None:
        return next(self._frames, None)

    def release(self) -> None:
        pass


class FakeDetector(DetectorPort):
    def __init__(self, results: list[list[Detection]]) -> None:
        self._results = iter(results)

    def detect_cats(self, frame: Frame) -> list[Detection]:
        return next(self._results, [])


@pytest.fixture
def sample_frame() -> Frame:
    return Frame(data=None, timestamp=datetime.now(timezone.utc), width=640, height=480)


@pytest.fixture
def sample_detection() -> Detection:
    return Detection(
        label="cat",
        bbox=BoundingBox(10, 20, 100, 200, 0.95),
        timestamp=datetime.now(timezone.utc),
    )


@pytest.fixture
def test_config(tmp_path) -> WhiskerscopeConfig:
    return WhiskerscopeConfig(db_path=str(tmp_path / "test.db"))
