from __future__ import annotations

from whiskerscope.adapters.opencv_camera import OpenCVCamera
from whiskerscope.adapters.opencv_recorder import OpenCVRecorder
from whiskerscope.adapters.yolo_detector import YOLOv8Detector
from whiskerscope.application.clip_recorder import ClipRecorderService
from whiskerscope.application.detection_service import DetectionService
from whiskerscope.application.event_counter import EventCounter


def create_detection_service(
    camera_index: int = 0,
    model_name: str = "yolov8n.pt",
    confidence: float = 0.5,
) -> DetectionService:
    camera = OpenCVCamera(camera_index)
    detector = YOLOv8Detector(model_name, confidence)
    return DetectionService(camera, detector)


def create_clip_recorder(
    output_dir: str = "clips",
    cooldown_secs: float = 3.0,
) -> ClipRecorderService:
    recorder = OpenCVRecorder(output_dir)
    return ClipRecorderService(recorder, cooldown_secs)


def create_event_counter() -> EventCounter:
    return EventCounter()
