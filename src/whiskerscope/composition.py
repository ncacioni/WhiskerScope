from __future__ import annotations

from whiskerscope.adapters.opencv_camera import OpenCVCamera
from whiskerscope.adapters.opencv_recorder import OpenCVRecorder
from whiskerscope.adapters.sqlite_store import SQLiteStore
from whiskerscope.adapters.yolo_detector import YOLOv8Detector
from whiskerscope.application.clip_recorder import ClipRecorderService
from whiskerscope.application.detection_service import DetectionService
from whiskerscope.application.event_counter import EventCounter
from whiskerscope.config import WhiskerscopeConfig
from whiskerscope.logging_setup import setup_logging

_initialized = False


def init_app(config: WhiskerscopeConfig | None = None) -> WhiskerscopeConfig:
    global _initialized
    if config is None:
        config = WhiskerscopeConfig.from_env()
    if not _initialized:
        setup_logging(config)
        _initialized = True
    return config


def create_detection_service(config: WhiskerscopeConfig | None = None) -> DetectionService:
    config = init_app(config)
    camera = OpenCVCamera(config.camera_index)
    if config.detector_backend == "onnx":
        from whiskerscope.adapters.onnx_detector import ONNXDetector
        detector = ONNXDetector(config.model_name.replace(".pt", ".onnx"), config.confidence)
    else:
        detector = YOLOv8Detector(config.model_name, config.confidence)
    return DetectionService(camera, detector)


def create_clip_recorder(config: WhiskerscopeConfig | None = None) -> ClipRecorderService:
    config = init_app(config)
    recorder = OpenCVRecorder(config.clip_output_dir)
    return ClipRecorderService(recorder, config.clip_cooldown_secs)


def create_event_counter(config: WhiskerscopeConfig | None = None) -> EventCounter:
    config = init_app(config)
    store = SQLiteStore(config.db_path)
    return EventCounter(store=store, session_id=config.session_id)
