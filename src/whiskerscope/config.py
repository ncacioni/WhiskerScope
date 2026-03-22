from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field


@dataclass
class WhiskerscopeConfig:
    camera_index: int = 0
    model_name: str = "yolov8n.pt"
    confidence: float = 0.5
    clip_cooldown_secs: float = 3.0
    clip_output_dir: str = "clips"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    log_level: str = "INFO"
    db_path: str = "whiskerscope.db"
    detector_backend: str = "yolo"
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])

    @classmethod
    def from_env(cls) -> WhiskerscopeConfig:
        return cls(
            camera_index=int(os.environ.get("WS_CAMERA_INDEX", "0")),
            model_name=os.environ.get("WS_MODEL_NAME", "yolov8n.pt"),
            confidence=float(os.environ.get("WS_CONFIDENCE", "0.5")),
            clip_cooldown_secs=float(os.environ.get("WS_CLIP_COOLDOWN", "3.0")),
            clip_output_dir=os.environ.get("WS_CLIP_DIR", "clips"),
            api_host=os.environ.get("WS_API_HOST", "0.0.0.0"),
            api_port=int(os.environ.get("WS_API_PORT", "8000")),
            log_level=os.environ.get("WS_LOG_LEVEL", "INFO"),
            db_path=os.environ.get("WS_DB_PATH", "whiskerscope.db"),
            detector_backend=os.environ.get("WS_DETECTOR_BACKEND", "yolo"),
        )
