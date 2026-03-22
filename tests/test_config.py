
from whiskerscope.config import WhiskerscopeConfig


def test_default_config():
    config = WhiskerscopeConfig()
    assert config.camera_index == 0
    assert config.model_name == "yolov8n.pt"
    assert config.confidence == 0.5
    assert config.log_level == "INFO"
    assert config.detector_backend == "yolo"


def test_from_env(monkeypatch):
    monkeypatch.setenv("WS_CONFIDENCE", "0.8")
    monkeypatch.setenv("WS_MODEL_NAME", "yolov8s.pt")
    monkeypatch.setenv("WS_API_PORT", "9000")
    config = WhiskerscopeConfig.from_env()
    assert config.confidence == 0.8
    assert config.model_name == "yolov8s.pt"
    assert config.api_port == 9000


def test_session_id_generated():
    c1 = WhiskerscopeConfig()
    c2 = WhiskerscopeConfig()
    assert c1.session_id != c2.session_id
    assert len(c1.session_id) == 8
