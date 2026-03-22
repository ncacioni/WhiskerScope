from unittest.mock import MagicMock

import pytest
from httpx import ASGITransport, AsyncClient

from whiskerscope.adapters.fastapi_app import app
from whiskerscope.adapters.sqlite_store import SQLiteStore
from whiskerscope.application.event_counter import EventCounter
from whiskerscope.config import WhiskerscopeConfig


@pytest.fixture(autouse=True)
def setup_app_state():
    """Ensure app.state is populated for tests (bypassing lifespan)."""
    if not hasattr(app.state, "counter"):
        config = WhiskerscopeConfig()
        store = SQLiteStore(":memory:")
        app.state.config = config
        app.state.store = store
        app.state.counter = EventCounter(store=store, session_id=config.session_id)
        # Mock detector to avoid loading YOLO model in tests
        mock_det = MagicMock()
        mock_det.detect_cats.return_value = []
        app.state.detector = mock_det
    yield


@pytest.mark.anyio
async def test_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "X-Request-ID" in resp.headers


@pytest.mark.anyio
async def test_detect_invalid_type():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/v1/detect", files={"file": ("test.txt", b"not an image", "text/plain")})
    assert resp.status_code == 422
    assert resp.json()["code"] == "INVALID_FILE_TYPE"


@pytest.mark.anyio
async def test_stats():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/v1/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_detections" in data
    assert "daily_summary" in data


@pytest.mark.anyio
async def test_request_id_header():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert "X-Request-ID" in resp.headers
    assert "X-Response-Time-Ms" in resp.headers
