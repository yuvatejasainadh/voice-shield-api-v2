import os

os.environ.setdefault("PRELOAD_MODEL_ON_STARTUP", "false")

from fastapi.testclient import TestClient

from app.main import app


def test_health_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "healthy"
        assert payload["service"] == "voice-clone-detection-api"
        assert payload["version"] == "0.1.0"


def test_ready_endpoint_shape() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v1/ready")
        assert response.status_code in {200, 503}
        payload = response.json()
        assert payload["status"] in {"ready", "not_ready"}
        assert isinstance(payload["database"], bool)
        assert "providers" in payload
        assert "aurigin" in payload["providers"]
        assert "reality_defender" in payload["providers"]
        assert payload["primary_realtime_detector"] == "aurigin"
        assert payload["file_analysis_detector"] == "aurigin"

