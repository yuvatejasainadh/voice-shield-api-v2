import os

os.environ.setdefault("PRELOAD_MODEL_ON_STARTUP", "false")

from fastapi.testclient import TestClient

from app.main import app


def test_root_endpoint() -> None:
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        payload = response.json()
        assert payload["name"] == "VoiceShield API"
        assert payload["version"] == "V2"
        assert payload["status"] == "ok"


def test_health_endpoint_v2() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v2/health")
        assert response.status_code == 200
        payload = response.json()
        assert payload["status"] == "ok"
        assert payload["version"] == "v2"
        assert payload["service"] == "voiceshield-api"


def test_ready_endpoint_v2() -> None:
    with TestClient(app) as client:
        response = client.get("/api/v2/ready")
        assert response.status_code in {200, 503}
        payload = response.json()
        assert payload["status"] in {"ready", "not_ready"}
        assert payload["version"] == "v2"
        assert payload["database"] in {"connected", "disconnected"}
        assert "providers" in payload
        assert "aurigin" in payload["providers"]
        assert "reality_defender" in payload["providers"]
        assert payload["primary_realtime_detector"] == "aurigin"
        assert payload["file_analysis_detector"] == "aurigin"


def test_v1_routes_not_registered() -> None:
    with TestClient(app) as client:
        assert client.get("/api/v1/health").status_code == 404
        assert client.get("/api/v1/ready").status_code == 404
        assert client.get("/api/v1/history").status_code == 404
        assert client.get("/api/v1/reports").status_code == 404
