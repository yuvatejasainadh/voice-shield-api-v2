"""Unit and API tests for Aurigin voice clone / deepfake file analysis (/analyze & /analysis)."""

from __future__ import annotations

from io import BytesIO
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.database import Base, engine
from app.main import app
from tests.helpers import generate_wav_bytes


@pytest.fixture(autouse=True)
def setup_db():
    Base.metadata.create_all(bind=engine)
    yield


MOCK_AURIGIN_SPOOFED_RESULT = {
    "provider": "aurigin",
    "prediction_id": "pred_spoof_123",
    "result": "SPOOFED",
    "score": 0.94,
    "confidence": 0.98,
    "risk_score": 94,
    "segments": [
        {"segment_start": 0.0, "segment_end": 5.0, "score": 0.95, "result": "spoofed"}
    ],
    "processing_ms": 45,
    "raw_provider_status": "PROCESSED",
    "raw_result": "spoofed",
    "reason": "Synthetic vocoder artifacts detected",
}

MOCK_AURIGIN_AUTHENTIC_RESULT = {
    "provider": "aurigin",
    "prediction_id": "pred_auth_456",
    "result": "REAL",
    "score": 0.08,
    "confidence": 0.96,
    "risk_score": 8,
    "segments": [
        {"segment_start": 0.0, "segment_end": 5.0, "score": 0.08, "result": "bonafide"}
    ],
    "processing_ms": 40,
    "raw_provider_status": "PROCESSED",
    "raw_result": "bonafide",
    "reason": None,
}


def test_analysis_validation_missing_file() -> None:
    with TestClient(app) as client:
        response = client.post("/api/v1/analysis")
        assert response.status_code == 422

        response_alias = client.post("/analyze")
        assert response_alias.status_code == 422


def test_analysis_unsupported_format() -> None:
    with TestClient(app) as client:
        files = {"audio": ("bad.txt", BytesIO(b"not-audio"), "text/plain")}
        response = client.post("/api/v1/analysis", files=files)
        assert response.status_code == 415
        assert response.json()["detail"]["code"] == "UNSUPPORTED_AUDIO_FORMAT"


def test_analysis_aurigin_spoofed() -> None:
    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=MOCK_AURIGIN_SPOOFED_RESULT) as mock_analyze:
        with TestClient(app) as client:
            files = {"audio": ("sample.wav", BytesIO(generate_wav_bytes(duration_seconds=1.0)), "audio/wav")}
            response = client.post("/analyze", files=files)

            assert response.status_code == 201
            payload = response.json()
            assert "analysis_id" in payload and payload["analysis_id"]
            assert payload["status"] == "completed"
            assert payload["classification"] == "LIKELY_AI_GENERATED"
            assert payload["risk_score"] == 94
            assert payload["ai_probability"] == 0.94
            assert payload["confidence"] == 0.98
            assert payload["detector_version"].startswith("aurigin")
            assert len(payload["reasons"]) >= 1

            assert mock_analyze.call_count == 1

            # Check GET /analyze/{id} and /api/v1/analysis/{id}
            detail_resp = client.get(f"/analyze/{payload['analysis_id']}")
            assert detail_resp.status_code == 200
            assert detail_resp.json()["analysis_id"] == payload["analysis_id"]

            detail_resp2 = client.get(f"/api/v1/analysis/{payload['analysis_id']}")
            assert detail_resp2.status_code == 200


def test_analysis_aurigin_authentic() -> None:
    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=MOCK_AURIGIN_AUTHENTIC_RESULT):
        with TestClient(app) as client:
            files = {"audio": ("sample.wav", BytesIO(generate_wav_bytes(duration_seconds=1.0)), "audio/wav")}
            response = client.post("/api/v1/analysis", files=files)

            assert response.status_code == 201
            payload = response.json()
            assert payload["classification"] == "LIKELY_GENUINE"
            assert payload["risk_score"] == 8
            assert payload["ai_probability"] == 0.08
            assert payload["confidence"] == 0.96


def test_analysis_aurigin_not_configured() -> None:
    mock_not_configured = {
        "provider": "aurigin",
        "prediction_id": None,
        "result": "UNKNOWN",
        "confidence": None,
        "risk_score": None,
        "segments": [],
        "processing_ms": 0,
        "raw_provider_status": "NOT_CONFIGURED",
    }
    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_not_configured):
        with TestClient(app) as client:
            files = {"audio": ("sample.wav", BytesIO(generate_wav_bytes(duration_seconds=1.0)), "audio/wav")}
            response = client.post("/analyze", files=files)
            assert response.status_code == 503
            assert response.json()["detail"]["code"] == "AURIGIN_NOT_CONFIGURED"


def test_analysis_aurigin_timeout() -> None:
    mock_timeout = {
        "provider": "aurigin",
        "prediction_id": None,
        "result": "UNKNOWN",
        "confidence": None,
        "risk_score": None,
        "segments": [],
        "processing_ms": 30000,
        "raw_provider_status": "TIMEOUT",
    }
    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_timeout):
        with TestClient(app) as client:
            files = {"audio": ("sample.wav", BytesIO(generate_wav_bytes(duration_seconds=1.0)), "audio/wav")}
            response = client.post("/analyze", files=files)
            assert response.status_code == 504
            assert response.json()["detail"]["code"] == "AURIGIN_TIMEOUT"


def test_analysis_aurigin_error() -> None:
    mock_error = {
        "provider": "aurigin",
        "prediction_id": None,
        "result": "UNKNOWN",
        "confidence": None,
        "risk_score": None,
        "segments": [],
        "processing_ms": 100,
        "raw_provider_status": "ERROR",
        "reason": "Internal provider inference failure",
    }
    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_error):
        with TestClient(app) as client:
            files = {"audio": ("sample.wav", BytesIO(generate_wav_bytes(duration_seconds=1.0)), "audio/wav")}
            response = client.post("/analyze", files=files)
            assert response.status_code == 502
            assert response.json()["detail"]["code"] == "AURIGIN_ERROR"
