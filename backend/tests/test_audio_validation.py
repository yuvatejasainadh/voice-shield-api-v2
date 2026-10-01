from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app
from tests.helpers import generate_audio_format_bytes, generate_wav_bytes


MOCK_AURIGIN_ANALYSIS = {
    "provider": "aurigin",
    "prediction_id": "test-aurigin-analysis-1",
    "result": "REAL",
    "score": 0.10,
    "confidence": 0.95,
    "risk_score": 10,
    "segments": [],
    "processing_ms": 50,
    "raw_provider_status": "PROCESSED",
    "raw_result": "bonafide",
}


@pytest.fixture(autouse=True)
def setup_aurigin_key(monkeypatch):
    monkeypatch.setattr(get_settings(), "aurigin_api_key", "aurigin_test_mock_key")
    monkeypatch.setattr(get_settings(), "aurigin_enabled", True)


@pytest.mark.parametrize("fmt,content_type", [
    ("wav", "audio/wav"),
    ("mp3", "audio/mpeg"),
    ("m4a", "audio/m4a"),
    ("aac", "audio/aac"),
    ("ogg", "audio/ogg"),
    ("flac", "audio/flac"),
    ("webm", "audio/webm"),
])
def test_all_seven_audio_formats_accepted(fmt: str, content_type: str) -> None:
    """Test that all 7 supported formats (.wav, .mp3, .m4a, .aac, .ogg, .flac, .webm) are validated and accepted."""
    audio_bytes = generate_audio_format_bytes(fmt, duration_seconds=1.0)
    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=MOCK_AURIGIN_ANALYSIS):
        with TestClient(app) as client:
            files = {"audio": (f"sample.{fmt}", BytesIO(audio_bytes), content_type)}
            response = client.post("/api/v2/analysis", files=files)
            assert response.status_code == 201, f"Failed for format {fmt}: {response.text}"
            payload = response.json()
            assert payload["status"] == "completed"
            assert payload["classification"] == "LIKELY_GENUINE"


def test_stereo_wav_is_accepted_and_converted_for_inference() -> None:
    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=MOCK_AURIGIN_ANALYSIS):
        with TestClient(app) as client:
            files = {"audio": ("stereo.wav", BytesIO(generate_wav_bytes(channels=2)), "audio/wav")}
            response = client.post("/api/v2/analysis", files=files)
            assert response.status_code == 201
            assert response.json()["status"] == "completed"


def test_invalid_extension() -> None:
    with TestClient(app) as client:
        files = {"audio": ("bad.txt", BytesIO(b"not audio"), "text/plain")}
        response = client.post("/api/v2/analysis", files=files)
        assert response.status_code == 415
        assert response.json()["detail"]["code"] == "UNSUPPORTED_AUDIO_FORMAT"


def test_empty_file() -> None:
    with TestClient(app) as client:
        files = {"audio": ("empty.wav", BytesIO(b""), "audio/wav")}
        response = client.post("/api/v2/analysis", files=files)
        assert response.status_code == 422
        assert response.json()["detail"]["code"] == "EMPTY_AUDIO"


def test_corrupted_audio() -> None:
    with TestClient(app) as client:
        files = {"audio": ("broken.wav", BytesIO(b"RIFF\x00\x00bad"), "audio/wav")}
        response = client.post("/api/v2/analysis", files=files)
        assert response.status_code == 422
        assert response.json()["detail"]["code"] in {"INVALID_AUDIO", "EMPTY_AUDIO"}


def test_audio_duration_limit() -> None:
    settings = get_settings()
    original_limit = settings.max_audio_duration_seconds
    settings.max_audio_duration_seconds = 1
    try:
        with TestClient(app) as client:
            files = {"audio": ("long.wav", BytesIO(generate_wav_bytes(duration_seconds=2.0)), "audio/wav")}
            response = client.post("/api/v2/analysis", files=files)
            assert response.status_code == 413
            assert response.json()["detail"]["code"] == "AUDIO_TOO_LONG"
    finally:
        settings.max_audio_duration_seconds = original_limit


def test_silent_audio_is_accepted() -> None:
    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=MOCK_AURIGIN_ANALYSIS):
        import soundfile as sf
        import numpy as np

        with TestClient(app) as client:
            silence = BytesIO()
            sf.write(silence, np.zeros(16000, dtype=np.float32), 16000, format="WAV")
            silence.seek(0)
            response = client.post(
                "/api/v2/analysis",
                files={"audio": ("silence.wav", silence, "audio/wav")},
            )
            assert response.status_code == 201
            assert response.json()["status"] == "completed"


def test_oversized_file() -> None:
    settings = get_settings()
    oversized = BytesIO(b"0" * (settings.max_upload_size_bytes + 1))
    with TestClient(app) as client:
        files = {"audio": ("big.wav", oversized, "audio/wav")}
        response = client.post("/api/v2/analysis", files=files)
        assert response.status_code == 413
        assert response.json()["detail"]["code"] == "FILE_TOO_LARGE"

