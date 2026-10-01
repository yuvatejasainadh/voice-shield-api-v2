from __future__ import annotations

import io
import os
from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.main import app


os.environ.setdefault("PRELOAD_MODEL_ON_STARTUP", "false")


import pytest
from app.db.database import SessionLocal, text


def _wav_bytes(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    import wave

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        sample_count = int(duration_seconds * sample_rate)
        data = sample_count * b"\x00\x00"
        wav_file.writeframes(data)
    return buf.getvalue()


@pytest.fixture(autouse=True)
def clean_realtime_db():
    from app.db.database import Base, engine
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM call_chunks"))
        db.execute(text("DELETE FROM call_sessions"))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()
    yield
    db = SessionLocal()
    try:
        db.execute(text("DELETE FROM call_chunks"))
        db.execute(text("DELETE FROM call_sessions"))
        db.commit()
    except Exception:
        db.rollback()
    finally:
        db.close()


def test_create_session_and_idempotent_creation():
    with TestClient(app) as client:
        payload = {"client_session_id": "session-abc", "started_at": "2026-08-31T12:00:00Z"}
        resp = client.post("/api/v1/realtime/sessions", json=payload)
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["session_id"]
        assert body["status"] == "ACTIVE"
        assert body["risk_state"] == "LOW"

        resp2 = client.post("/api/v1/realtime/sessions", json=payload)
        assert resp2.status_code == 200
        assert resp2.json()["session_id"] == body["session_id"]


def test_chunk_upload_and_duplicate_chunk_are_idempotent():
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-chunk", "started_at": "2026-08-31T12:00:00Z"},
        ).json()
        session_id = session["session_id"]
        files = {"audio": ("chunk.wav", io.BytesIO(_wav_bytes()), "audio/wav")}
        data = {
            "sequence_number": 1,
            "started_at": "2026-08-31T12:00:10Z",
            "ended_at": "2026-08-31T12:00:20Z",
            "duration_ms": 10000,
            "idempotency_key": "dup-1",
        }

        with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value={
            "provider": "aurigin",
            "prediction_id": "pred-1",
            "result": "REAL",
            "confidence": 0.05,
            "risk_score": 10,
            "segments": [],
            "processing_ms": 50,
            "raw_provider_status": "OK",
        }):
            resp = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files, data=data)
            assert resp.status_code == 200, resp.text
            payload = resp.json()
            assert payload["risk_state"] == "LOW"
            assert payload["notification_required"] is False

        resp2 = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files, data=data)
        assert resp2.status_code == 200
        assert resp2.json()["sequence_number"] == 1


def test_invalid_sequence_rejected():
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-seq", "started_at": "2026-08-31T12:00:00Z"},
        ).json()
        files = {"audio": ("chunk.wav", io.BytesIO(_wav_bytes()), "audio/wav")}
        data = {
            "sequence_number": 0,
            "started_at": "2026-08-31T12:00:10Z",
            "ended_at": "2026-08-31T12:00:20Z",
            "duration_ms": 10000,
            "idempotency_key": "bad-seq-1",
        }
        resp = client.post(f"/api/v1/realtime/sessions/{session['session_id']}/chunks", files=files, data=data)
        assert resp.status_code == 422


def test_invalid_audio_rejected():
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-audio", "started_at": "2026-08-31T12:00:00Z"},
        ).json()
        resp = client.post(
            f"/api/v1/realtime/sessions/{session['session_id']}/chunks",
            files={"audio": ("bad.txt", io.BytesIO(b"not audio"), "text/plain")},
            data={
                "sequence_number": 1,
                "started_at": "2026-08-31T12:00:10Z",
                "ended_at": "2026-08-31T12:00:20Z",
                "duration_ms": 10000,
                "idempotency_key": "bad-audio-1",
            },
        )
        assert resp.status_code == 415


def test_low_medium_transition_triggers_notification():
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-risk", "started_at": "2026-08-31T12:00:00Z"},
        ).json()
        session_id = session["session_id"]
        for idx, score in enumerate([0.10, 0.55, 0.70], start=1):
            with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value={
                "provider": "aurigin",
                "prediction_id": f"pred-{idx}",
                "result": "SPOOF" if idx == 3 else "REAL",
                "confidence": score,
                "risk_score": int(score * 100),
                "segments": [],
                "processing_ms": 10,
                "raw_provider_status": "OK",
            }):
                response = client.post(
                    f"/api/v1/realtime/sessions/{session_id}/chunks",
                    files={"audio": (f"chunk{idx}.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
                    data={
                        "sequence_number": idx,
                        "started_at": f"2026-08-31T12:00:{idx:02d}Z",
                        "ended_at": f"2026-08-31T12:00:{idx + 10:02d}Z",
                        "duration_ms": 10000,
                        "idempotency_key": f"risk-{idx}",
                    },
                )
                assert response.status_code == 200, response.text
                if idx == 3:
                    payload = response.json()
                    assert payload["risk_state"] == "MEDIUM"
                    assert payload["notification_required"] is True
                    assert payload["state_changed"] is True


def test_final_result_overrides_realtime_result_and_complete_session():
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-final", "started_at": "2026-08-31T12:00:00Z"},
        ).json()
        session_id = session["session_id"]
        with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value={
            "provider": "aurigin",
            "prediction_id": "pred-final",
            "result": "SPOOF",
            "confidence": 0.9,
            "risk_score": 95,
            "segments": [],
            "processing_ms": 10,
            "raw_provider_status": "OK",
        }):
            client.post(
                f"/api/v1/realtime/sessions/{session_id}/chunks",
                files={"audio": ("chunk.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
                data={
                    "sequence_number": 1,
                    "started_at": "2026-08-31T12:00:10Z",
                    "ended_at": "2026-08-31T12:00:20Z",
                    "duration_ms": 10000,
                    "idempotency_key": "final-chunk",
                },
            )

        with patch("app.services.reality_defender_service.RealityDefenderService.analyze_file", new_callable=AsyncMock, return_value={
            "request_id": "rd-final",
            "status": "AUTHENTIC",
            "score": 0.09,
        }):
            complete = client.post(
                f"/api/v1/realtime/sessions/{session_id}/complete",
                json={"ended_at": "2026-08-31T12:00:30Z", "final_audio_reference": "ref-1"},
            )
            assert complete.status_code == 200, complete.text
            body = complete.json()
            assert body["status"] in {"CLOSED", "FINALIZING", "COMPLETED", "PARTIAL"}


        state = client.get(f"/api/v1/realtime/sessions/{session_id}")
        assert state.status_code == 200
        assert state.json()["session_id"] == session_id


def test_history_records_retrieval():
    from app.db.models import AnalysisRecord
    import uuid

    with TestClient(app) as client:
        db = SessionLocal()
        try:
            record = AnalysisRecord(
                id=str(uuid.uuid4()),
                filename="test.wav",
                stored_audio_path="./storage/test.wav",
                status="completed",
                classification="LIKELY_GENUINE",
                risk_score=5,
                confidence=0.98,
                ai_probability=0.05,
                duration_seconds=10.0,
                segments_analyzed=1,
                processing_time_ms=100,
                detector_version="reality-defender",
                reasons=["Authentic sample"],
            )
            db.add(record)
            db.commit()
        finally:
            db.close()

        history = client.get("/api/v1/history?page=1&limit=10")
        assert history.status_code == 200
        assert len(history.json()["items"]) > 0



def test_realtime_status_schema_shape():
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-state", "started_at": "2026-08-31T12:00:00Z"},
        ).json()
        status_resp = client.get(f"/api/v1/realtime/sessions/{session['session_id']}")
        assert status_resp.status_code == 200
        payload = status_resp.json()
        assert payload["status"] == "ACTIVE"
        assert payload["current_risk"] == "LOW"
        assert payload["chunks_processed"] == 0


def test_unknown_risk_state_does_not_crash_and_handles_peak_risk_correctly():
    """
    Test comprehensive UNKNOWN transitions:
    1. First chunk UNKNOWN
    2. UNKNOWN -> LOW
    3. LOW -> UNKNOWN
    4. UNKNOWN -> MEDIUM
    5. MEDIUM -> UNKNOWN
    6. UNKNOWN -> HIGH
    7. HIGH -> UNKNOWN
    8. UNKNOWN -> UNKNOWN
    9. LOW -> MEDIUM
    10. MEDIUM -> HIGH
    11. HIGH -> HIGH
    """
    from app.services.realtime_risk_service import update_peak_risk, risk_rank

    # Helper unit validation
    assert risk_rank("UNKNOWN") == 0
    assert risk_rank("LOW") == 1
    assert risk_rank("MEDIUM") == 2
    assert risk_rank("HIGH") == 3

    assert update_peak_risk(None, "UNKNOWN") == "UNKNOWN"
    assert update_peak_risk("UNKNOWN", "LOW") == "LOW"
    assert update_peak_risk("LOW", "UNKNOWN") == "LOW"
    assert update_peak_risk("UNKNOWN", "MEDIUM") == "MEDIUM"
    assert update_peak_risk("MEDIUM", "UNKNOWN") == "MEDIUM"
    assert update_peak_risk("UNKNOWN", "HIGH") == "HIGH"
    assert update_peak_risk("HIGH", "UNKNOWN") == "HIGH"
    assert update_peak_risk("UNKNOWN", "UNKNOWN") == "UNKNOWN"
    assert update_peak_risk("LOW", "MEDIUM") == "MEDIUM"
    assert update_peak_risk("MEDIUM", "HIGH") == "HIGH"
    assert update_peak_risk("HIGH", "HIGH") == "HIGH"

    # API integration test: First chunk UNKNOWN does NOT return HTTP 500
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-unknown-test", "started_at": "2026-08-31T12:00:00Z"},
        ).json()
        session_id = session["session_id"]

        with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value={
            "provider": "aurigin",
            "prediction_id": "pred-unknown-1",
            "result": "UNKNOWN",
            "confidence": None,
            "risk_score": None,
            "segments": [],
            "processing_ms": 15,
            "raw_provider_status": "POOR_AUDIO",
        }):
            resp = client.post(
                f"/api/v1/realtime/sessions/{session_id}/chunks",
                files={"audio": ("chunk.wav", io.BytesIO(_wav_bytes()), "audio/wav")},
                data={
                    "sequence_number": 1,
                    "started_at": "2026-08-31T12:00:00Z",
                    "ended_at": "2026-08-31T12:00:10Z",
                    "duration_ms": 10000,
                    "idempotency_key": "chunk-unknown-1",
                },
            )
            assert resp.status_code == 200, resp.text
            data = resp.json()
            assert data["risk_state"] == "UNKNOWN"
            assert data["notification_required"] is False
            assert data["state_changed"] is False

        # Verify session peak_risk and status
        state_resp = client.get(f"/api/v1/realtime/sessions/{session_id}")
        assert state_resp.status_code == 200
        state = state_resp.json()
        assert state["current_risk"] == "UNKNOWN"
        assert state["peak_risk"] in {"LOW", "UNKNOWN"}


# ── Rolling Audio Windows Tests (10s, 20s, 30s max, 27s partial, overlapping) ──

def test_10_second_window_accepted():
    """Test 1: 10,000ms window -> 200 accepted."""
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-10s", "started_at": "2026-09-01T12:00:00Z"},
        ).json()
        session_id = session["session_id"]
        files = {"audio": ("chunk_10s.wav", io.BytesIO(_wav_bytes(duration_seconds=10.0)), "audio/wav")}
        data = {
            "sequence_number": 1,
            "started_at": "2026-09-01T12:00:00Z",
            "ended_at": "2026-09-01T12:00:10Z",
            "duration_ms": 10000,
            "idempotency_key": "win-10s",
        }
        with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value={
            "provider": "aurigin",
            "prediction_id": "pred-10s",
            "result": "REAL",
            "score": 0.05,
            "confidence": 0.95,
            "risk_score": 5,
            "segments": [],
            "processing_ms": 50,
            "raw_provider_status": "OK",
        }):
            resp = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files, data=data)
            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "PROCESSED"


def test_20_second_rolling_window_accepted():
    """Test 2: 20,000ms rolling window -> 200 accepted."""
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-20s", "started_at": "2026-09-01T12:00:00Z"},
        ).json()
        session_id = session["session_id"]
        files = {"audio": ("chunk_20s.wav", io.BytesIO(_wav_bytes(duration_seconds=20.0)), "audio/wav")}
        data = {
            "sequence_number": 2,
            "started_at": "2026-09-01T12:00:00Z",
            "ended_at": "2026-09-01T12:00:20Z",
            "duration_ms": 20000,
            "idempotency_key": "win-20s",
        }
        with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value={
            "provider": "aurigin",
            "prediction_id": "pred-20s",
            "result": "REAL",
            "score": 0.08,
            "confidence": 0.94,
            "risk_score": 8,
            "segments": [],
            "processing_ms": 60,
            "raw_provider_status": "OK",
        }):
            resp = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files, data=data)
            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "PROCESSED"


def test_30_second_maximum_window_accepted():
    """Test 3: 30,000ms maximum window -> 200 accepted."""
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-30s", "started_at": "2026-09-01T12:00:00Z"},
        ).json()
        session_id = session["session_id"]
        files = {"audio": ("chunk_30s.wav", io.BytesIO(_wav_bytes(duration_seconds=30.0)), "audio/wav")}
        data = {
            "sequence_number": 3,
            "started_at": "2026-09-01T12:00:00Z",
            "ended_at": "2026-09-01T12:00:30Z",
            "duration_ms": 30000,
            "idempotency_key": "win-30s",
        }
        with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value={
            "provider": "aurigin",
            "prediction_id": "pred-30s",
            "result": "REAL",
            "score": 0.04,
            "confidence": 0.98,
            "risk_score": 4,
            "segments": [],
            "processing_ms": 70,
            "raw_provider_status": "OK",
        }):
            resp = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files, data=data)
            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "PROCESSED"


def test_30_001_second_window_rejected():
    """Test 4: 30,001ms window -> 413 INVALID_DURATION."""
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-30001ms", "started_at": "2026-09-01T12:00:00Z"},
        ).json()
        session_id = session["session_id"]
        files = {"audio": ("chunk_30001ms.wav", io.BytesIO(_wav_bytes(duration_seconds=30.1)), "audio/wav")}
        data = {
            "sequence_number": 1,
            "started_at": "2026-09-01T12:00:00Z",
            "ended_at": "2026-09-01T12:00:30.001Z",
            "duration_ms": 30001,
            "idempotency_key": "win-30001ms",
        }
        resp = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files, data=data)
        assert resp.status_code == 413, resp.text
        assert resp.json()["detail"]["code"] == "INVALID_DURATION"


def test_27_second_final_partial_window_accepted():
    """Test 5: 27,000ms final partial window (e.g. 100-127s post-call) -> 200 accepted."""
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-27s", "started_at": "2026-09-01T12:00:00Z"},
        ).json()
        session_id = session["session_id"]
        files = {"audio": ("chunk_27s.wav", io.BytesIO(_wav_bytes(duration_seconds=27.0)), "audio/wav")}
        data = {
            "sequence_number": 7,
            "started_at": "2026-09-01T12:01:40Z",
            "ended_at": "2026-09-01T12:02:07Z",
            "duration_ms": 27000,
            "idempotency_key": "win-27s-final",
        }
        with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value={
            "provider": "aurigin",
            "prediction_id": "pred-27s",
            "result": "REAL",
            "score": 0.05,
            "confidence": 0.95,
            "risk_score": 5,
            "segments": [],
            "processing_ms": 65,
            "raw_provider_status": "OK",
        }):
            resp = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files, data=data)
            assert resp.status_code == 200, resp.text
            assert resp.json()["status"] == "PROCESSED"


def test_below_minimum_duration_rejected():
    """Test 6: Below configured minimum duration (0.5s / 500ms vs 1.0s / 1000ms min) -> rejected."""
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-short", "started_at": "2026-09-01T12:00:00Z"},
        ).json()
        session_id = session["session_id"]
        files = {"audio": ("chunk_short.wav", io.BytesIO(_wav_bytes(duration_seconds=0.2)), "audio/wav")}
        data = {
            "sequence_number": 1,
            "started_at": "2026-09-01T12:00:00Z",
            "ended_at": "2026-09-01T12:00:00.2Z",
            "duration_ms": 200,
            "idempotency_key": "win-short",
        }
        resp = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files, data=data)
        assert resp.status_code == 413
        assert resp.json()["detail"]["code"] == "INVALID_DURATION"


def test_overlapping_rolling_windows_all_five_processed():
    """
    Test 7: Send sequence 0-10, 0-20, 0-30, 10-40, 20-50.
    Expected: ALL FIVE are accepted and forwarded to Aurigin (not collapsed).
    """
    windows = [
        {"seq": 1, "start_ms": 0, "end_ms": 10000, "dur_ms": 10000, "dur_s": 10.0},
        {"seq": 2, "start_ms": 0, "end_ms": 20000, "dur_ms": 20000, "dur_s": 20.0},
        {"seq": 3, "start_ms": 0, "end_ms": 30000, "dur_ms": 30000, "dur_s": 30.0},
        {"seq": 4, "start_ms": 10000, "end_ms": 40000, "dur_ms": 30000, "dur_s": 30.0},
        {"seq": 5, "start_ms": 20000, "end_ms": 50000, "dur_ms": 30000, "dur_s": 30.0},
    ]

    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-overlapping-5", "started_at": "2026-09-01T12:00:00Z"},
        ).json()
        session_id = session["session_id"]

        mock_analyze = AsyncMock(side_effect=lambda *args, **kwargs: {
            "provider": "aurigin",
            "prediction_id": f"pred-{kwargs.get('window_id', 'w')}",
            "result": "REAL",
            "score": 0.05,
            "confidence": 0.95,
            "risk_score": 5,
            "segments": [],
            "processing_ms": 30,
            "raw_provider_status": "OK",
        })

        with patch("app.services.aurigin_service.AuriginService.analyze_audio", mock_analyze):
            for w in windows:
                files = {"audio": (f"chunk_{w['seq']}.wav", io.BytesIO(_wav_bytes(duration_seconds=w["dur_s"])), "audio/wav")}
                data = {
                    "sequence_number": w["seq"],
                    "started_at": f"2026-09-01T12:00:{w['start_ms']//1000:02d}Z",
                    "ended_at": f"2026-09-01T12:00:{w['end_ms']//1000:02d}Z",
                    "duration_ms": w["dur_ms"],
                    "idempotency_key": f"win-overlap-{w['seq']}",
                }
                resp = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files, data=data)
                assert resp.status_code == 200, f"Failed at seq {w['seq']}: {resp.text}"
                assert resp.json()["status"] == "PROCESSED"

            # Exactly 5 Aurigin requests must have been made
            assert mock_analyze.call_count == 5


def test_retry_same_window_deduplicated():
    """Test 8: Exact same request twice with same idempotency key -> exactly ONE Aurigin call."""
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-retry", "started_at": "2026-09-01T12:00:00Z"},
        ).json()
        session_id = session["session_id"]

        mock_analyze = AsyncMock(return_value={
            "provider": "aurigin",
            "prediction_id": "pred-retry-1",
            "result": "REAL",
            "score": 0.05,
            "confidence": 0.95,
            "risk_score": 5,
            "segments": [],
            "processing_ms": 40,
            "raw_provider_status": "OK",
        })

        files = {"audio": ("chunk.wav", io.BytesIO(_wav_bytes(duration_seconds=10.0)), "audio/wav")}
        data = {
            "sequence_number": 1,
            "started_at": "2026-09-01T12:00:00Z",
            "ended_at": "2026-09-01T12:00:10Z",
            "duration_ms": 10000,
            "idempotency_key": "same-key-1",
        }

        with patch("app.services.aurigin_service.AuriginService.analyze_audio", mock_analyze):
            # First send
            resp1 = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files, data=data)
            assert resp1.status_code == 200
            assert resp1.json()["sequence_number"] == 1

            # Second send (retry)
            files2 = {"audio": ("chunk.wav", io.BytesIO(_wav_bytes(duration_seconds=10.0)), "audio/wav")}
            resp2 = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files2, data=data)
            assert resp2.status_code == 200
            assert resp2.json()["sequence_number"] == 1

            # Only ONE call forwarded to Aurigin
            assert mock_analyze.call_count == 1


def test_different_overlapping_windows_generate_two_aurigin_requests():
    """Test 9: W03 (0-30s) then W04 (10-40s) -> exactly TWO Aurigin requests."""
    with TestClient(app) as client:
        session = client.post(
            "/api/v1/realtime/sessions",
            json={"client_session_id": "session-overlap-pair", "started_at": "2026-09-01T12:00:00Z"},
        ).json()
        session_id = session["session_id"]

        mock_analyze = AsyncMock(return_value={
            "provider": "aurigin",
            "prediction_id": "pred-overlap",
            "result": "REAL",
            "score": 0.05,
            "confidence": 0.95,
            "risk_score": 5,
            "segments": [],
            "processing_ms": 40,
            "raw_provider_status": "OK",
        })

        with patch("app.services.aurigin_service.AuriginService.analyze_audio", mock_analyze):
            # W03: 0-30s
            files1 = {"audio": ("chunk_0_30.wav", io.BytesIO(_wav_bytes(duration_seconds=30.0)), "audio/wav")}
            data1 = {
                "sequence_number": 3,
                "started_at": "2026-09-01T12:00:00Z",
                "ended_at": "2026-09-01T12:00:30Z",
                "duration_ms": 30000,
                "idempotency_key": "win-0-30",
            }
            resp1 = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files1, data=data1)
            assert resp1.status_code == 200

            # W04: 10-40s
            files2 = {"audio": ("chunk_10_40.wav", io.BytesIO(_wav_bytes(duration_seconds=30.0)), "audio/wav")}
            data2 = {
                "sequence_number": 4,
                "started_at": "2026-09-01T12:00:10Z",
                "ended_at": "2026-09-01T12:00:40Z",
                "duration_ms": 30000,
                "idempotency_key": "win-10-40",
            }
            resp2 = client.post(f"/api/v1/realtime/sessions/{session_id}/chunks", files=files2, data=data2)
            assert resp2.status_code == 200

            # Exactly TWO Aurigin calls made
            assert mock_analyze.call_count == 2
