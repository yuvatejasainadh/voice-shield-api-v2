from __future__ import annotations

import base64
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db.database import Base, SessionLocal, engine
from app.db.models import CallSession
from app.main import app
from app.services.realtime_session_manager import RealtimeSessionManager


def _make_pcm_base64(duration_seconds: float = 1.0, sample_rate: int = 16000) -> str:
    """Generate base64 encoded 16-bit mono PCM bytes."""
    sample_count = int(duration_seconds * sample_rate)
    pcm_bytes = b"\x00\x00" * sample_count
    return base64.b64encode(pcm_bytes).decode("utf-8")


@pytest.fixture(autouse=True)
def cleanup_database_and_sessions():
    Base.metadata.create_all(bind=engine)
    RealtimeSessionManager.get_instance().clear_all()
    yield
    RealtimeSessionManager.get_instance().clear_all()


# ── Test 1: 2.5 second window ─────────────────────────────────────────────

def test_windowing_sync_2_5s_window():
    """
    Test 1: 2.5 second window (W001: 0-2500 ms)
    - Accepted by backend
    - Triggers exactly 1 ML inference
    - Preserves window boundaries without secondary windowing
    """
    mock_aurigin = {
        "provider": "aurigin",
        "prediction_id": "pred-w1",
        "result": "REAL",
        "score": 0.05,
        "confidence": 0.96,
        "risk_score": 5,
        "segments": [],
        "processing_ms": 30,
        "raw_provider_status": "PROCESSED",
        "raw_result": "bonafide",
    }

    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin) as mock_analyze:
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/realtime/ws") as ws:
                # 1. Start call
                ws.send_text(json.dumps({
                    "type": "call_start",
                    "callSessionId": "test-sync-2-5s",
                    "timestamp": time.time() * 1000,
                }))
                msg1 = json.loads(ws.receive_text())
                assert msg1["type"] == "call_started"
                msg2 = json.loads(ws.receive_text())
                assert msg2["type"] == "call_report_created"

                # 2. Send 2.5s window W001
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": "test-sync-2-5s",
                    "sequenceNumber": 1,
                    "windowStartMs": 0,
                    "windowEndMs": 2500,
                    "windowId": "W001",
                    "requestId": "REQ_W001",
                    "durationMs": 2500,
                    "audioFormat": {
                        "sampleRate": 16000,
                        "channels": 1,
                        "encoding": "pcm_s16le",
                    },
                    "audio": _make_pcm_base64(2.5),
                    "clientTimestampMs": 1000.0,
                }))

                # Receive detection_result
                res = json.loads(ws.receive_text())
                assert res["type"] == "detection_result"
                assert res["sequenceNumber"] == 1
                assert res["windowStartMs"] == 0
                assert res["windowEndMs"] == 2500
                assert res["windowId"] == "W001"
                assert res["requestId"] == "REQ_W001"
                assert res["durationMs"] == 2500

                # Receive call_report_updated
                rep_upd = json.loads(ws.receive_text())
                assert rep_upd["type"] == "call_report_updated"
                assert rep_upd["report"]["totalDurationSeconds"] == 2.5

                # Verify exactly 1 ML inference was called
                assert mock_analyze.call_count == 1
                call_args = mock_analyze.call_args[1]
                assert call_args["window_id"] == "W001"
                assert call_args["duration_ms"] == 2500


# ── Test 2: 5 second window ───────────────────────────────────────────────

def test_windowing_sync_5s_window():
    """
    Test 2: 5 second window (W002: 0-5000 ms)
    - Accepted by backend
    - Triggers exactly 1 ML inference
    """
    mock_aurigin = {
        "provider": "aurigin",
        "prediction_id": "pred-w2",
        "result": "REAL",
        "score": 0.08,
        "confidence": 0.95,
        "risk_score": 8,
        "segments": [],
        "processing_ms": 40,
        "raw_provider_status": "PROCESSED",
        "raw_result": "bonafide",
    }

    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin) as mock_analyze:
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/realtime/ws") as ws:
                ws.send_text(json.dumps({
                    "type": "call_start",
                    "callSessionId": "test-sync-5s",
                }))
                ws.receive_text()  # call_started
                ws.receive_text()  # call_report_created

                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": "test-sync-5s",
                    "sequenceNumber": 2,
                    "windowStartMs": 0,
                    "windowEndMs": 5000,
                    "windowId": "W002",
                    "requestId": "REQ_W002",
                    "durationMs": 5000,
                    "audio": _make_pcm_base64(5.0),
                }))

                res = json.loads(ws.receive_text())
                assert res["type"] == "detection_result"
                assert res["sequenceNumber"] == 2
                assert res["windowStartMs"] == 0
                assert res["windowEndMs"] == 5000
                assert res["windowId"] == "W002"

                assert mock_analyze.call_count == 1


# ── Test 3: Overlapping Sequence & Duration Accounting ────────────────────

def test_windowing_sync_overlapping_sequence_duration_accounting():
    """
    Test 3: Overlapping sequence
    W001 = 0–2500
    W002 = 0–5000
    W003 = 2500–7500
    W004 = 5000–10000

    Expected:
    - 4 windows received
    - 4 distinct ML inferences
    - Total timeline duration = 10.0s (NOT 17.5s)
    """
    mock_aurigin = {
        "provider": "aurigin",
        "prediction_id": "pred-overlap",
        "result": "REAL",
        "score": 0.05,
        "confidence": 0.97,
        "risk_score": 5,
        "segments": [],
        "processing_ms": 25,
        "raw_provider_status": "PROCESSED",
        "raw_result": "bonafide",
    }

    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin) as mock_analyze:
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/realtime/ws") as ws:
                ws.send_text(json.dumps({"type": "call_start", "callSessionId": "test-overlap-seq"}))
                ws.receive_text()  # call_started
                ws.receive_text()  # call_report_created

                windows = [
                    (1, 0, 2500, "W001", 2.5),
                    (2, 0, 5000, "W002", 5.0),
                    (3, 2500, 7500, "W003", 5.0),
                    (4, 5000, 10000, "W004", 5.0),
                ]

                for seq, start, end, wid, dur_s in windows:
                    ws.send_text(json.dumps({
                        "type": "audio_window",
                        "callSessionId": "test-overlap-seq",
                        "sequenceNumber": seq,
                        "windowStartMs": start,
                        "windowEndMs": end,
                        "windowId": wid,
                        "durationMs": int(dur_s * 1000),
                        "audio": _make_pcm_base64(dur_s),
                    }))
                    res = json.loads(ws.receive_text())
                    assert res["type"] == "detection_result"
                    assert res["sequenceNumber"] == seq
                    rep_upd = json.loads(ws.receive_text())
                    assert rep_upd["type"] == "call_report_updated"

                assert mock_analyze.call_count == 4

                # End call and verify total duration
                ws.send_text(json.dumps({"type": "call_end", "callSessionId": "test-overlap-seq"}))
                fin_rep = json.loads(ws.receive_text())
                assert fin_rep["type"] == "call_report_completed"
                # Total duration must be 10.0s (max timeline end), NOT 17.5s (sum)
                assert fin_rep["report"]["totalDurationSeconds"] == 10.0
                assert fin_rep["report"]["chunksAnalyzed"] == 4

                # Verify database persistence
                with SessionLocal() as db:
                    db_sess = db.query(CallSession).filter(CallSession.id == "test-overlap-seq").first()
                    assert db_sess is not None
                    assert db_sess.total_duration_seconds == 10.0
                    assert db_sess.chunks_analyzed == 4
                    assert len(db_sess.evidence) == 4
                    assert db_sess.evidence[0]["windowId"] == "W001"
                    assert db_sess.evidence[1]["windowId"] == "W002"
                    assert db_sess.evidence[2]["windowId"] == "W003"
                    assert db_sess.evidence[3]["windowId"] == "W004"


# ── Test 4: 500 ms Partial Window ─────────────────────────────────────────

def test_windowing_sync_500ms_partial_window():
    """
    Test 4: 500 ms partial window (W005: 10000-10500 ms)
    - Accepted without padding
    - Triggers exactly 1 ML inference
    """
    mock_aurigin = {
        "provider": "aurigin",
        "prediction_id": "pred-500ms",
        "result": "REAL",
        "score": 0.05,
        "confidence": 0.90,
        "risk_score": 5,
        "segments": [],
        "processing_ms": 20,
        "raw_provider_status": "PROCESSED",
        "raw_result": "bonafide",
    }

    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin) as mock_analyze:
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/realtime/ws") as ws:
                ws.send_text(json.dumps({"type": "call_start", "callSessionId": "test-partial-500ms"}))
                ws.receive_text()  # call_started
                ws.receive_text()  # call_report_created

                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": "test-partial-500ms",
                    "sequenceNumber": 5,
                    "windowStartMs": 10000,
                    "windowEndMs": 10500,
                    "windowId": "W005",
                    "durationMs": 500,
                    "audio": _make_pcm_base64(0.5),
                }))

                res = json.loads(ws.receive_text())
                assert res["type"] == "detection_result"
                assert res["sequenceNumber"] == 5
                assert res["durationMs"] == 500

                assert mock_analyze.call_count == 1
                call_args = mock_analyze.call_args[1]
                assert call_args["duration_ms"] == 500


# ── Test 5: Duplicate Sequence Deduplication ──────────────────────────────

def test_windowing_sync_duplicate_sequence_dropped():
    """
    Test 5: Duplicate sequence
    Send W003 twice -> First processed, second dropped, exactly 1 ML inference.
    """
    mock_aurigin = {
        "provider": "aurigin",
        "prediction_id": "pred-dup",
        "result": "REAL",
        "score": 0.05,
        "confidence": 0.95,
        "risk_score": 5,
        "segments": [],
        "processing_ms": 25,
        "raw_provider_status": "PROCESSED",
        "raw_result": "bonafide",
    }

    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin) as mock_analyze:
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/realtime/ws") as ws:
                ws.send_text(json.dumps({"type": "call_start", "callSessionId": "test-dup-seq"}))
                ws.receive_text()
                ws.receive_text()

                # Send W003 first time
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": "test-dup-seq",
                    "sequenceNumber": 3,
                    "windowStartMs": 2500,
                    "windowEndMs": 7500,
                    "windowId": "W003",
                    "durationMs": 5000,
                    "audio": _make_pcm_base64(5.0),
                }))
                res1 = json.loads(ws.receive_text())
                assert res1["type"] == "detection_result"
                rep1 = json.loads(ws.receive_text())
                assert rep1["type"] == "call_report_updated"
                assert mock_analyze.call_count == 1

                # Send W003 duplicate second time
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": "test-dup-seq",
                    "sequenceNumber": 3,
                    "windowStartMs": 2500,
                    "windowEndMs": 7500,
                    "windowId": "W003",
                    "durationMs": 5000,
                    "audio": _make_pcm_base64(5.0),
                }))

                # Send ping to confirm queue processed
                ws.send_text(json.dumps({"type": "ping"}))
                pong = json.loads(ws.receive_text())
                assert pong["type"] == "pong"

                # ML inference count remains 1
                assert mock_analyze.call_count == 1


# ── Test 6: Metadata Preservation ─────────────────────────────────────────

def test_windowing_sync_metadata_preservation():
    """
    Test 6: Metadata preservation
    Verify: sequenceNumber, windowStartMs, windowEndMs, durationMs, windowId,
    requestId, clientTimestampMs remain correctly preserved and returned.
    """
    mock_aurigin = {
        "provider": "aurigin",
        "prediction_id": "pred-meta-123",
        "result": "REAL",
        "score": 0.04,
        "confidence": 0.99,
        "risk_score": 4,
        "segments": [],
        "processing_ms": 28,
        "raw_provider_status": "PROCESSED",
        "raw_result": "bonafide",
    }

    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin):
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/realtime/ws") as ws:
                ws.send_text(json.dumps({"type": "call_start", "callSessionId": "test-meta-pres"}))
                ws.receive_text()
                ws.receive_text()

                client_ts = 1727500000123.0
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": "test-meta-pres",
                    "sequenceNumber": 42,
                    "windowStartMs": 102500,
                    "windowEndMs": 107500,
                    "durationMs": 5000,
                    "windowId": "W042",
                    "requestId": "REQ_CUSTOM_042",
                    "clientTimestampMs": client_ts,
                    "audio": _make_pcm_base64(5.0),
                }))

                res = json.loads(ws.receive_text())
                assert res["type"] == "detection_result"
                assert res["callSessionId"] == "test-meta-pres"
                assert res["sequenceNumber"] == 42
                assert res["windowStartMs"] == 102500
                assert res["windowEndMs"] == 107500
                assert res["durationMs"] == 5000
                assert res["windowId"] == "W042"
                assert res["requestId"] == "REQ_CUSTOM_042"
                assert res["metrics"] is not None
                assert res["metrics"]["windowDurationMs"] == 5000
                assert res["metrics"]["clientTransmissionLatencyMs"] is not None

                rep_upd = json.loads(ws.receive_text())
                latest_win = rep_upd["latestWindow"]
                assert latest_win["windowId"] == "W042"
                assert latest_win["sequenceNumber"] == 42
                assert latest_win["windowStartMs"] == 102500
                assert latest_win["windowEndMs"] == 107500


# ── Test 7: Audio Format Compatibility Validation ─────────────────────────

def test_windowing_sync_unsupported_audio_format_rejected():
    """
    Test 7: Audio Format Validation
    Unsupported sample rate or encoding should be rejected with UNSUPPORTED_AUDIO_FORMAT.
    """
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/realtime/ws") as ws:
            ws.send_text(json.dumps({"type": "call_start", "callSessionId": "test-bad-format"}))
            ws.receive_text()
            ws.receive_text()

            # Send 44.1kHz stereo format
            ws.send_text(json.dumps({
                "type": "audio_window",
                "callSessionId": "test-bad-format",
                "sequenceNumber": 1,
                "windowStartMs": 0,
                "windowEndMs": 2500,
                "audioFormat": {
                    "sampleRate": 44100,
                    "channels": 2,
                    "encoding": "aac",
                },
                "audio": _make_pcm_base64(2.5),
            }))

            err = json.loads(ws.receive_text())
            assert err["type"] == "error"
            assert err["code"] == "UNSUPPORTED_AUDIO_FORMAT"
