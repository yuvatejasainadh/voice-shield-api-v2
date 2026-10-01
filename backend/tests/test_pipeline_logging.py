"""Tests for Voice Shield Clean Structured Realtime Pipeline Logging."""

from __future__ import annotations

import base64
import json
import logging
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.pipeline_logger import (
    VoiceShieldPipelineLogger,
    format_call_id,
    format_range_ms,
    format_window_id,
)
from app.db.database import Base, engine
from app.main import app
from app.services.realtime_session_manager import RealtimeSessionManager


def _make_base64_pcm(duration_seconds: float = 10.0, sample_rate: int = 16000) -> str:
    sample_count = int(duration_seconds * sample_rate)
    return base64.b64encode(b"\x00\x00" * sample_count).decode("utf-8")


@pytest.fixture(autouse=True)
def cleanup_state():
    Base.metadata.create_all(bind=engine)
    RealtimeSessionManager.get_instance().clear_all()
    yield
    RealtimeSessionManager.get_instance().clear_all()


def test_formatting_helpers():
    """Verify call_id, window_id, and time range formatters."""
    # 1. Call ID formatting
    assert format_call_id("CALL-7F3A21") == "CALL-7F3A21"
    assert format_call_id("96b4f597-d56c-4e04-a3ac-75d3f1e77905") == "CALL-E77905"
    assert format_call_id("abc") == "CALL-ABC"
    assert format_call_id(None) == "CALL-UNKNOWN"

    # 2. Window ID formatting
    assert format_window_id(1) == "W01"
    assert format_window_id(13) == "W13"
    assert format_window_id("W04") == "W04"
    assert format_window_id("w06") == "W06"

    # 3. Time range formatting
    assert format_range_ms(0, 10000) == "00:00-00:10"
    assert format_range_ms(0, 30000) == "00:00-00:30"
    assert format_range_ms(10000, 40000) == "00:10-00:40"
    assert format_range_ms(24000, 54000) == "00:24-00:54"
    assert format_range_ms(50000, 54000) == "00:50-00:54"


def test_pipeline_logger_lifecycle_events(caplog):
    """Verify clean pipeline logger lifecycle methods format messages correctly without logging raw audio."""
    caplog.set_level(logging.INFO, logger="voice-clone-detection")
    call_id = "CALL-7F3A21"

    # 1. Call start
    VoiceShieldPipelineLogger.call_started(call_id)
    assert "[CALL] STARTED call_id=CALL-7F3A21" in caplog.text

    # 2. TCED Config
    VoiceShieldPipelineLogger.tced_config(10.0, 20.0, 30.0)
    assert "[TCED] CONFIG new_audio=10s stride=20s max_window=30s" in caplog.text

    # 3. Window Created
    VoiceShieldPipelineLogger.window_created("W01", 0, 10000, 1)
    assert "[TCED] [W01] WINDOW_CREATED range=00:00-00:10 duration=10s sequence=1" in caplog.text

    # 4. Detector Started
    VoiceShieldPipelineLogger.detector_started("W01", "AURIGIN", "REQ_001")
    assert "[DETECTOR] [W01] ANALYSIS_STARTED detector=AURIGIN request_id=REQ_001" in caplog.text

    # 5. Detector Completed
    VoiceShieldPipelineLogger.detector_completed(
        "W01",
        result="BONAFIDE",
        confidence=0.94,
        score=0.06,
        latency_ms=597,
        prediction_id="pred_001",
    )
    assert "[DETECTOR] [W01] ANALYSIS_COMPLETED result=BONAFIDE confidence=0.94 score=0.06 latency_ms=597 prediction_id=pred_001" in caplog.text

    # 6. Decision
    VoiceShieldPipelineLogger.decision("W01", risk="LOW", result="BONAFIDE", confidence=0.94)
    assert "[DECISION] [W01] risk=LOW result=BONAFIDE confidence=0.94" in caplog.text

    # 7. Tail Handling
    VoiceShieldPipelineLogger.tced_call_ended(54.0)
    assert "[TCED] CALL_ENDED duration=54s" in caplog.text

    VoiceShieldPipelineLogger.tced_tail_detected(50000, 54000)
    assert "[TCED] TAIL_DETECTED unprocessed_range=00:50-00:54" in caplog.text

    VoiceShieldPipelineLogger.tail_window_created("W06", 24000, 54000, 6)
    assert "[TCED] [W06] TAIL_WINDOW_CREATED range=00:24-00:54 duration=30s sequence=6" in caplog.text

    VoiceShieldPipelineLogger.tced_tail_suppressed("DUPLICATE", 0, 30000)
    assert "[TCED] TAIL_SUPPRESSED reason=DUPLICATE range=00:00-00:30" in caplog.text

    # 8. Call Completion & History
    VoiceShieldPipelineLogger.call_completed(
        duration_s=54.0,
        windows=6,
        bonafide=3,
        spoofed=3,
        partially_spoofed=0,
        final_risk="HIGH",
        alert_triggered=True,
    )
    assert "[CALL] COMPLETED duration=54s windows=6 bonafide=3 spoofed=3 partially_spoofed=0 final_risk=HIGH alert_triggered=true" in caplog.text

    hist = [("W01", "BONAFIDE", 0.91), ("W02", "BONAFIDE", 0.92), ("W03", "BONAFIDE", 0.92), ("W04", "SPOOF", 0.87), ("W05", "SPOOF", 0.93), ("W06", "SPOOF", 0.96)]
    VoiceShieldPipelineLogger.call_history(hist)
    assert "[CALL] HISTORY W01=BONAFIDE(0.91) W02=BONAFIDE(0.92) W03=BONAFIDE(0.92) W04=SPOOF(0.87) W05=SPOOF(0.93) W06=SPOOF(0.96)" in caplog.text


def test_realtime_websocket_full_clean_pipeline_logging(caplog):
    """End-to-end test verifying WebSocket logs follow the clean lifecycle: CALL -> TCED -> DETECTOR -> DECISION."""
    caplog.set_level(logging.INFO, logger="voice-clone-detection")
    call_id = "CALL-TEST01"

    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/realtime/ws") as ws:
            # 1. Start Call
            ws.send_text(json.dumps({"type": "call_start", "callSessionId": call_id}))
            assert json.loads(ws.receive_text())["type"] == "call_started"
            assert json.loads(ws.receive_text())["type"] == "call_report_created"

            # Verify call start and TCED config logged once
            assert "[CALL] STARTED call_id=CALL-TEST01" in caplog.text
            assert "[TCED] CONFIG new_audio=2.5s stride=2.5s max_window=5s" in caplog.text

            # 2. Window 1 (0-10s)
            mock_aurigin_1 = {
                "provider": "aurigin",
                "prediction_id": "pred-1",
                "result": "REAL",
                "score": 0.06,
                "confidence": 0.94,
                "risk_score": 6,
                "segments": [],
                "processing_ms": 50,
                "raw_provider_status": "PROCESSED",
                "raw_result": "bonafide",
            }

            with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin_1):
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": call_id,
                    "sequenceNumber": 1,
                    "windowStartMs": 0,
                    "windowEndMs": 10000,
                    "audio": _make_base64_pcm(10.0),
                    "clientTimestampMs": time.time() * 1000,
                }))

                det1 = json.loads(ws.receive_text())
                assert det1["type"] == "detection_result"
                upd1 = json.loads(ws.receive_text())
                assert upd1["type"] == "call_report_updated"

            # Check clean logs for Window 1
            assert "[TCED] [W01] WINDOW_CREATED range=00:00-00:10 duration=10s sequence=1" in caplog.text
            assert "[DETECTOR] [W01] ANALYSIS_STARTED detector=AURIGIN" in caplog.text
            assert "[DETECTOR] [W01] ANALYSIS_COMPLETED result=BONAFIDE confidence=0.94 score=0.06" in caplog.text
            assert "[DECISION] [W01] risk=LOW result=BONAFIDE confidence=0.94" in caplog.text

            # 3. Window 2 (0-20s, Partially Spoofed)
            mock_aurigin_2 = {
                "provider": "aurigin",
                "prediction_id": "pred-2",
                "result": "SPOOFED",
                "score": 0.76,
                "confidence": 0.91,
                "risk_score": 76,
                "segments": [],
                "processing_ms": 60,
                "raw_provider_status": "PROCESSED",
                "raw_result": "partially_spoofed",
            }

            with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin_2):
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": call_id,
                    "sequenceNumber": 2,
                    "windowStartMs": 0,
                    "windowEndMs": 20000,
                    "audio": _make_base64_pcm(10.0),
                    "clientTimestampMs": time.time() * 1000,
                }))

                det2 = json.loads(ws.receive_text())
                assert det2["type"] == "detection_result"
                upd2 = json.loads(ws.receive_text())
                assert upd2["type"] == "call_report_updated"
                risk_msg = json.loads(ws.receive_text())
                assert risk_msg["type"] == "risk_update"

            # Check clean logs for Window 2
            assert "[TCED] [W02] WINDOW_CREATED range=00:00-00:20 duration=20s sequence=2" in caplog.text
            assert "[DETECTOR] [W02] ANALYSIS_STARTED detector=AURIGIN" in caplog.text
            assert "[DETECTOR] [W02] ANALYSIS_COMPLETED result=PARTIALLY_SPOOFED confidence=0.91 score=0.76" in caplog.text
            assert "[DECISION] [W02] risk=MEDIUM result=PARTIALLY_SPOOFED confidence=0.91" in caplog.text

            # 4. End Call
            ws.send_text(json.dumps({"type": "call_end", "callSessionId": call_id}))
            comp_msg = json.loads(ws.receive_text())
            assert comp_msg["type"] == "call_report_completed"
            assert json.loads(ws.receive_text())["type"] == "session_closed"

            # Check Call Completed Summary & Final History
            assert "[CALL] COMPLETED duration=20s windows=2" in caplog.text
            assert "[CALL] HISTORY W01=BONAFIDE(0.94) W02=PARTIALLY_SPOOFED(0.91)" in caplog.text

            # Ensure old verbose/redundant messages are NOT emitted at INFO level
            assert "WINDOW_DETECTED" not in caplog.text
            assert "WINDOW_VALIDATED" not in caplog.text
            assert "WINDOW_ACCEPTED" not in caplog.text
            assert "POST #" not in caplog.text
            assert "[DETECTION_RESULT]" not in caplog.text
            assert "[SESSION_STATE]" not in caplog.text


def test_websocket_54s_call_tced_and_tail_clean_logging(caplog):
    """Verify 54s call emits full windowing and tail logs (W01..W06)."""
    caplog.set_level(logging.INFO, logger="voice-clone-detection")
    session_id = "CALL-54S-LOG"

    mock_aurigin = {
        "provider": "aurigin",
        "prediction_id": "pred-54s",
        "result": "REAL",
        "score": 0.05,
        "confidence": 0.98,
        "risk_score": 5,
        "segments": [],
        "processing_ms": 35,
        "raw_provider_status": "PROCESSED",
        "raw_result": "bonafide",
    }

    with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin):
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/realtime/ws") as ws:
                ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
                assert json.loads(ws.receive_text())["type"] == "call_started"
                assert json.loads(ws.receive_text())["type"] == "call_report_created"

                windows_54s = [
                    (1, 0, 10000, 10.0),
                    (2, 0, 20000, 20.0),
                    (3, 0, 30000, 30.0),
                    (4, 10000, 40000, 30.0),
                    (5, 20000, 50000, 30.0),
                    (6, 24000, 54000, 30.0),
                ]

                for seq, start_ms, end_ms, dur_s in windows_54s:
                    ws.send_text(json.dumps({
                        "type": "audio_window",
                        "callSessionId": session_id,
                        "sequenceNumber": seq,
                        "windowStartMs": start_ms,
                        "windowEndMs": end_ms,
                        "durationMs": int(dur_s * 1000),
                        "audio": _make_base64_pcm(dur_s),
                    }))
                    assert json.loads(ws.receive_text())["type"] == "detection_result"
                    assert json.loads(ws.receive_text())["type"] == "call_report_updated"

                # End call at 54s
                ws.send_text(json.dumps({"type": "call_end", "callSessionId": session_id}))
                assert json.loads(ws.receive_text())["type"] == "call_report_completed"
                assert json.loads(ws.receive_text())["type"] == "session_closed"

    # Verify TCED window creations
    assert "[TCED] [W01] WINDOW_CREATED range=00:00-00:10 duration=10s sequence=1" in caplog.text
    assert "[TCED] [W02] WINDOW_CREATED range=00:00-00:20 duration=20s sequence=2" in caplog.text
    assert "[TCED] [W03] WINDOW_CREATED range=00:00-00:30 duration=30s sequence=3" in caplog.text
    assert "[TCED] [W04] WINDOW_CREATED range=00:10-00:40 duration=30s sequence=4" in caplog.text
    assert "[TCED] [W05] WINDOW_CREATED range=00:20-00:50 duration=30s sequence=5" in caplog.text
    assert "[TCED] [W06] WINDOW_CREATED range=00:24-00:54 duration=30s sequence=6" in caplog.text

    # Verify Completion & History
    assert "[TCED] CALL_ENDED duration=54s" in caplog.text
    assert "[CALL] COMPLETED duration=54s windows=6" in caplog.text
    assert "[CALL] HISTORY W01=BONAFIDE(0.98) W02=BONAFIDE(0.98) W03=BONAFIDE(0.98) W04=BONAFIDE(0.98) W05=BONAFIDE(0.98) W06=BONAFIDE(0.98)" in caplog.text

