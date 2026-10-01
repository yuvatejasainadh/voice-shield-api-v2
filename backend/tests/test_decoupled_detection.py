"""Tests verifying complete decoupling of transcription from the detection pipeline."""

from __future__ import annotations

import base64
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.db.database import Base, engine
from app.main import app
from app.schemas.realtime import CallReportSchema, WindowEvidenceSchema
from app.services.detection_decision_engine import DetectionDecisionEngine
from app.services.realtime_session_manager import RealtimeSessionManager


def _make_base64_pcm(duration_seconds: float = 10.0, sample_rate: int = 16000) -> str:
    sample_count = int(duration_seconds * sample_rate)
    raw_bytes = b"\x00\x00" * sample_count
    return base64.b64encode(raw_bytes).decode("utf-8")


@pytest.fixture(autouse=True)
def cleanup_db():
    Base.metadata.create_all(bind=engine)
    RealtimeSessionManager.get_instance().clear_all()
    yield
    RealtimeSessionManager.get_instance().clear_all()


def test_call_report_schema_contains_no_transcript_fields():
    """Verify that CallReportSchema and WindowEvidenceSchema do not contain transcript or STT fields."""
    report_fields = CallReportSchema.model_fields.keys()
    for forbidden in ["transcript", "transcription", "words", "speech_to_text", "stt", "language_transcript"]:
        assert forbidden not in report_fields, f"Forbidden field '{forbidden}' found in CallReportSchema"

    window_fields = WindowEvidenceSchema.model_fields.keys()
    for forbidden in ["transcript", "transcription", "words", "text"]:
        assert forbidden not in window_fields, f"Forbidden field '{forbidden}' found in WindowEvidenceSchema"


def test_decision_engine_operates_strictly_on_acoustic_scores():
    """Verify DecisionEngine accumulates acoustic observations without requiring any speech text."""
    engine = DetectionDecisionEngine(
        spoof_threshold=0.70,
        genuine_threshold=0.25,
        primary_decision_window_seconds=30,
    )

    out = engine.add_observation(
        sequence_number=1,
        window_start_ms=0,
        window_end_ms=10000,
        raw_result="bonafide",
        normalized_result="REAL",
        score=0.05,
        confidence=0.98,
        processing_ms=45,
        reason="Acoustic features authentic",
    )

    assert out.decision_state == "observing"
    assert out.call_score == 0.05
    evidence = engine.get_evidence_list()
    assert len(evidence) == 1
    assert "transcript" not in evidence[0]
    assert "text" not in evidence[0]


def test_realtime_websocket_pipeline_without_transcription():
    """End-to-end WebSocket test verifying audio windows produce reports with no transcription."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/realtime/ws") as ws:
            session_id = "decoupled-test-1"

            # 1. Start Call
            ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
            assert json.loads(ws.receive_text())["type"] == "call_started"
            
            created_evt = json.loads(ws.receive_text())
            assert created_evt["type"] == "call_report_created"
            assert "transcript" not in json.dumps(created_evt)

            # 2. Audio Window (Acoustic Deepfake Provider Mock)
            mock_aurigin = {
                "provider": "aurigin",
                "prediction_id": "aurigin-pred-decoupled",
                "result": "REAL",
                "score": 0.04,
                "confidence": 0.99,
                "risk_score": 4,
                "segments": [],
                "processing_ms": 35,
                "raw_provider_status": "PROCESSED",
                "raw_result": "bonafide",
                "reason": "Authentic acoustic characteristics",
            }

            with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin):
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": session_id,
                    "sequenceNumber": 1,
                    "windowStartMs": 0,
                    "windowEndMs": 10000,
                    "audio": _make_base64_pcm(10.0),
                    "clientTimestampMs": time.time() * 1000,
                }))

                # Result & Report Update
                det_msg = json.loads(ws.receive_text())
                assert det_msg["type"] == "detection_result"
                assert "transcript" not in json.dumps(det_msg)

                upd_msg = json.loads(ws.receive_text())
                assert upd_msg["type"] == "call_report_updated"
                assert "transcript" not in json.dumps(upd_msg)

            # 3. End Call
            ws.send_text(json.dumps({"type": "call_end", "callSessionId": session_id}))
            comp_msg = json.loads(ws.receive_text())
            assert comp_msg["type"] == "call_report_completed"
            assert "transcript" not in json.dumps(comp_msg)

            assert json.loads(ws.receive_text())["type"] == "session_closed"


def test_partially_spoofed_preserves_raw_and_normalized_classification_in_evidence():
    """Test 5: A realtime window with Aurigin partially_spoofed is stored with rawClassification=partially_spoofed and normalizedClassification=SPOOFED."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v1/realtime/ws") as ws:
            session_id = "test-partially-spoofed-preservation"

            # 1. Start Call
            ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
            assert json.loads(ws.receive_text())["type"] == "call_started"
            assert json.loads(ws.receive_text())["type"] == "call_report_created"

            # 2. Mock Aurigin returning partially_spoofed
            mock_aurigin = {
                "provider": "aurigin",
                "prediction_id": "aurigin-pred-partial-1",
                "result": "SPOOFED",
                "score": 0.72,
                "confidence": 0.88,
                "risk_score": 72,
                "segments": [
                    {"start": 0.0, "end": 5.0, "result": "bonafide"},
                    {"start": 5.0, "end": 10.0, "result": "spoofed"},
                ],
                "processing_ms": 40,
                "raw_provider_status": "PROCESSED",
                "raw_result": "partially_spoofed",
                "reason": "Mixed genuine and spoofed speech",
            }

            with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin):
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": session_id,
                    "sequenceNumber": 1,
                    "windowStartMs": 0,
                    "windowEndMs": 10000,
                    "audio": _make_base64_pcm(10.0),
                    "clientTimestampMs": time.time() * 1000,
                }))

                det_msg = json.loads(ws.receive_text())
                assert det_msg["type"] == "detection_result"
                assert det_msg["result"] == "partially_spoofed"

                upd_msg = json.loads(ws.receive_text())
                assert upd_msg["type"] == "call_report_updated"
                assert upd_msg["latestWindow"]["rawClassification"] == "partially_spoofed"
                assert upd_msg["latestWindow"]["normalizedClassification"] == "SPOOFED"
                # Check for risk_update message emitted on spoof / risk escalation
                msg3 = json.loads(ws.receive_text())
                if msg3["type"] == "risk_update":
                    assert msg3["report"]["evidence"][0]["rawClassification"] == "partially_spoofed"
                    assert msg3["report"]["evidence"][0]["normalizedClassification"] == "SPOOFED"

            # 3. End Call
            ws.send_text(json.dumps({"type": "call_end", "callSessionId": session_id}))
            comp_msg = json.loads(ws.receive_text())
            assert comp_msg["type"] == "call_report_completed"
            assert comp_msg["report"]["evidence"][0]["rawClassification"] == "partially_spoofed"
            assert comp_msg["report"]["evidence"][0]["normalizedClassification"] == "SPOOFED"
            assert json.loads(ws.receive_text())["type"] == "session_closed"

        # 4. Verify REST report API
        resp = client.get(f"/api/v1/reports/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["evidence"][0]["rawClassification"] == "partially_spoofed"
        assert data["evidence"][0]["normalizedClassification"] == "SPOOFED"


