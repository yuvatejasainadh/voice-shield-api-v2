from __future__ import annotations

import asyncio
import base64
import io
import json
import logging
import time
import wave
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings
from app.db.database import Base, engine
from app.main import app
from app.services.detection_decision_engine import DetectionDecisionEngine
from app.services.realtime_session_manager import RealtimeSessionManager


def _make_pcm_bytes(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate raw 16-bit mono PCM bytes."""
    sample_count = int(duration_seconds * sample_rate)
    return b"\x00\x00" * sample_count


def _make_base64_pcm(duration_seconds: float = 1.0) -> str:
    return base64.b64encode(_make_pcm_bytes(duration_seconds)).decode("utf-8")


@pytest.fixture(autouse=True)
def cleanup_sessions():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    RealtimeSessionManager.get_instance().clear_all()
    yield
    RealtimeSessionManager.get_instance().clear_all()


# ── DecisionEngine Tests ──────────────────────────────────────────────────

def test_decision_engine_cumulative_windows_0_10_20_30():
    """Test standard cumulative window evaluation (0-10s, 0-20s, 0-30s)."""
    engine = DetectionDecisionEngine(
        spoof_threshold=0.70,
        genuine_threshold=0.25,
        primary_decision_window_seconds=30,
    )

    # 1. 0-10s window (bonafide)
    out1 = engine.add_observation(
        sequence_number=1,
        window_start_ms=0,
        window_end_ms=10000,
        raw_result="bonafide",
        normalized_result="REAL",
        score=0.08,
        confidence=0.95,
    )
    assert out1.decision_state == "observing"
    assert out1.user_label == "ANALYZING"
    assert out1.notification_required is False

    # 2. 0-20s window (bonafide)
    out2 = engine.add_observation(
        sequence_number=2,
        window_start_ms=0,
        window_end_ms=20000,
        raw_result="bonafide",
        normalized_result="REAL",
        score=0.10,
        confidence=0.96,
    )
    assert out2.decision_state == "observing"

    # 3. 0-30s window (bonafide reaching 30s)
    out3 = engine.add_observation(
        sequence_number=3,
        window_start_ms=0,
        window_end_ms=30000,
        raw_result="bonafide",
        normalized_result="REAL",
        score=0.05,
        confidence=0.98,
    )
    assert out3.decision_state == "confirmed"
    assert out3.user_label == "SAFE / LOW RISK"
    assert out3.call_score < 0.25


def test_decision_engine_mixed_windows_partially_spoofed():
    """Test mixed call behavior: bonafide -> spoofed -> bonafide yields PARTIALLY_SPOOFED."""
    engine = DetectionDecisionEngine(
        spoof_threshold=0.70,
        early_spoof_threshold=0.90,
        min_windows_for_confirmation=2,
    )

    # 0-10s: bonafide
    out1 = engine.add_observation(
        sequence_number=1,
        window_start_ms=0,
        window_end_ms=10000,
        raw_result="bonafide",
        normalized_result="REAL",
        score=0.05,
        confidence=0.95,
    )
    assert out1.decision_state == "observing"

    # 0-20s: partially_spoofed (score 0.72) -> transitions to suspicious / SPOOFED
    out2 = engine.add_observation(
        sequence_number=2,
        window_start_ms=0,
        window_end_ms=20000,
        raw_result="partially_spoofed",
        normalized_result="SPOOFED",
        score=0.72,
        confidence=0.80,
    )
    assert out2.decision_state == "suspicious"
    assert out2.user_label == "SPOOFED"

    # 0-30s: confirms spoof with repeated evidence (score 0.85) -> now confirmed
    out3 = engine.add_observation(
        sequence_number=3,
        window_start_ms=0,
        window_end_ms=30000,
        raw_result="spoofed",
        normalized_result="SPOOFED",
        score=0.85,
        confidence=0.95,
    )
    assert out3.decision_state == "confirmed"
    assert out3.user_label == "VOICE CLONING DETECTED"
    assert out3.notification_required is True



def test_decision_engine_early_spoof_detection():
    """Test that a strong, high-confidence spoof at 0-10s produces an early confirmed warning."""
    engine = DetectionDecisionEngine(
        early_spoof_threshold=0.85,
        min_confidence=0.60,
        allow_early_detection=True,
    )

    out = engine.add_observation(
        sequence_number=1,
        window_start_ms=0,
        window_end_ms=10000,
        raw_result="spoofed",
        normalized_result="SPOOFED",
        score=0.96,
        confidence=0.99,
    )
    assert out.decision_state == "confirmed"
    assert out.user_label == "VOICE CLONING DETECTED"
    assert out.is_early_detection is True
    assert out.notification_required is True



def test_decision_engine_uncertain_provider_error():
    """Test handling when provider returns UNKNOWN / error."""
    engine = DetectionDecisionEngine()
    out = engine.add_observation(
        sequence_number=1,
        window_start_ms=0,
        window_end_ms=10000,
        raw_result="unknown",
        normalized_result="UNKNOWN",
        score=0.45,
        confidence=0.0,
    )
    assert out.decision_state == "uncertain"
    assert out.user_label == "INSUFFICIENT EVIDENCE"
    assert out.notification_required is False


# ── Session Manager & Call Report Lifecycle Tests ─────────────────────────

@pytest.mark.anyio
async def test_session_report_creation_and_finalization():
    manager = RealtimeSessionManager()
    session_id = "test-report-session-1"

    session = await manager.create_session(session_id)
    assert session.status == "ACTIVE"
    report = session.get_report_schema()
    assert report.callSessionId == session_id
    assert report.status == "ACTIVE"
    assert report.chunksAnalyzed == 0
    assert report.finalClassification == "ANALYZING"
    assert len(report.evidence) == 0

    # Add evidence
    session.decision_engine.add_observation(
        sequence_number=1,
        window_start_ms=0,
        window_end_ms=10000,
        raw_result="bonafide",
        normalized_result="REAL",
        score=0.05,
        confidence=0.98,
    )
    manager.sync_report_to_db(session_id)
    upd_report = session.get_report_schema()
    assert upd_report.chunksAnalyzed == 1
    assert len(upd_report.evidence) == 1
    assert upd_report.evidence[0].windowId == "W01"

    # Close session
    closed = await manager.close_session(session_id)
    assert closed.status == "CLOSED"
    final_report = closed.get_report_schema()
    assert final_report.status == "COMPLETED"
    assert final_report.endedAt is not None


@pytest.mark.anyio
async def test_session_isolation_delayed_call_a_ignored_in_call_b():
    """Verify that late results from closed Call A are never attributed to Call B."""
    manager = RealtimeSessionManager()
    call_a = "call-A"
    call_b = "call-B"

    await manager.create_session(call_a)
    await manager.close_session(call_a)

    await manager.create_session(call_b)

    assert manager.is_session_active(call_a) is False
    assert manager.is_session_active(call_b) is True


# ── WebSocket End-to-End Report Integration Tests ─────────────────────────

def test_websocket_call_start_creates_report_and_ends():
    with TestClient(app) as client:
        with client.websocket_connect("/api/v2/realtime/ws") as ws:
            # 1. Start call
            ws.send_text(json.dumps({
                "type": "call_start",
                "callSessionId": "ws-report-1",
                "timestamp": time.time() * 1000,
            }))
            resp1 = json.loads(ws.receive_text())
            assert resp1["type"] == "call_started"
            assert resp1["callSessionId"] == "ws-report-1"

            # Check call_report_created event
            resp2 = json.loads(ws.receive_text())
            assert resp2["type"] == "call_report_created"
            assert resp2["callSessionId"] == "ws-report-1"
            assert resp2["report"]["status"] == "ACTIVE"
            assert resp2["report"]["chunksAnalyzed"] == 0

            # 2. Ping / Pong
            ws.send_text(json.dumps({"type": "ping"}))
            pong = json.loads(ws.receive_text())
            assert pong["type"] == "pong"

            # 3. End call
            ws.send_text(json.dumps({
                "type": "call_end",
                "callSessionId": "ws-report-1",
                "timestamp": time.time() * 1000,
            }))
            resp_completed = json.loads(ws.receive_text())
            assert resp_completed["type"] == "call_report_completed"
            assert resp_completed["report"]["status"] == "COMPLETED"

            resp_end = json.loads(ws.receive_text())
            assert resp_end["type"] == "session_closed"


def test_websocket_audio_window_emits_report_updated_and_persists():
    with TestClient(app) as client:
        with client.websocket_connect("/api/v2/realtime/ws") as ws:
            session_id = "ws-session-report-pipeline"

            # Start call
            ws.send_text(json.dumps({
                "type": "call_start",
                "callSessionId": session_id,
            }))
            assert json.loads(ws.receive_text())["type"] == "call_started"
            assert json.loads(ws.receive_text())["type"] == "call_report_created"

            mock_aurigin_result = {
                "provider": "aurigin",
                "prediction_id": "aurigin-pred-1",
                "result": "REAL",
                "score": 0.05,
                "confidence": 0.98,
                "risk_score": 5,
                "segments": [],
                "processing_ms": 45,
                "raw_provider_status": "PROCESSED",
                "raw_result": "bonafide",
                "reason": "Authentic voice sample",
            }

            with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin_result):
                # Send 0-10s audio window
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": session_id,
                    "sequenceNumber": 1,
                    "windowStartMs": 0,
                    "windowEndMs": 10000,
                    "audioFormat": {
                        "sampleRate": 16000,
                        "channels": 1,
                        "encoding": "pcm_s16le",
                    },
                    "audio": _make_base64_pcm(10.0),
                    "clientTimestampMs": time.time() * 1000,
                }))

                # 1. Detection result message
                result_msg = json.loads(ws.receive_text())
                assert result_msg["type"] == "detection_result"
                assert result_msg["callSessionId"] == session_id
                assert result_msg["sequenceNumber"] == 1

                # 2. Call report updated message
                report_upd = json.loads(ws.receive_text())
                assert report_upd["type"] == "call_report_updated"
                assert report_upd["callSessionId"] == session_id
                assert report_upd["report"]["chunksAnalyzed"] == 1
                assert report_upd["latestWindow"]["sequenceNumber"] == 1
                assert report_upd["latestWindow"]["rawClassification"] == "bonafide"

            # 3. End call
            ws.send_text(json.dumps({"type": "call_end", "callSessionId": session_id}))
            assert json.loads(ws.receive_text())["type"] == "call_report_completed"
            assert json.loads(ws.receive_text())["type"] == "session_closed"

        # 4. Verify REST Reports API returns the persisted report
        resp = client.get(f"/api/v2/reports/{session_id}")
        assert resp.status_code == 200, resp.text
        report_data = resp.json()
        assert report_data["callSessionId"] == session_id
        assert report_data["status"] == "COMPLETED"
        assert report_data["chunksAnalyzed"] == 1
        assert len(report_data["evidence"]) == 1
        assert report_data["evidence"][0]["rawClassification"] == "bonafide"


def test_reports_list_api_pagination():
    with TestClient(app) as client:
        # Create 2 sessions
        for i in range(1, 3):
            s_id = f"report-list-test-{i}"
            with client.websocket_connect("/api/v2/realtime/ws") as ws:
                ws.send_text(json.dumps({"type": "call_start", "callSessionId": s_id}))
                ws.receive_text()
                ws.receive_text()
                ws.send_text(json.dumps({"type": "call_end", "callSessionId": s_id}))
                ws.receive_text()
                ws.receive_text()

        # Query reports list
        list_resp = client.get("/api/v2/reports?page=1&limit=10")
        assert list_resp.status_code == 200
        data = list_resp.json()
        assert data["total"] >= 2
        assert len(data["items"]) >= 2
        session_ids = [item["callSessionId"] for item in data["items"]]
        assert "report-list-test-1" in session_ids
        assert "report-list-test-2" in session_ids


# ── Reliability & Disconnect Resilience Tests ──────────────────────────────

def test_websocket_stays_alive_during_silence():
    """TEST 7: WebSocket remains alive during an active call; audio silence / no message must NOT close session."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v2/realtime/ws") as ws:
            session_id = "ws-silence-session-test"
            ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
            assert json.loads(ws.receive_text())["type"] == "call_started"
            assert json.loads(ws.receive_text())["type"] == "call_report_created"

            # Ping/Pong while waiting with no audio
            ws.send_text(json.dumps({"type": "ping"}))
            pong = json.loads(ws.receive_text())
            assert pong["type"] == "pong"

            # Session is still ACTIVE
            mgr_sess = RealtimeSessionManager.get_instance().get_session(session_id)
            assert mgr_sess is not None
            assert mgr_sess.status == "ACTIVE"


def test_single_aurigin_failure_does_not_close_websocket():
    """TEST 8: One Aurigin failure must NOT close the WebSocket or kill the session."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v2/realtime/ws") as ws:
            session_id = "ws-aurigin-err-session-test"
            ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
            assert json.loads(ws.receive_text())["type"] == "call_started"
            assert json.loads(ws.receive_text())["type"] == "call_report_created"

            # Send window with Aurigin throwing exception
            with patch("app.services.aurigin_service.AuriginService.analyze_audio", side_effect=RuntimeError("Aurigin 500 Network Error")):
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": session_id,
                    "sequenceNumber": 1,
                    "windowStartMs": 0,
                    "windowEndMs": 10000,
                    "audio": _make_base64_pcm(10.0),
                }))

                # Receives graceful detection_result with UNKNOWN / error
                res_msg = json.loads(ws.receive_text())
                assert res_msg["type"] == "detection_result"
                assert res_msg["result"] == "unknown"

                upd_msg = json.loads(ws.receive_text())
                assert upd_msg["type"] == "call_report_updated"

            # Subsequent ping/pong and next window work fine without socket closing
            ws.send_text(json.dumps({"type": "ping"}))
            assert json.loads(ws.receive_text())["type"] == "pong"

            mgr_sess = RealtimeSessionManager.get_instance().get_session(session_id)
            assert mgr_sess is not None
            assert mgr_sess.status in ("ACTIVE", "ANALYZING")


def test_db_persistence_failure_does_not_terminate_websocket():
    """TEST 9: DB sync failure does not crash the WebSocket or stop detection."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v2/realtime/ws") as ws:
            session_id = "ws-db-err-session-test"
            ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
            assert json.loads(ws.receive_text())["type"] == "call_started"
            assert json.loads(ws.receive_text())["type"] == "call_report_created"

            mock_aurigin = {
                "provider": "aurigin",
                "prediction_id": "pred-db-test",
                "result": "REAL",
                "score": 0.05,
                "confidence": 0.95,
                "risk_score": 5,
                "segments": [],
                "processing_ms": 30,
                "raw_provider_status": "OK",
            }

            with patch("app.services.realtime_session_manager.RealtimeSessionManager._sync_db_session", side_effect=Exception("DB connection dropped")):
                with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin):
                    ws.send_text(json.dumps({
                        "type": "audio_window",
                        "callSessionId": session_id,
                        "sequenceNumber": 1,
                        "windowStartMs": 0,
                        "windowEndMs": 10000,
                        "audio": _make_base64_pcm(10.0),
                    }))

                    res_msg = json.loads(ws.receive_text())
                    assert res_msg["type"] == "detection_result"
                    assert res_msg["score"] < 0.25

                    upd_msg = json.loads(ws.receive_text())
                    assert upd_msg["type"] == "call_report_updated"


@pytest.mark.anyio
async def test_websocket_disconnect_awaits_in_flight_tasks():
    """TEST 10: WebSocket disconnect gracefully preserves and awaits in-flight upload task."""
    mgr = RealtimeSessionManager.get_instance()
    session_id = "test-disconnect-in-flight"
    sess = await mgr.create_session(session_id)

    async def _mock_in_flight():
        await asyncio.sleep(0.1)
        sess.decision_engine.add_observation(
            sequence_number=1,
            window_start_ms=0,
            window_end_ms=10000,
            raw_result="bonafide",
            normalized_result="REAL",
            score=0.05,
            confidence=0.95,
        )

    task = asyncio.create_task(_mock_in_flight())
    mgr.register_task(session_id, task)

    # Disconnect occurs while task is still running
    await mgr.handle_client_disconnect(session_id)

    # Task was awaited and evidence preserved
    assert len(sess.decision_engine.observations) == 1
    assert sess.status == "INTERRUPTED"
    report = sess.get_report_schema()
    assert report.chunksAnalyzed == 1


# ── Production Regression Tests (Android Direct Window Ingestion) ────────────

def test_realistic_58s_streaming_pipeline():
    """Regression Test 1: Realistic 58-second call produces exactly 6 detector windows with final duration ≈ 58s."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v2/realtime/ws") as ws:
            session_id = "test-58s-pipeline"

            # 1. Start call
            ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
            assert json.loads(ws.receive_text())["type"] == "call_started"
            assert json.loads(ws.receive_text())["type"] == "call_report_created"

            mock_aurigin = {
                "provider": "aurigin",
                "prediction_id": "pred-58s",
                "result": "REAL",
                "score": 0.05,
                "confidence": 0.96,
                "raw_result": "bonafide",
                "raw_provider_status": "OK",
            }

            windows_data = [
                (1, 0, 10000, 10.0),      # W001: 00-10s
                (2, 0, 20000, 20.0),      # W002: 00-20s
                (3, 0, 30000, 30.0),      # W003: 00-30s
                (4, 10000, 40000, 30.0),  # W004: 10-40s
                (5, 20000, 50000, 30.0),  # W005: 20-50s
                (6, 30000, 58000, 28.0),  # W006: 30-58s
            ]

            with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin):
                for seq, start_ms, end_ms, dur_s in windows_data:
                    ws.send_text(json.dumps({
                        "type": "audio_window",
                        "callSessionId": session_id,
                        "sequenceNumber": seq,
                        "windowStartMs": start_ms,
                        "windowEndMs": end_ms,
                        "durationMs": int(dur_s * 1000),
                        "audio": _make_base64_pcm(dur_s),
                    }))
                    res_msg = json.loads(ws.receive_text())
                    assert res_msg["type"] == "detection_result"
                    assert res_msg["sequenceNumber"] == seq
                    assert res_msg["windowStartMs"] == start_ms
                    assert res_msg["windowEndMs"] == end_ms
                    upd_msg = json.loads(ws.receive_text())
                    assert upd_msg["type"] == "call_report_updated"

                # 2. End call -> finalizes session without creating artificial backend tails
                ws.send_text(json.dumps({"type": "call_end", "callSessionId": session_id}))

                # Final completion messages
                comp_msg = json.loads(ws.receive_text())
                assert comp_msg["type"] == "call_report_completed"
                final_rep = comp_msg["report"]
                assert final_rep["chunksAnalyzed"] == 6
                assert final_rep["totalDurationSeconds"] == 58.0
                assert len(final_rep["evidence"]) == 6

                # Verify exact window sequences matching Android's timeline
                ranges = [(e["windowStartMs"], e["windowEndMs"]) for e in final_rep["evidence"]]
                assert ranges == [
                    (0, 10000),
                    (0, 20000),
                    (0, 30000),
                    (10000, 40000),
                    (20000, 50000),
                    (30000, 58000),
                ]

                closed_msg = json.loads(ws.receive_text())
                assert closed_msg["type"] == "session_closed"


def test_realistic_67s_streaming_pipeline():
    """Regression Test 2: Realistic 67-second call generates exactly 7 windows matching Android timeline."""
    with TestClient(app) as client:
        with client.websocket_connect("/api/v2/realtime/ws") as ws:
            session_id = "test-67s-pipeline"

            # 1. Start call
            ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
            assert json.loads(ws.receive_text())["type"] == "call_started"
            assert json.loads(ws.receive_text())["type"] == "call_report_created"

            mock_aurigin = {
                "provider": "aurigin",
                "prediction_id": "pred-67s",
                "result": "REAL",
                "score": 0.05,
                "confidence": 0.96,
                "raw_result": "bonafide",
                "raw_provider_status": "OK",
            }

            windows_data = [
                (1, 0, 10000, 10.0),      # W001: 00-10s
                (2, 0, 20000, 20.0),      # W002: 00-20s
                (3, 0, 30000, 30.0),      # W003: 00-30s
                (4, 10000, 40000, 30.0),  # W004: 10-40s
                (5, 20000, 50000, 30.0),  # W005: 20-50s
                (6, 30000, 60000, 30.0),  # W006: 30-60s
                (7, 37000, 67000, 30.0),  # W007: 37-67s
            ]

            with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin):
                for seq, start_ms, end_ms, dur_s in windows_data:
                    ws.send_text(json.dumps({
                        "type": "audio_window",
                        "callSessionId": session_id,
                        "sequenceNumber": seq,
                        "windowStartMs": start_ms,
                        "windowEndMs": end_ms,
                        "durationMs": int(dur_s * 1000),
                        "audio": _make_base64_pcm(dur_s),
                    }))
                    res_msg = json.loads(ws.receive_text())
                    assert res_msg["type"] == "detection_result"
                    assert res_msg["sequenceNumber"] == seq
                    upd_msg = json.loads(ws.receive_text())
                    assert upd_msg["type"] == "call_report_updated"

                # 2. End call
                ws.send_text(json.dumps({"type": "call_end", "callSessionId": session_id}))

                # Final completion messages
                comp_msg = json.loads(ws.receive_text())
                assert comp_msg["type"] == "call_report_completed"
                final_rep = comp_msg["report"]
                assert final_rep["chunksAnalyzed"] == 7
                assert final_rep["totalDurationSeconds"] == 67.0
                assert len(final_rep["evidence"]) == 7

                ranges = [(e["windowStartMs"], e["windowEndMs"]) for e in final_rep["evidence"]]
                assert ranges == [
                    (0, 10000),
                    (0, 20000),
                    (0, 30000),
                    (10000, 40000),
                    (20000, 50000),
                    (30000, 60000),
                    (37000, 67000),
                ]

                closed_msg = json.loads(ws.receive_text())
                assert closed_msg["type"] == "session_closed"


def test_disconnect_without_call_end_triggers_finalization():
    """Regression Test 3: Sudden client disconnect without call_end executes full finalization using Android timeline."""
    session_id = "test-disconnect-finalization"
    mock_aurigin = {
        "provider": "aurigin",
        "prediction_id": "pred-disc",
        "result": "REAL",
        "score": 0.05,
        "confidence": 0.95,
        "raw_result": "bonafide",
        "raw_provider_status": "OK",
    }
    windows_data = [
        (1, 0, 10000, 10.0),
        (2, 0, 20000, 20.0),
        (3, 0, 30000, 30.0),
        (4, 10000, 40000, 30.0),
        (5, 20000, 50000, 30.0),
        (6, 30000, 60000, 30.0),
        (7, 37000, 67000, 30.0),
    ]
    with TestClient(app) as client:
        with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin):
            with client.websocket_connect("/api/v2/realtime/ws") as ws:
                ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
                assert json.loads(ws.receive_text())["type"] == "call_started"
                assert json.loads(ws.receive_text())["type"] == "call_report_created"

                for seq, start_ms, end_ms, dur_s in windows_data:
                    ws.send_text(json.dumps({
                        "type": "audio_window",
                        "callSessionId": session_id,
                        "sequenceNumber": seq,
                        "windowStartMs": start_ms,
                        "windowEndMs": end_ms,
                        "durationMs": int(dur_s * 1000),
                        "audio": _make_base64_pcm(dur_s),
                    }))
                    json.loads(ws.receive_text())
                    json.loads(ws.receive_text())

                # Client abruptly closes socket without sending call_end
                ws.close()

        # Post-disconnect verification
        sess = RealtimeSessionManager.get_instance().get_session(session_id)
        assert sess is not None
        assert sess.status == "CLOSED"
        report = sess.get_report_schema()
        assert report.status == "COMPLETED"
        assert report.chunksAnalyzed == 7
        assert report.totalDurationSeconds == 67.0
        assert len(report.evidence) == 7
        assert report.evidence[-1].windowStartMs == 37000
        assert report.evidence[-1].windowEndMs == 67000

        # REST report query returns finalized report
        resp = client.get(f"/api/v2/reports/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "COMPLETED"
        assert data["chunksAnalyzed"] == 7


def test_duplicate_finalization_idempotency():
    """Regression Test 4: call_end followed by WebSocketDisconnect executes finalization exactly once."""
    session_id = "test-idempotent-finalization"
    with TestClient(app) as client:
        with client.websocket_connect("/api/v2/realtime/ws") as ws:
            ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
            assert json.loads(ws.receive_text())["type"] == "call_started"
            assert json.loads(ws.receive_text())["type"] == "call_report_created"

            mock_aurigin = {
                "provider": "aurigin",
                "prediction_id": "pred-idem",
                "result": "REAL",
                "score": 0.05,
                "confidence": 0.95,
                "raw_result": "bonafide",
                "raw_provider_status": "OK",
            }

            windows_data = [
                (1, 0, 10000, 10.0),
                (2, 0, 20000, 20.0),
                (3, 0, 30000, 30.0),
            ]

            with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin):
                for seq, start_ms, end_ms, dur_s in windows_data:
                    ws.send_text(json.dumps({
                        "type": "audio_window",
                        "callSessionId": session_id,
                        "sequenceNumber": seq,
                        "windowStartMs": start_ms,
                        "windowEndMs": end_ms,
                        "durationMs": int(dur_s * 1000),
                        "audio": _make_base64_pcm(dur_s),
                    }))
                    json.loads(ws.receive_text())
                    json.loads(ws.receive_text())

                # 1. First finalization via call_end
                ws.send_text(json.dumps({"type": "call_end", "callSessionId": session_id}))
                comp_msg = json.loads(ws.receive_text())
                assert comp_msg["type"] == "call_report_completed"
                assert comp_msg["report"]["chunksAnalyzed"] == 3
                assert json.loads(ws.receive_text())["type"] == "session_closed"

                # 2. Second finalization triggered by socket disconnect
        
        sess = RealtimeSessionManager.get_instance().get_session(session_id)
        assert sess is not None
        assert sess.status == "CLOSED"
        assert len(sess.decision_engine.observations) == 3


def test_duplicate_window_rejected_idempotent():
    """Regression Test 5: Re-sending the same Android window sequence is deduplicated (1 detector invocation)."""
    session_id = "test-duplicate-dedupe"
    mock_aurigin = {
        "provider": "aurigin",
        "prediction_id": "pred-dup",
        "result": "REAL",
        "score": 0.05,
        "confidence": 0.95,
        "raw_result": "bonafide",
        "raw_provider_status": "OK",
    }
    with TestClient(app) as client:
        with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin) as mock_analyze:
            with client.websocket_connect("/api/v2/realtime/ws") as ws:
                ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
                assert json.loads(ws.receive_text())["type"] == "call_started"
                assert json.loads(ws.receive_text())["type"] == "call_report_created"

                # Send Window 1
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": session_id,
                    "sequenceNumber": 1,
                    "windowStartMs": 0,
                    "windowEndMs": 10000,
                    "durationMs": 10000,
                    "audio": _make_base64_pcm(10.0),
                }))
                res1 = json.loads(ws.receive_text())
                assert res1["type"] == "detection_result"
                assert res1["sequenceNumber"] == 1
                upd1 = json.loads(ws.receive_text())
                assert upd1["type"] == "call_report_updated"

                # Send duplicate Window 1
                ws.send_text(json.dumps({
                    "type": "audio_window",
                    "callSessionId": session_id,
                    "sequenceNumber": 1,
                    "windowStartMs": 0,
                    "windowEndMs": 10000,
                    "durationMs": 10000,
                    "audio": _make_base64_pcm(10.0),
                }))

                # Only 1 call to Aurigin was made
                assert mock_analyze.call_count == 1


def test_call_end_exits_receive_loop_cleanly_without_exception(caplog):
    """
    Regression Test 6: Verifies that after call_end finalization, the WebSocket receive loop exits cleanly,
    finalization occurs exactly once, no RuntimeError is raised, no reason=server_exception is logged,
    and the final report is properly persisted.
    """
    session_id = "test-clean-loop-exit"
    mock_aurigin = {
        "provider": "aurigin",
        "prediction_id": "pred-clean-exit",
        "result": "REAL",
        "score": 0.05,
        "confidence": 0.96,
        "raw_result": "bonafide",
        "raw_provider_status": "OK",
    }
    with caplog.at_level(logging.INFO):
        with patch("app.services.aurigin_service.AuriginService.analyze_audio", new_callable=AsyncMock, return_value=mock_aurigin):
            with TestClient(app) as client:
                with client.websocket_connect("/api/v2/realtime/ws") as ws:
                    # 1. Start call
                    ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
                    assert json.loads(ws.receive_text())["type"] == "call_started"
                    assert json.loads(ws.receive_text())["type"] == "call_report_created"

                    # 2. Send 1 window
                    ws.send_text(json.dumps({
                        "type": "audio_window",
                        "callSessionId": session_id,
                        "sequenceNumber": 1,
                        "windowStartMs": 0,
                        "windowEndMs": 10000,
                        "durationMs": 10000,
                        "audio": _make_base64_pcm(10.0),
                    }))
                    assert json.loads(ws.receive_text())["type"] == "detection_result"
                    assert json.loads(ws.receive_text())["type"] == "call_report_updated"

                    # 3. Send call_end -> server finalizes and breaks out of loop
                    ws.send_text(json.dumps({"type": "call_end", "callSessionId": session_id}))

                    comp_msg = json.loads(ws.receive_text())
                    assert comp_msg["type"] == "call_report_completed"
                    assert comp_msg["report"]["status"] == "COMPLETED"
                    assert comp_msg["report"]["chunksAnalyzed"] == 1

                    closed_msg = json.loads(ws.receive_text())
                    assert closed_msg["type"] == "session_closed"

    # Verify session and DB report state
    sess = RealtimeSessionManager.get_instance().get_session(session_id)
    assert sess is not None
    assert sess.status == "CLOSED"
    assert len(sess.decision_engine.observations) == 1

    # Verify NO server_exception was logged
    log_text = caplog.text
    assert "reason=server_exception" not in log_text
    assert "Need to call \"accept\" first" not in log_text




