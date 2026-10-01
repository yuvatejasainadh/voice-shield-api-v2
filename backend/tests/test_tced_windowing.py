from __future__ import annotations

import base64
import json
import time
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings, get_settings
from app.db.database import Base, engine
from app.main import app
from app.services.realtime_session_manager import RealtimeSessionManager
from app.services.tced_window_manager import TCEDConfig, TCEDWindow, TCEDWindowManager


def _make_pcm_bytes(duration_seconds: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate raw 16-bit mono PCM bytes."""
    sample_count = int(duration_seconds * sample_rate)
    return b"\x00\x00" * sample_count


def _make_base64_pcm(duration_seconds: float = 1.0, sample_rate: int = 16000) -> str:
    return base64.b64encode(_make_pcm_bytes(duration_seconds, sample_rate)).decode("utf-8")


@pytest.fixture(autouse=True)
def cleanup_sessions():
    Base.metadata.create_all(bind=engine)
    RealtimeSessionManager.get_instance().clear_all()
    yield
    RealtimeSessionManager.get_instance().clear_all()


# ── Standalone TCED Window Manager Tests ────────────────────────────────────

def test_tced_test_a_54_second_call():
    """
    Test A — 54-second call:
    Expected windows:
      W01  00–10 s (Accumulation)
      W02  00–20 s (Accumulation)
      W03  00–30 s (Accumulation / Max Window)
      W04  10–40 s (Sliding Window)
      W05  20–50 s (Sliding Window)
      W06  24–54 s [TAIL] (Tail-aligned terminal window: 54 - 30 = 24)
    """
    mgr = TCEDWindowManager(
        call_session_id="test-54s",
        config=TCEDConfig(
            new_audio_interval_seconds=10.0,
            window_stride_seconds=20.0,
            max_window_length_seconds=30.0,
            sample_rate=16000,
        ),
    )

    # Ingest 5 chunks of 10s audio (total 50s)
    all_new_windows: list[TCEDWindow] = []
    for _ in range(5):
        audio_10s = _make_pcm_bytes(10.0)
        wins = mgr.add_audio(audio_10s)
        all_new_windows.extend(wins)

    assert len(all_new_windows) == 5

    # W01
    assert all_new_windows[0].window_id == "W01"
    assert all_new_windows[0].sequence == 1
    assert all_new_windows[0].window_start_ms == 0
    assert all_new_windows[0].window_end_ms == 10000
    assert all_new_windows[0].duration_ms == 10000
    assert all_new_windows[0].is_tail is False

    # W02
    assert all_new_windows[1].window_id == "W02"
    assert all_new_windows[1].sequence == 2
    assert all_new_windows[1].window_start_ms == 0
    assert all_new_windows[1].window_end_ms == 20000
    assert all_new_windows[1].duration_ms == 20000
    assert all_new_windows[1].is_tail is False

    # W03
    assert all_new_windows[2].window_id == "W03"
    assert all_new_windows[2].sequence == 3
    assert all_new_windows[2].window_start_ms == 0
    assert all_new_windows[2].window_end_ms == 30000
    assert all_new_windows[2].duration_ms == 30000
    assert all_new_windows[2].is_tail is False

    # W04
    assert all_new_windows[3].window_id == "W04"
    assert all_new_windows[3].sequence == 4
    assert all_new_windows[3].window_start_ms == 10000
    assert all_new_windows[3].window_end_ms == 40000
    assert all_new_windows[3].duration_ms == 30000
    assert all_new_windows[3].is_tail is False

    # W05
    assert all_new_windows[4].window_id == "W05"
    assert all_new_windows[4].sequence == 5
    assert all_new_windows[4].window_start_ms == 20000
    assert all_new_windows[4].window_end_ms == 50000
    assert all_new_windows[4].duration_ms == 30000
    assert all_new_windows[4].is_tail is False

    # Final 4s chunk arriving before call terminates at 54s
    audio_4s = _make_pcm_bytes(4.0)
    extra_wins = mgr.add_audio(audio_4s)
    assert len(extra_wins) == 0  # 54s has not hit 60s regular boundary

    # Finalize call at 54s
    tail_win = mgr.finalize_call(54000)
    assert tail_win is not None
    assert tail_win.window_id == "W06"
    assert tail_win.sequence == 6
    assert tail_win.window_start_ms == 24000  # 54 - 30 = 24
    assert tail_win.window_end_ms == 54000
    assert tail_win.duration_ms == 30000
    assert tail_win.is_tail is True

    # Total windows generated is exactly 6
    assert len(mgr.generated_windows) == 6


def test_tced_test_b_36_second_call():
    """
    Test B — 36-second call (Legacy 10s-30s TCED config):
    Expected windows:
      W01  00–10 s
      W02  00–20 s
      W03  00–30 s
      W04  06–36 s [TAIL] (Tail-aligned terminal window: 36 - 30 = 6)
    """
    legacy_cfg = TCEDConfig(new_audio_interval_seconds=10.0, window_stride_seconds=20.0, max_window_length_seconds=30.0, sample_rate=16000)
    mgr = TCEDWindowManager(call_session_id="test-36s", config=legacy_cfg)

    # Ingest 3 chunks of 10s (total 30s) + 1 chunk of 6s (total 36s)
    for _ in range(3):
        mgr.add_audio(_make_pcm_bytes(10.0))
    mgr.add_audio(_make_pcm_bytes(6.0))

    assert len(mgr.generated_windows) == 3

    # Finalize call at 36s
    tail_win = mgr.finalize_call()
    assert tail_win is not None
    assert tail_win.window_id == "W04"
    assert tail_win.sequence == 4
    assert tail_win.window_start_ms == 6000  # 36 - 30 = 6
    assert tail_win.window_end_ms == 36000
    assert tail_win.duration_ms == 30000
    assert tail_win.is_tail is True

    assert len(mgr.generated_windows) == 4


def test_tced_test_c_30_second_call_suppresses_duplicate_tail():
    """
    Test C — 30-second call (Legacy 10s-30s TCED config):
    Expected windows:
      W01  00–10 s
      W02  00–20 s
      W03  00–30 s
    Do NOT create an unnecessary duplicate 0–30 terminal window if 0–30 has already been analyzed.
    """
    legacy_cfg = TCEDConfig(new_audio_interval_seconds=10.0, window_stride_seconds=20.0, max_window_length_seconds=30.0, sample_rate=16000)
    mgr = TCEDWindowManager(call_session_id="test-30s", config=legacy_cfg)

    # Ingest 30s of audio
    for _ in range(3):
        mgr.add_audio(_make_pcm_bytes(10.0))

    assert len(mgr.generated_windows) == 3
    assert mgr.generated_windows[2].window_start_ms == 0
    assert mgr.generated_windows[2].window_end_ms == 30000

    # Finalize call at 30s
    tail_win = mgr.finalize_call(30000)
    assert tail_win is None  # Suppressed duplicate!

    assert len(mgr.generated_windows) == 3


def test_tced_test_d_short_call_under_10s():
    """
    Test D — short call (<10 s, e.g. 7 seconds with legacy 10s interval):
    Generate one terminal window covering the available call:
      W01  00–07 s [TAIL]
    Do not fabricate missing audio.
    """
    legacy_cfg = TCEDConfig(new_audio_interval_seconds=10.0, window_stride_seconds=20.0, max_window_length_seconds=30.0, sample_rate=16000)
    mgr = TCEDWindowManager(call_session_id="test-7s", config=legacy_cfg)

    # Ingest 7s of audio
    mgr.add_audio(_make_pcm_bytes(7.0))
    assert len(mgr.generated_windows) == 0  # No 10s interval reached

    # Finalize call at 7s
    tail_win = mgr.finalize_call(7000)
    assert tail_win is not None
    assert tail_win.window_id == "W01"
    assert tail_win.sequence == 1
    assert tail_win.window_start_ms == 0
    assert tail_win.window_end_ms == 7000
    assert tail_win.duration_ms == 7000
    assert tail_win.is_tail is True

    assert len(mgr.generated_windows) == 1


def test_tced_empty_call_generates_no_windows():
    mgr = TCEDWindowManager(call_session_id="test-empty")
    tail_win = mgr.finalize_call(0)
    assert tail_win is None
    assert len(mgr.generated_windows) == 0


def test_tced_default_synchronized_config_2_5s():
    """Verify default TCEDConfig aligns with synchronized contract: 2.5s interval/stride, 5.0s max length."""
    mgr = TCEDWindowManager(call_session_id="test-sync-default")
    assert mgr.config.new_audio_interval_seconds == 2.5
    assert mgr.config.window_stride_seconds == 2.5
    assert mgr.config.max_window_length_seconds == 5.0

    # 2.5s audio -> generates W01 (0-2.5s)
    w1 = mgr.add_audio(_make_pcm_bytes(2.5))
    assert len(w1) == 1
    assert w1[0].window_start_ms == 0
    assert w1[0].window_end_ms == 2500

    # Next 2.5s audio (total 5.0s) -> generates W02 (0-5.0s)
    w2 = mgr.add_audio(_make_pcm_bytes(2.5))
    assert len(w2) == 1
    assert w2[0].window_start_ms == 0
    assert w2[0].window_end_ms == 5000

    # Next 2.5s audio (total 7.5s) -> generates W03 (2.5-7.5s)
    w3 = mgr.add_audio(_make_pcm_bytes(2.5))
    assert len(w3) == 1
    assert w3[0].window_start_ms == 2500
    assert w3[0].window_end_ms == 7500


# ── WebSocket End-to-End TCED Pipeline Integration Test ───────────────────

def test_websocket_tced_54_second_call_pipeline():
    """
    Full WebSocket integration test for a 54-second call:
    Streams 6 Android windows (W01..W06 including terminal 24-54s), sends call_end at 54s,
    verifies all 6 windows are emitted and recorded in report.
    """
    mock_aurigin = {
        "provider": "aurigin",
        "prediction_id": "pred-tced",
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
                session_id = "ws-tced-54s-integration"

                # 1. Start call
                ws.send_text(json.dumps({"type": "call_start", "callSessionId": session_id}))
                assert json.loads(ws.receive_text())["type"] == "call_started"
                assert json.loads(ws.receive_text())["type"] == "call_report_created"

                windows_data = [
                    (1, 0, 10000, 10.0),      # W01: 00-10s
                    (2, 0, 20000, 20.0),      # W02: 00-20s
                    (3, 0, 30000, 30.0),      # W03: 00-30s
                    (4, 10000, 40000, 30.0),  # W04: 10-40s
                    (5, 20000, 50000, 30.0),  # W05: 20-50s
                    (6, 24000, 54000, 30.0),  # W06: 24-54s
                ]

                # Send 6 windows
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

                    # Expect detection_result
                    det_res = json.loads(ws.receive_text())
                    assert det_res["type"] == "detection_result"
                    assert det_res["sequenceNumber"] == seq
                    assert det_res["windowStartMs"] == start_ms
                    assert det_res["windowEndMs"] == end_ms

                    # Expect call_report_updated
                    rep_upd = json.loads(ws.receive_text())
                    assert rep_upd["type"] == "call_report_updated"
                    assert rep_upd["report"]["chunksAnalyzed"] == seq

                # End call at 54s
                ws.send_text(json.dumps({
                    "type": "call_end",
                    "callSessionId": session_id,
                }))

                # Expect final call_report_completed
                rep_comp = json.loads(ws.receive_text())
                assert rep_comp["type"] == "call_report_completed"
                assert rep_comp["report"]["status"] == "COMPLETED"
                assert rep_comp["report"]["chunksAnalyzed"] == 6
                assert rep_comp["report"]["totalDurationSeconds"] == 54.0
                assert len(rep_comp["report"]["evidence"]) == 6

                # Check evidence timeline
                evidence = rep_comp["report"]["evidence"]
                assert evidence[0]["windowStartMs"] == 0 and evidence[0]["windowEndMs"] == 10000
                assert evidence[1]["windowStartMs"] == 0 and evidence[1]["windowEndMs"] == 20000
                assert evidence[2]["windowStartMs"] == 0 and evidence[2]["windowEndMs"] == 30000
                assert evidence[3]["windowStartMs"] == 10000 and evidence[3]["windowEndMs"] == 40000
                assert evidence[4]["windowStartMs"] == 20000 and evidence[4]["windowEndMs"] == 50000
                assert evidence[5]["windowStartMs"] == 24000 and evidence[5]["windowEndMs"] == 54000

                # Expect session_closed
                closed_msg = json.loads(ws.receive_text())
                assert closed_msg["type"] == "session_closed"
