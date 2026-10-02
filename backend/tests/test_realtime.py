"""Tests for realtime session lifecycle, chunk ingestion, and status checks."""

from __future__ import annotations

import io
import uuid
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import CallChunk, CallSession


def test_create_realtime_session_and_idempotency(client: TestClient):
    client_sess_id = f"client_{uuid.uuid4().hex[:12]}"
    payload = {
        "client_session_id": client_sess_id,
        "started_at": "2026-10-01T12:00:00Z",
    }

    # 1. First creation
    res1 = client.post("/api/v2/realtime/sessions", json=payload)
    assert res1.status_code == 200
    data1 = res1.json()
    assert data1["session_id"] is not None
    assert data1["status"] == "ACTIVE"
    assert data1["risk_state"] == "LOW"

    # 2. Idempotent re-creation with same client_session_id
    res2 = client.post("/api/v2/realtime/sessions", json=payload)
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["session_id"] == data1["session_id"]
    assert data2["status"] == "ACTIVE"


def test_realtime_status_endpoint(client: TestClient):
    client_sess_id = f"client_{uuid.uuid4().hex[:12]}"
    create_res = client.post(
        "/api/v2/realtime/sessions",
        json={"client_session_id": client_sess_id, "started_at": "2026-10-01T12:00:00Z"},
    )
    assert create_res.status_code == 200
    session_id = create_res.json()["session_id"]

    status_res = client.get(f"/api/v2/realtime/sessions/{session_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["session_id"] == session_id
    assert status_data["current_risk"] == "LOW"
    assert status_data["status"] == "ACTIVE"
    assert status_data["chunks_processed"] == 0


def test_complete_session_endpoint(client: TestClient):
    client_sess_id = f"client_{uuid.uuid4().hex[:12]}"
    create_res = client.post(
        "/api/v2/realtime/sessions",
        json={"client_session_id": client_sess_id, "started_at": "2026-10-01T12:00:00Z"},
    )
    assert create_res.status_code == 200
    session_id = create_res.json()["session_id"]

    complete_res = client.post(
        f"/api/v2/realtime/sessions/{session_id}/complete",
        json={"ended_at": "2026-10-01T12:05:00Z"},
    )
    assert complete_res.status_code == 200
    data = complete_res.json()
    assert data["session_id"] == session_id
    assert data["status"] == "CLOSED"


def test_final_audio_endpoint(client: TestClient):
    client_sess_id = f"client_{uuid.uuid4().hex[:12]}"
    create_res = client.post(
        "/api/v2/realtime/sessions",
        json={"client_session_id": client_sess_id, "started_at": "2026-10-01T12:00:00Z"},
    )
    assert create_res.status_code == 200
    session_id = create_res.json()["session_id"]

    dummy_wav = b"RIFF$\x00\x00\x00WAVEfmt \x10\x00\x00\x00\x01\x00\x01\x00\x80>\x00\x00\x00}\x00\x00\x02\x00\x10\x00data\x00\x00\x00\x00"
    files = {"audio": ("final.wav", dummy_wav, "audio/wav")}

    final_res = client.post(
        f"/api/v2/realtime/sessions/{session_id}/final-audio",
        files=files,
    )
    assert final_res.status_code == 200
    data = final_res.json()
    assert data["session_id"] == session_id
    assert data["status"] == "COMPLETED"
    assert data["final_risk"] == "LOW"
