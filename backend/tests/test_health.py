"""Tests for health, readiness, and root metadata endpoints in VoiceShield V2."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings


def test_root_endpoint(client: TestClient):
    """Test root GET / returns metadata with V2 status."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "VoiceShield API"
    assert data["version"] == "V2"
    assert data["status"] == "ok"
    assert data["docs"] == "/docs"


def test_health_endpoint_v2(client: TestClient):
    """Test GET /api/v2/health liveness probe."""
    settings = get_settings()
    response = client.get(f"{settings.api_prefix}/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["version"] == "v2"
    assert data["service"] == "voiceshield-api"


def test_ready_endpoint_v2(client: TestClient):
    """Test GET /api/v2/ready readiness probe."""
    settings = get_settings()
    response = client.get(f"{settings.api_prefix}/ready")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ready"
    assert data["version"] == "v2"
    assert data["database"] in ("connected", "ready", "ok")
    assert "database_engine" in data
    assert "migration_revision" in data


def test_legacy_v1_routes_not_registered(client: TestClient):
    """Verify legacy /api/v1 endpoints return 404."""
    response = client.get("/api/v1/health")
    assert response.status_code == 404
