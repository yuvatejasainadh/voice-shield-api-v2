"""Automated tests for Server-Side Device Compatibility Matrix and Seeding."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import DeviceRecordingCompatibility
from app.db.seed_compatibility import seed_device_compatibilities


def test_vivo_v70_fe_v2558_compatibility(client: TestClient):
    """Test Vivo V70 FE (hardware model V2558) resolves to ROOT_LEVEL with .m4a extension."""
    payload = {
        "manufacturer": "Vivo",
        "model": "V2558",
        "os_family": "Origin OS",
        "os_version": "6",
        "android_version": "16",
    }
    resp = client.post("/api/v2/device/compatibility", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["supported"] is True
    assert data["profile"] == "ROOT_LEVEL"
    assert data["profile_version"] == 1
    assert data["configuration"]["search_scope"] == "ROOT"
    assert data["configuration"]["extensions"] == [".m4a"]


def test_vivo_t2_pro_v2321_compatibility(client: TestClient):
    """Test Vivo T2 Pro (hardware model V2321) resolves to ROOT_LEVEL with .m4a extension."""
    payload = {
        "manufacturer": "Vivo",
        "model": "V2321",
        "os_family": "Funtouch OS",
        "os_version": "15",
        "android_version": "15",
    }
    resp = client.post("/api/v2/device/compatibility", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["supported"] is True
    assert data["profile"] == "ROOT_LEVEL"
    assert data["configuration"]["extensions"] == [".m4a"]


def test_infinix_note_40_pro_plus_x6851b_compatibility(client: TestClient):
    """Test Infinix Note 40 Pro+ (model Infinix X6851B) resolves to SUB_FOLDER with .aac extension."""
    payload = {
        "manufacturer": "Infinix",
        "model": "Infinix X6851B",
        "os_family": "XOS",
        "os_version": "15",
        "android_version": "15",
    }
    resp = client.post("/api/v2/device/compatibility", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["supported"] is True
    assert data["profile"] == "SUB_FOLDER"
    assert data["configuration"]["recording_folder"] == "Music/PhoneRecord"
    assert data["configuration"]["extensions"] == [".aac"]


def test_marketing_name_unsupported(client: TestClient):
    """Test marketing names are rejected when hardware model is expected."""
    payload = {
        "manufacturer": "Vivo",
        "model": "V70 FE",
    }
    resp = client.post("/api/v2/device/compatibility", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["supported"] is False
    assert data["profile"] is None


def test_case_and_whitespace_normalization(client: TestClient):
    """Test manufacturer and model matching is case-insensitive and trims whitespace."""
    resp = client.post(
        "/api/v2/device/compatibility",
        json={"manufacturer": "  viVO  ", "model": "  v2558  "},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["supported"] is True
    assert data["profile"] == "ROOT_LEVEL"


def test_seed_device_compatibilities_idempotent(db_session: Session):
    """Test that repeatedly calling seed_device_compatibilities does not create duplicate records."""
    # Seed multiple times
    seed_device_compatibilities(db_session)
    seed_device_compatibilities(db_session)
    seed_device_compatibilities(db_session)

    records = db_session.execute(select(DeviceRecordingCompatibility)).scalars().all()
    assert len(records) == 3

    record_map = {(r.manufacturer, r.model): r for r in records}
    assert ("Vivo", "V2558") in record_map
    assert ("Vivo", "V2321") in record_map
    assert ("Infinix", "Infinix X6851B") in record_map
    assert record_map[("Vivo", "V2558")].marketing_name == "V70 FE"
