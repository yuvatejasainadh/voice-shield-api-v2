"""Automated tests for Server-Side Device Compatibility Matrix."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import select

from app.db.database import Base, SessionLocal, engine, upgrade_sqlite_schema
from app.db.models import DeviceRecordingCompatibility
from app.db.seed_compatibility import seed_device_compatibilities
from app.main import app


@pytest.fixture(autouse=True)
def setup_compatibility_db():
    Base.metadata.create_all(bind=engine)
    upgrade_sqlite_schema()
    with SessionLocal() as db:
        seed_device_compatibilities(db)
    yield


def test_vivo_v70_fe_v2558_compatibility():
    """1. Test Vivo V70 FE (hardware model V2558) resolves to ROOT_LEVEL with .m4a extension."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "Vivo",
            "model": "V2558",
            "os_family": "Origin OS",
            "os_version": "6",
            "android_version": "16",
        }
        resp = client.post("/api/v1/device/compatibility", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is True
        assert data["profile"] == "ROOT_LEVEL"
        assert data["profile_version"] == 1
        assert data["configuration"] is not None
        assert data["configuration"]["search_scope"] == "ROOT"
        assert data["configuration"]["extensions"] == [".m4a"]


def test_vivo_t2_pro_v2321_compatibility():
    """2. Test Vivo T2 Pro (hardware model V2321) resolves to ROOT_LEVEL with .m4a extension."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "Vivo",
            "model": "V2321",
            "os_family": "Funtouch OS",
            "os_version": "15",
            "android_version": "15",
        }
        resp = client.post("/api/v1/device/compatibility", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is True
        assert data["profile"] == "ROOT_LEVEL"
        assert data["profile_version"] == 1
        assert data["configuration"] is not None
        assert data["configuration"]["search_scope"] == "ROOT"
        assert data["configuration"]["extensions"] == [".m4a"]


def test_infinix_note_40_pro_plus_x6851b_compatibility():
    """3. Test Infinix Note 40 Pro+ (hardware model Infinix X6851B) resolves to SUB_FOLDER with .aac extension and recording_folder."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "Infinix",
            "model": "Infinix X6851B",
            "os_family": "XOS",
            "os_version": "15",
            "android_version": "15",
        }
        resp = client.post("/api/v1/device/compatibility", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is True
        assert data["profile"] == "SUB_FOLDER"
        assert data["profile_version"] == 1
        assert data["configuration"] is not None
        assert data["configuration"]["search_scope"] == "SUB_FOLDER"
        assert data["configuration"]["recording_folder"] == "Music/PhoneRecord"
        assert data["configuration"]["extensions"] == [".aac"]


def test_vivo_marketing_name_v70_fe_unsupported():
    """4. Test Vivo with marketing name 'V70 FE' is rejected (not authoritative hardware model)."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "Vivo",
            "model": "V70 FE",
            "os_family": "Origin OS",
            "os_version": "6",
            "android_version": "16",
        }
        resp = client.post("/api/v1/device/compatibility", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is False
        assert data["profile"] is None
        assert data["configuration"] is None
        assert data["profile_version"] is None


def test_vivo_marketing_name_t2_pro_unsupported():
    """5. Test Vivo with marketing name 'T2 Pro' is rejected (not authoritative hardware model)."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "Vivo",
            "model": "T2 Pro",
            "os_family": "Funtouch OS",
            "os_version": "15",
            "android_version": "15",
        }
        resp = client.post("/api/v1/device/compatibility", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is False
        assert data["profile"] is None
        assert data["configuration"] is None
        assert data["profile_version"] is None


def test_infinix_marketing_name_note_40_pro_plus_unsupported():
    """6. Test Infinix with marketing name 'Note 40 Pro+' is rejected (not authoritative hardware model)."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "Infinix",
            "model": "Note 40 Pro+",
            "os_family": "XOS",
            "os_version": "15",
            "android_version": "15",
        }
        resp = client.post("/api/v1/device/compatibility", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is False
        assert data["profile"] is None
        assert data["configuration"] is None
        assert data["profile_version"] is None


def test_infinix_truncated_model_x6851b_unsupported():
    """7. Test Infinix with 'X6851B' is rejected (hardware model must be 'Infinix X6851B')."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "Infinix",
            "model": "X6851B",
            "os_family": "XOS",
            "os_version": "15",
            "android_version": "15",
        }
        resp = client.post("/api/v1/device/compatibility", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is False
        assert data["profile"] is None
        assert data["configuration"] is None
        assert data["profile_version"] is None


def test_case_and_whitespace_normalization_all_three():
    """8. Test manufacturer and model matching is case-insensitive and trims whitespace for all three verified devices."""
    with TestClient(app) as client:
        # Vivo V2558
        resp1 = client.post(
            "/api/v1/device/compatibility",
            json={"manufacturer": "  viVO  ", "model": "  v2558  "},
        )
        assert resp1.status_code == 200
        data1 = resp1.json()
        assert data1["supported"] is True
        assert data1["profile"] == "ROOT_LEVEL"
        assert data1["configuration"]["extensions"] == [".m4a"]

        # Vivo V2321
        resp2 = client.post(
            "/api/v1/device/compatibility",
            json={"manufacturer": " VIVO ", "model": " v2321 "},
        )
        assert resp2.status_code == 200
        data2 = resp2.json()
        assert data2["supported"] is True
        assert data2["profile"] == "ROOT_LEVEL"
        assert data2["configuration"]["extensions"] == [".m4a"]

        # Infinix Infinix X6851B
        resp3 = client.post(
            "/api/v1/device/compatibility",
            json={"manufacturer": "INFINIX", "model": "  INFINIX X6851B  "},
        )
        assert resp3.status_code == 200
        data3 = resp3.json()
        assert data3["supported"] is True
        assert data3["profile"] == "SUB_FOLDER"
        assert data3["configuration"]["search_scope"] == "SUB_FOLDER"
        assert data3["configuration"]["recording_folder"] == "Music/PhoneRecord"
        assert data3["configuration"]["extensions"] == [".aac"]


def test_infinix_api_response_serialization_explicit():
    """Verify the API response JSON explicitly contains recording_folder key and value."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "INFINIX",
            "model": "Infinix X6851B",
            "os_family": "Android",
            "os_version": "15",
            "android_version": "15",
        }
        resp = client.post("/api/v1/device/compatibility", json=payload)
        assert resp.status_code == 200
        assert '"recording_folder": "Music/PhoneRecord"' in resp.text or '"recording_folder":"Music/PhoneRecord"' in resp.text
        data = resp.json()
        assert data["configuration"]["recording_folder"] == "Music/PhoneRecord"


def test_backward_compatibility_null_or_missing_recording_folder():
    """Verify DeviceRecordingConfig handles configurations without recording_folder gracefully."""
    from app.schemas.device_compatibility import DeviceRecordingConfig

    # 1. Without recording_folder key
    legacy_dict = {"search_scope": "ROOT", "extensions": [".m4a"]}
    config1 = DeviceRecordingConfig(**legacy_dict)
    assert config1.search_scope == "ROOT"
    assert config1.recording_folder is None
    assert config1.extensions == [".m4a"]

    # 2. With explicit None recording_folder
    explicit_null_dict = {"search_scope": "ROOT", "recording_folder": None, "extensions": [".m4a"]}
    config2 = DeviceRecordingConfig(**explicit_null_dict)
    assert config2.recording_folder is None

    # 3. Vivo endpoint response has None/unresolved recording_folder without failing
    with TestClient(app) as client:
        resp = client.post("/api/v1/device/compatibility", json={"manufacturer": "Vivo", "model": "V2558"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is True
        assert data["configuration"]["recording_folder"] is None


def test_unknown_vivo_model_unsupported():
    """9. Test unvalidated model from supported manufacturer is rejected."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "Vivo",
            "model": "X90 Pro",
            "os_family": "Funtouch OS",
            "os_version": "14",
        }
        resp = client.post("/api/v1/device/compatibility", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is False
        assert data["profile"] is None
        assert data["configuration"] is None
        assert data["profile_version"] is None


def test_unknown_manufacturer_unsupported():
    """10. Test unknown manufacturer is rejected."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "AcmeBrand",
            "model": "SuperPhone 1",
        }
        resp = client.post("/api/v1/device/compatibility", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is False
        assert data["profile"] is None
        assert data["configuration"] is None
        assert data["profile_version"] is None


def test_seed_idempotent():
    """11. Test that repeatedly calling seed_device_compatibilities does not create duplicate records."""
    with SessionLocal() as db:
        # Run seed multiple times
        seed_device_compatibilities(db)
        seed_device_compatibilities(db)
        seed_device_compatibilities(db)

        # Query all records
        records = db.execute(select(DeviceRecordingCompatibility)).scalars().all()
        # Should have exactly 3 authoritative records
        assert len(records) == 3

        record_map = {(r.manufacturer, r.model): r for r in records}
        assert ("Vivo", "V2558") in record_map
        assert ("Vivo", "V2321") in record_map
        assert ("Infinix", "Infinix X6851B") in record_map

        # Verify marketing names and configurations
        assert record_map[("Vivo", "V2558")].marketing_name == "V70 FE"
        assert record_map[("Vivo", "V2321")].marketing_name == "T2 Pro"
        assert record_map[("Infinix", "Infinix X6851B")].marketing_name == "Note 40 Pro+"
        assert record_map[("Infinix", "Infinix X6851B")].configuration["recording_folder"] == "Music/PhoneRecord"


def test_oppo_unvalidated_model_unsupported():
    """Test Oppo device remains unsupported."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "Oppo",
            "model": "Find X7",
            "os_family": "ColorOS",
            "os_version": "14",
        }
        resp = client.post("/api/v1/device/compatibility", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is False
        assert data["profile"] is None
        assert data["configuration"] is None


def test_missing_required_fields_validation_error():
    """Test missing manufacturer or model returns 422 Unprocessable Entity."""
    with TestClient(app) as client:
        resp = client.post("/api/v1/device/compatibility", json={"manufacturer": "Vivo"})
        assert resp.status_code == 422


def test_api_alias_endpoint():
    """Test endpoint is also reachable via /api/device/compatibility."""
    with TestClient(app) as client:
        payload = {
            "manufacturer": "Vivo",
            "model": "V2558",
        }
        resp = client.post("/api/device/compatibility", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["supported"] is True
        assert data["profile"] == "ROOT_LEVEL"
