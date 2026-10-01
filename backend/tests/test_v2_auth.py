"""Tests for Firebase User Authentication, device registration, and auth event recording."""

import uuid
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.main import app
from app.schemas.auth import AuthSyncRequest
from app.services.auth_service import AuthService


@pytest.fixture
def test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine, autocommit=False, autoflush=False)
    session = Session()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def client(test_db):
    def _get_db():
        try:
            yield test_db
        finally:
            pass

    app.dependency_overrides[get_db] = _get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def test_auth_service_sync_creates_user_and_device(test_db):
    fb_uid = f"fb_{uuid.uuid4().hex[:12]}"
    phone = "+919876543210"
    dev_fp = f"fp_{uuid.uuid4().hex[:16]}"

    req = AuthSyncRequest(
        firebase_uid=fb_uid,
        phone_number=phone,
        display_name="Rahul Sharma",
        device_fingerprint=dev_fp,
        manufacturer="Samsung",
        model="Galaxy S24",
        android_version="14",
        app_version="2.0.0",
        event_type="SIGNUP",
    )

    resp = AuthService.sync_user(db=test_db, payload=req)
    assert resp.success is True
    assert resp.user.firebase_uid == fb_uid
    assert resp.user.phone_number == phone
    assert resp.user.display_name == "Rahul Sharma"
    assert resp.device_id is not None

    # Sync again as LOGIN (should update last_login_at and not duplicate user)
    login_req = AuthSyncRequest(
        firebase_uid=fb_uid,
        phone_number=phone,
        display_name="Rahul S.",
        device_fingerprint=dev_fp,
        event_type="LOGIN",
    )
    login_resp = AuthService.sync_user(db=test_db, payload=login_req)
    assert login_resp.user.id == resp.user.id
    assert login_resp.user.display_name == "Rahul S."
    assert login_resp.device_id == resp.device_id


def test_auth_api_routes(client):
    fb_uid = f"fb_api_{uuid.uuid4().hex[:12]}"
    phone = "+919999988888"

    payload = {
        "firebase_uid": fb_uid,
        "phone_number": phone,
        "display_name": "API Test User",
        "device_fingerprint": f"dev_api_{uuid.uuid4().hex[:8]}",
        "manufacturer": "OnePlus",
        "model": "12R",
        "android_version": "14",
        "app_version": "2.0.0",
        "event_type": "LOGIN",
    }

    # POST /api/v1/auth/sync
    res = client.post("/api/v1/auth/sync", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["user"]["firebase_uid"] == fb_uid
    user_id = data["user"]["id"]

    # GET /api/v1/auth/user?firebase_uid=...
    get_res = client.get(f"/api/v1/auth/user?firebase_uid={fb_uid}")
    assert get_res.status_code == 200
    assert get_res.json()["id"] == user_id
    assert len(get_res.json()["devices"]) == 1

    # GET /api/v1/auth/user?user_id=...
    get_res2 = client.get(f"/api/v1/auth/user?user_id={user_id}")
    assert get_res2.status_code == 200
    assert get_res2.json()["phone_number"] == phone
