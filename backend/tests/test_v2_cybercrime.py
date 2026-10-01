"""Tests for Cybercrime categories, fraud detection templates, versioning, and API endpoints."""

import uuid
from fastapi.testclient import TestClient
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base, get_db
from app.main import app
from app.schemas.cybercrime import (
    CreateCategoryRequest,
    CreateTemplateRequest,
    CreateTemplateVersionRequest,
    UpdateTemplateRequest,
)
from app.services.cybercrime_service import CybercrimeService


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


def test_cybercrime_service_category_and_template_crud(test_db):
    # 1. Create Category
    cat_schema = CybercrimeService.create_category(
        test_db,
        CreateCategoryRequest(
            name="Bank KYC Scam",
            description="Fake messages urging immediate KYC update with APK download",
        ),
    )
    assert cat_schema.name == "Bank KYC Scam"

    # 2. Create Template
    tmpl_schema = CybercrimeService.create_template(
        test_db,
        CreateTemplateRequest(
            category_id=cat_schema.id,
            name="SBI Account Blocking Alert",
            description="SMS stating SBI YONO account will be blocked today",
            severity="HIGH",
            indicators=["SMS with APK download link", "Urgency to verify PAN/Aadhaar"],
            recommended_actions=["Do not click unknown links", "Report on 1930"],
            metadata={"alert_type": "SMS", "financial_institution": "SBI"},
        ),
    )
    assert tmpl_schema.name == "SBI Account Blocking Alert"
    assert tmpl_schema.severity == "HIGH"
    assert len(tmpl_schema.indicators) == 2
    assert len(tmpl_schema.versions) == 1
    assert tmpl_schema.versions[0].version == 1

    # 3. Update Template and generate version 2
    updated = CybercrimeService.update_template(
        test_db,
        tmpl_schema.id,
        UpdateTemplateRequest(
            severity="CRITICAL",
            indicators=tmpl_schema.indicators + ["Requests remote desktop app AnyDesk/TeamViewer"],
            create_new_version=True,
            change_notes="Added remote access vector",
        ),
    )
    assert updated.severity == "CRITICAL"
    assert len(updated.versions) == 2
    assert updated.versions[1].version == 2
    assert updated.versions[1].change_notes == "Added remote access vector"


def test_cybercrime_api_endpoints(client):
    # 1. POST /api/v2/cybercrime/categories
    cat_payload = {
        "name": "Electricity Bill Scam",
        "description": "Threat of power disconnection tonight unless paid immediately",
        "is_active": True,
    }
    cat_res = client.post("/api/v2/cybercrime/categories", json=cat_payload)
    assert cat_res.status_code == 201
    cat_data = cat_res.json()
    cat_id = cat_data["id"]

    # 2. GET /api/v2/cybercrime/categories
    cats_res = client.get("/api/v2/cybercrime/categories")
    assert cats_res.status_code == 200
    assert any(c["id"] == cat_id for c in cats_res.json())

    # 3. POST /api/v2/cybercrime/templates
    tmpl_payload = {
        "category_id": cat_id,
        "name": "Power Cut Tonight Fraud",
        "description": "Fake officer calling to update payment via WhatsApp",
        "severity": "HIGH",
        "indicators": ["Calls at evening hours", "Asks to install quick support app"],
        "recommended_actions": ["Contact official electricity board helpline"],
        "metadata": {"scam_channel": "VOICE_CALL"},
        "is_active": True,
    }
    tmpl_res = client.post("/api/v2/cybercrime/templates", json=tmpl_payload)
    assert tmpl_res.status_code == 201
    tmpl_data = tmpl_res.json()
    tmpl_id = tmpl_data["id"]
    assert tmpl_data["name"] == "Power Cut Tonight Fraud"
    assert len(tmpl_data["versions"]) == 1

    # 4. GET /api/v2/cybercrime/templates?category_id=...
    filter_res = client.get(f"/api/v2/cybercrime/templates?category_id={cat_id}")
    assert filter_res.status_code == 200
    assert len(filter_res.json()) == 1

    # 5. GET /api/v2/cybercrime/templates/{id}
    detail_res = client.get(f"/api/v2/cybercrime/templates/{tmpl_id}")
    assert detail_res.status_code == 200
    assert detail_res.json()["category_name"] == "Electricity Bill Scam"

    # 6. POST /api/v2/cybercrime/templates/{id}/versions
    new_ver_payload = {
        "template_data": {
            "name": "Power Cut Tonight Fraud v2",
            "indicators": ["Threatens penalty surcharge"],
        },
        "change_notes": "Added penalty fee pattern",
    }
    ver_res = client.post(f"/api/v2/cybercrime/templates/{tmpl_id}/versions", json=new_ver_payload)
    assert ver_res.status_code == 201
    assert ver_res.json()["version"] == 2

    # 7. GET /api/v2/cybercrime/templates/{id}/versions
    vers_res = client.get(f"/api/v2/cybercrime/templates/{tmpl_id}/versions")
    assert vers_res.status_code == 200
    assert len(vers_res.json()) == 2
