"""Tests for all 10 VoiceShield V2 database schema tables, constraints, cascades, and JSONB handling."""

import uuid
from datetime import datetime, timezone
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.database import Base
from app.db.models import (
    AnalysisRecord,
    CallChunk,
    CallSession,
    CybercrimeCategory,
    CybercrimeTemplate,
    CybercrimeTemplateVersion,
    DeviceRecordingCompatibility,
    User,
    UserAuthEvent,
    UserDevice,
)


@pytest.fixture
def db_session():
    """Create an isolated in-memory SQLite database for schema testing."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    Session = sessionmaker(bind=engine)
    session = Session()
    try:
        yield session
    finally:
        session.close()


# 1. AnalysisRecord
def test_analysis_record_crud(db_session):
    record = AnalysisRecord(
        id=str(uuid.uuid4()),
        filename="test_call.wav",
        stored_audio_path="/storage/test_call.wav",
        status="completed",
        classification="LIKELY_GENUINE",
        risk_score=15,
        confidence=0.95,
        ai_probability=0.05,
        duration_seconds=5.2,
        segments_analyzed=2,
        processing_time_ms=180,
        detector_version="aurigin-v1",
        reasons=["Model verified natural vocal acoustic characteristics"],
    )
    db_session.add(record)
    db_session.commit()

    queried = db_session.execute(select(AnalysisRecord).where(AnalysisRecord.id == record.id)).scalar_one()
    assert queried.filename == "test_call.wav"
    assert queried.risk_score == 15
    assert queried.reasons == ["Model verified natural vocal acoustic characteristics"]


# 2 & 3. CallSession & CallChunk (with sequence uniqueness, idempotency, cascade delete)
def test_call_session_and_chunks(db_session):
    session_id = str(uuid.uuid4())
    client_sess_id = f"client_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc)

    session = CallSession(
        id=session_id,
        client_session_id=client_sess_id,
        started_at=now,
        current_risk="LOW",
        peak_risk="LOW",
        current_score=0.1,
        peak_score=0.1,
        realtime_alert_state="LOW",
        status="ACTIVE",
    )
    db_session.add(session)
    db_session.commit()

    chunk1 = CallChunk(
        id=str(uuid.uuid4()),
        call_session_id=session_id,
        sequence_number=1,
        duration_ms=2500,
        provider="aurigin",
        score=0.1,
        risk="LOW",
        processing_ms=120,
        status="PROCESSED",
        idempotency_key="idemp_1",
    )
    chunk2 = CallChunk(
        id=str(uuid.uuid4()),
        call_session_id=session_id,
        sequence_number=2,
        duration_ms=2500,
        provider="aurigin",
        score=0.85,
        risk="HIGH",
        processing_ms=130,
        status="PROCESSED",
        idempotency_key="idemp_2",
    )
    db_session.add_all([chunk1, chunk2])
    db_session.commit()

    # Verify chunks queryable via relationship
    db_session.refresh(session)
    assert len(session.chunks) == 2
    assert session.chunks[0].sequence_number == 1
    assert session.chunks[1].score == 0.85

    # Test sequence number uniqueness constraint per session
    duplicate_seq = CallChunk(
        id=str(uuid.uuid4()),
        call_session_id=session_id,
        sequence_number=1,
        duration_ms=2500,
        provider="aurigin",
        status="PROCESSED",
        idempotency_key="idemp_unique_key",
    )
    db_session.add(duplicate_seq)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Test idempotency key uniqueness constraint per session
    duplicate_idemp = CallChunk(
        id=str(uuid.uuid4()),
        call_session_id=session_id,
        sequence_number=3,
        duration_ms=2500,
        provider="aurigin",
        status="PROCESSED",
        idempotency_key="idemp_1",
    )
    db_session.add(duplicate_idemp)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Test cascade deletion
    db_session.delete(session)
    db_session.commit()
    remaining_chunks = db_session.execute(
        select(CallChunk).where(CallChunk.call_session_id == session_id)
    ).scalars().all()
    assert len(remaining_chunks) == 0


# 4. DeviceRecordingCompatibility
def test_device_recording_compatibility(db_session):
    drc = DeviceRecordingCompatibility(
        id=str(uuid.uuid4()),
        manufacturer="Vivo",
        model="V2558",
        marketing_name="V70 FE",
        recording_profile="ROOT_LEVEL",
        supported=True,
        configuration={"search_scope": "ROOT", "extensions": [".m4a"]},
        profile_version=1,
    )
    db_session.add(drc)
    db_session.commit()

    queried = db_session.execute(
        select(DeviceRecordingCompatibility).where(DeviceRecordingCompatibility.manufacturer == "Vivo")
    ).scalar_one()
    assert queried.marketing_name == "V70 FE"
    assert queried.configuration["extensions"] == [".m4a"]

    # Test uniqueness on (manufacturer, model)
    duplicate_drc = DeviceRecordingCompatibility(
        id=str(uuid.uuid4()),
        manufacturer="Vivo",
        model="V2558",
        recording_profile="SUB_FOLDER",
        supported=False,
        configuration={},
        profile_version=1,
    )
    db_session.add(duplicate_drc)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()


# 5, 6, 7. Users, UserDevices, and UserAuthEvents
def test_user_device_and_auth_events(db_session):
    user_id = uuid.uuid4()
    firebase_uid = f"fb_{uuid.uuid4().hex[:16]}"
    phone = "+14155552671"

    user = User(
        id=user_id,
        phone_number=phone,
        display_name="Alice User",
        firebase_uid=firebase_uid,
        status="ACTIVE",
    )
    db_session.add(user)
    db_session.commit()

    # Uniqueness constraint on phone_number and firebase_uid
    duplicate_user = User(
        id=uuid.uuid4(),
        phone_number=phone,
        firebase_uid=f"fb_other_{uuid.uuid4().hex[:8]}",
    )
    db_session.add(duplicate_user)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # User Device creation
    dev_fingerprint = f"fp_{uuid.uuid4().hex}"
    device = UserDevice(
        id=uuid.uuid4(),
        user_id=user.id,
        device_fingerprint=dev_fingerprint,
        manufacturer="Google",
        model="Pixel 8",
        android_version="14",
        app_version="2.0.0",
        is_active=True,
    )
    db_session.add(device)
    db_session.commit()

    # Device fingerprint uniqueness
    duplicate_dev = UserDevice(
        id=uuid.uuid4(),
        user_id=user.id,
        device_fingerprint=dev_fingerprint,
    )
    db_session.add(duplicate_dev)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Auth Event creation
    auth_event = UserAuthEvent(
        id=uuid.uuid4(),
        user_id=user.id,
        firebase_uid=firebase_uid,
        phone_number=phone,
        event_type="LOGIN",
        success=True,
        device_id=device.id,
    )
    db_session.add(auth_event)
    db_session.commit()

    # Relationship verify
    db_session.refresh(user)
    assert len(user.devices) == 1
    assert user.devices[0].model == "Pixel 8"
    assert len(user.auth_events) == 1
    assert user.auth_events[0].event_type == "LOGIN"

    # Cascade deletion on user -> devices cascade deleted, auth_events SET NULL
    db_session.delete(user)
    db_session.commit()

    devs = db_session.execute(select(UserDevice).where(UserDevice.id == device.id)).scalars().all()
    assert len(devs) == 0


# 8, 9, 10. Cybercrime Categories, Templates, and Template Versions
def test_cybercrime_schema_and_versioning(db_session):
    cat_id = uuid.uuid4()
    cat = CybercrimeCategory(
        id=cat_id,
        name="Digital Arrest Scam",
        description="Impersonation of law enforcement demanding immediate payment",
        is_active=True,
    )
    db_session.add(cat)
    db_session.commit()

    # Category uniqueness
    duplicate_cat = CybercrimeCategory(id=uuid.uuid4(), name="Digital Arrest Scam")
    db_session.add(duplicate_cat)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Create template
    tmpl_id = uuid.uuid4()
    tmpl = CybercrimeTemplate(
        id=tmpl_id,
        category_id=cat.id,
        name="Police Video Call Extortion",
        description="Fake police background with threats of warrant",
        severity="CRITICAL",
        indicators=["Demands Skype/WhatsApp video call", "Shows fake police badge", "Threatens immediate arrest"],
        recommended_actions=["Disconnect call immediately", "Call national cybercrime helpline 1930", "Do not transfer money"],
        template_metadata={"confidence_threshold": 0.85, "urgency": "HIGH"},
        is_active=True,
    )
    db_session.add(tmpl)
    db_session.commit()

    # Unique constraint on (category_id, name)
    duplicate_tmpl = CybercrimeTemplate(
        id=uuid.uuid4(),
        category_id=cat.id,
        name="Police Video Call Extortion",
    )
    db_session.add(duplicate_tmpl)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Template versions
    v1 = CybercrimeTemplateVersion(
        id=uuid.uuid4(),
        template_id=tmpl.id,
        version=1,
        template_data={"indicators": tmpl.indicators, "recommended_actions": tmpl.recommended_actions},
        change_notes="Initial baseline",
    )
    v2 = CybercrimeTemplateVersion(
        id=uuid.uuid4(),
        template_id=tmpl.id,
        version=2,
        template_data={"indicators": tmpl.indicators + ["Claims money in customs"], "recommended_actions": tmpl.recommended_actions},
        change_notes="Added customs variation",
    )
    db_session.add_all([v1, v2])
    db_session.commit()

    # Template version uniqueness constraint on (template_id, version)
    dup_v = CybercrimeTemplateVersion(
        id=uuid.uuid4(),
        template_id=tmpl.id,
        version=1,
        template_data={},
    )
    db_session.add(dup_v)
    with pytest.raises(IntegrityError):
        db_session.commit()
    db_session.rollback()

    # Query with relationships
    db_session.refresh(tmpl)
    assert len(tmpl.versions) == 2
    assert tmpl.versions[0].version == 1
    assert tmpl.versions[1].version == 2
    assert tmpl.category.name == "Digital Arrest Scam"
