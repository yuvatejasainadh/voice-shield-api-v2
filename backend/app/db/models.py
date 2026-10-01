"""SQLAlchemy models for persistence layer compatible with VoiceShield V2 PostgreSQL schema."""

from __future__ import annotations

import uuid as _uuid
from datetime import datetime
from typing import Any

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    Uuid,
    func,
    JSON,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base

# Dialect-agnostic JSON type that maps to native JSONB in PostgreSQL and JSON in SQLite
JSONType = JSON().with_variant(JSONB, "postgresql")


class AnalysisRecord(Base):
    """Stores analysis metadata and output for a processed audio sample."""

    __tablename__ = "analysis_records"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, index=True)
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    stored_audio_path: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False)
    classification: Mapped[str] = mapped_column(String(32), nullable=False)
    risk_score: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    ai_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    duration_seconds: Mapped[float] = mapped_column(Float, nullable=False)
    segments_analyzed: Mapped[int | None] = mapped_column(Integer, nullable=True)
    processing_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detector_version: Mapped[str] = mapped_column(String(64), nullable=False)
    reasons: Mapped[list[str]] = mapped_column(JSONType, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)


class CallSession(Base):
    """Represents one user phone call, with realtime provisional alerts and final authoritative results."""

    __tablename__ = "call_sessions"

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    client_session_id: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    current_risk: Mapped[str] = mapped_column(String(32), nullable=False, default="LOW")
    peak_risk: Mapped[str] = mapped_column(String(32), nullable=False, default="LOW")
    current_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    peak_score: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    realtime_alert_state: Mapped[str] = mapped_column(String(32), nullable=False, default="LOW")
    final_risk: Mapped[str | None] = mapped_column(String(32), nullable=True)
    final_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    final_classification: Mapped[str | None] = mapped_column(String(64), nullable=True, default="ANALYZING")
    confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    chunks_analyzed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    total_duration_seconds: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    evidence: Mapped[list[dict] | None] = mapped_column(JSONType, nullable=True)
    detector_version: Mapped[str] = mapped_column(String(64), nullable=False, default="aurigin-realtime")
    final_provider: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)
    last_notification_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    final_audio_reference: Mapped[str | None] = mapped_column(String(255), nullable=True)

    chunks: Mapped[list["CallChunk"]] = relationship(back_populates="session", cascade="all, delete-orphan")


class CallChunk(Base):
    """Stores chunk-level processing metadata without persisting raw audio."""

    __tablename__ = "call_chunks"
    __table_args__ = (
        UniqueConstraint("call_session_id", "sequence_number", name="uq_call_chunk_session_sequence"),
        UniqueConstraint("call_session_id", "idempotency_key", name="uq_call_chunk_session_idempotency_key"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    call_session_id: Mapped[str] = mapped_column(String(64), ForeignKey("call_sessions.id", ondelete="CASCADE"), nullable=False, index=True)
    sequence_number: Mapped[int] = mapped_column(Integer, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    provider: Mapped[str] = mapped_column(String(64), nullable=False, default="aurigin")
    provider_prediction_id: Mapped[str | None] = mapped_column(String(128), nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk: Mapped[str | None] = mapped_column(String(32), nullable=True)
    processing_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="PROCESSED")
    idempotency_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)

    session: Mapped[CallSession] = relationship(back_populates="chunks")


class DeviceRecordingCompatibility(Base):
    """Stores validated OEM device recording discovery profiles and constraints."""

    __tablename__ = "device_recording_compatibilities"
    __table_args__ = (
        UniqueConstraint("manufacturer", "model", name="uq_device_recording_compatibility_mfg_model"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True, index=True)
    manufacturer: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    model: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    marketing_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    os_family: Mapped[str | None] = mapped_column(String(64), nullable=True)
    os_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    android_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    recording_profile: Mapped[str] = mapped_column(String(32), nullable=False)
    supported: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    configuration: Mapped[dict] = mapped_column(JSONType, nullable=False)
    profile_version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=False)


# ==========================================
# VoiceShield V2.0 Models
# ==========================================


class User(Base):
    """Registered application user tied to Firebase Phone Authentication."""

    __tablename__ = "users"

    id: Mapped[_uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=_uuid.uuid4,
    )
    phone_number: Mapped[str] = mapped_column(String(20), unique=True, nullable=False, index=True)
    display_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    firebase_uid: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String(32), default="ACTIVE", server_default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    devices: Mapped[list["UserDevice"]] = relationship(back_populates="user", cascade="all, delete-orphan")
    auth_events: Mapped[list["UserAuthEvent"]] = relationship(back_populates="user")


class UserDevice(Base):
    """User hardware devices associated with authenticated accounts."""

    __tablename__ = "user_devices"

    id: Mapped[_uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=_uuid.uuid4,
    )
    user_id: Mapped[_uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    device_fingerprint: Mapped[str] = mapped_column(String(255), unique=True, nullable=False, index=True)
    manufacturer: Mapped[str | None] = mapped_column(String(128), nullable=True)
    model: Mapped[str | None] = mapped_column(String(128), nullable=True)
    android_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    app_version: Mapped[str | None] = mapped_column(String(64), nullable=True)
    first_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=True)
    last_seen_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")

    user: Mapped[User] = relationship(back_populates="devices")
    auth_events: Mapped[list["UserAuthEvent"]] = relationship(back_populates="device")


class UserAuthEvent(Base):
    """Audit log of authentication attempts and Firebase token verifications."""

    __tablename__ = "user_auth_events"

    id: Mapped[_uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=_uuid.uuid4,
    )
    user_id: Mapped[_uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    firebase_uid: Mapped[str | None] = mapped_column(String(128), nullable=True, index=True)
    phone_number: Mapped[str | None] = mapped_column(String(20), nullable=True)
    event_type: Mapped[str] = mapped_column(String(32), nullable=False)
    success: Mapped[bool] = mapped_column(Boolean, nullable=False)
    device_id: Mapped[_uuid.UUID | None] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("user_devices.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=True)

    user: Mapped[User | None] = relationship(back_populates="auth_events")
    device: Mapped[UserDevice | None] = relationship(back_populates="auth_events")


class CybercrimeCategory(Base):
    """High-level categories of cybercrime / scam patterns."""

    __tablename__ = "cybercrime_categories"

    id: Mapped[_uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=_uuid.uuid4,
    )
    name: Mapped[str] = mapped_column(String(128), unique=True, nullable=False, index=True)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=True)

    templates: Mapped[list["CybercrimeTemplate"]] = relationship(back_populates="category")


class CybercrimeTemplate(Base):
    """Known scam / fraud detection patterns and guidance templates."""

    __tablename__ = "cybercrime_templates"
    __table_args__ = (
        UniqueConstraint("category_id", "name", name="uq_cybercrime_template_category_name"),
    )

    id: Mapped[_uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=_uuid.uuid4,
    )
    category_id: Mapped[_uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("cybercrime_categories.id", ondelete="RESTRICT"),
        nullable=False,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    severity: Mapped[str | None] = mapped_column(String(32), nullable=True)
    indicators: Mapped[list[Any]] = mapped_column(
        JSONType,
        default=list,
        nullable=True,
    )
    recommended_actions: Mapped[list[Any]] = mapped_column(
        JSONType,
        default=list,
        nullable=True,
    )
    template_metadata: Mapped[dict[str, Any]] = mapped_column(
        "metadata",
        JSONType,
        default=dict,
        nullable=True,
    )
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, server_default="1")
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), onupdate=func.now(), nullable=True)

    category: Mapped[CybercrimeCategory] = relationship(back_populates="templates")
    versions: Mapped[list["CybercrimeTemplateVersion"]] = relationship(back_populates="template", cascade="all, delete-orphan")


class CybercrimeTemplateVersion(Base):
    """Historical versioning of cybercrime templates."""

    __tablename__ = "cybercrime_template_versions"
    __table_args__ = (
        UniqueConstraint("template_id", "version", name="uq_cybercrime_template_version"),
    )

    id: Mapped[_uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        primary_key=True,
        default=_uuid.uuid4,
    )
    template_id: Mapped[_uuid.UUID] = mapped_column(
        Uuid(as_uuid=True),
        ForeignKey("cybercrime_templates.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    template_data: Mapped[dict[str, Any]] = mapped_column(JSONType, nullable=False)
    change_notes: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now(), nullable=True)

    template: Mapped[CybercrimeTemplate] = relationship(back_populates="versions")
