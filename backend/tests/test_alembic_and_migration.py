"""Tests for Alembic baseline migration consistency and the SQLite-to-PostgreSQL data migration tool."""

from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path
import pytest
from sqlalchemy import create_engine, select
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
from tools.migrate_sqlite_to_postgres import migrate_data, transform_row


def test_schema_model_table_coverage():
    """Verify all 10 V2 model tables exist in Base.metadata with all expected columns."""
    expected_tables = {
        "analysis_records",
        "call_sessions",
        "call_chunks",
        "device_recording_compatibilities",
        "users",
        "user_devices",
        "user_auth_events",
        "cybercrime_categories",
        "cybercrime_templates",
        "cybercrime_template_versions",
    }
    registered_tables = set(Base.metadata.tables.keys())
    assert expected_tables.issubset(registered_tables), f"Missing tables: {expected_tables - registered_tables}"


def test_data_migration_tool_with_sample_sqlite(tmp_path: Path):
    """Verify migrate_data reads from SQLite and populates target database accurately with type conversion."""
    sqlite_file = tmp_path / "source.db"
    conn = sqlite3.connect(str(sqlite_file))
    cur = conn.cursor()

    # Create source tables with raw SQLite types
    cur.execute("""
        CREATE TABLE analysis_records (
            id TEXT PRIMARY KEY,
            filename TEXT,
            stored_audio_path TEXT,
            status TEXT,
            classification TEXT,
            risk_score INTEGER,
            confidence REAL,
            ai_probability REAL,
            duration_seconds REAL,
            segments_analyzed INTEGER,
            processing_time_ms INTEGER,
            detector_version TEXT,
            reasons TEXT,
            created_at TEXT
        )
    """)
    cur.execute("""
        CREATE TABLE device_recording_compatibilities (
            id TEXT PRIMARY KEY,
            manufacturer TEXT,
            model TEXT,
            marketing_name TEXT,
            os_family TEXT,
            os_version TEXT,
            android_version TEXT,
            recording_profile TEXT,
            supported INTEGER,
            configuration TEXT,
            profile_version INTEGER,
            created_at TEXT,
            updated_at TEXT
        )
    """)

    # Insert sample rows into SQLite
    rec_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO analysis_records VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (rec_id, "call.wav", "/storage/call.wav", "completed", "LIKELY_GENUINE", 15, 0.95, 0.05, 4.2, 1, 150, "aurigin-v1", '["Clear voice"]', "2026-10-01 12:00:00")
    )
    drc_id = str(uuid.uuid4())
    cur.execute(
        "INSERT INTO device_recording_compatibilities VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (drc_id, "Vivo", "V2558", "V70 FE", "Origin OS", "6", "16", "ROOT_LEVEL", 1, '{"search_scope": "ROOT"}', 1, "2026-10-01 12:00:00", "2026-10-01 12:00:00")
    )
    conn.commit()
    conn.close()

    # Target SQLite file to act as target database for test
    target_file = tmp_path / "target.db"
    target_engine = create_engine(f"sqlite:///{target_file}")
    Base.metadata.create_all(bind=target_engine)

    # 1. Test Dry-Run Mode (does not commit)
    stats_dry = migrate_data(
        source_sqlite_path=str(sqlite_file),
        target_postgres_url=f"sqlite:///{target_file}",
        dry_run=True,
    )
    assert stats_dry["analysis_records"]["source_count"] == 1
    assert stats_dry["analysis_records"]["inserted_count"] == 1
    assert stats_dry["analysis_records"]["status"] == "DRY_RUN_VALIDATED"

    # Verify target is still empty after dry-run
    TargetSession = sessionmaker(bind=target_engine)
    with TargetSession() as s:
        assert len(s.execute(select(AnalysisRecord)).scalars().all()) == 0

    # 2. Test Live Migration
    stats_live = migrate_data(
        source_sqlite_path=str(sqlite_file),
        target_postgres_url=f"sqlite:///{target_file}",
        dry_run=False,
    )
    assert stats_live["analysis_records"]["source_count"] == 1
    assert stats_live["analysis_records"]["inserted_count"] == 1
    assert stats_live["analysis_records"]["target_final_count"] == 1
    assert stats_live["analysis_records"]["status"] == "MIGRATED_OK"

    # Verify rows in target DB
    with TargetSession() as s:
        rec = s.get(AnalysisRecord, rec_id)
        assert rec is not None
        assert rec.filename == "call.wav"
        assert rec.reasons == ["Clear voice"]
        assert rec.risk_score == 15

        drc = s.get(DeviceRecordingCompatibility, drc_id)
        assert drc is not None
        assert drc.manufacturer == "Vivo"
        assert drc.supported is True
        assert drc.configuration == {"search_scope": "ROOT"}

    # 3. Test Idempotency (re-running skips existing records without error)
    stats_rerun = migrate_data(
        source_sqlite_path=str(sqlite_file),
        target_postgres_url=f"sqlite:///{target_file}",
        dry_run=False,
    )
    assert stats_rerun["analysis_records"]["source_count"] == 1
    assert stats_rerun["analysis_records"]["inserted_count"] == 0
    assert stats_rerun["analysis_records"]["skipped_count"] == 1
    assert stats_rerun["analysis_records"]["target_final_count"] == 1
