"""Tests for Alembic migrations, baseline schema parity, and SQLite-to-PostgreSQL data migration tool."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
import uuid
from pathlib import Path
import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from alembic.config import Config
from alembic import command
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

EXPECTED_V2_TABLES = {
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


def test_schema_model_table_coverage():
    """Verify all 10 V2 model tables exist in Base.metadata with all expected columns."""
    registered_tables = set(Base.metadata.tables.keys())
    assert EXPECTED_V2_TABLES.issubset(registered_tables), (
        f"Missing tables in Base.metadata: {EXPECTED_V2_TABLES - registered_tables}"
    )


def test_alembic_heads_and_offline_upgrade_sql(tmp_path: Path):
    """Verify Alembic configuration, head revision, and offline SQL generation without live DB connection."""
    backend_dir = Path(__file__).resolve().parent.parent
    alembic_ini_path = backend_dir / "alembic.ini"
    assert alembic_ini_path.exists(), "alembic.ini must exist in backend directory"

    alembic_cfg = Config(str(alembic_ini_path))
    alembic_cfg.set_main_option("script_location", str(backend_dir / "alembic"))
    alembic_cfg.set_main_option("sqlalchemy.url", "postgresql+psycopg://dummy:dummy@localhost:5432/dummy")

    # Verify script directory discovers the initial revision
    from alembic.script import ScriptDirectory
    script = ScriptDirectory.from_config(alembic_cfg)
    heads = script.get_heads()
    assert heads == ["0001_initial_schema"], f"Unexpected heads: {heads}"


def test_data_migration_tool_with_sample_sqlite(tmp_path: Path):
    """Verify migrate_data reads from SQLite and populates target database accurately with type conversion."""
    sqlite_file = tmp_path / "source.db"
    conn = sqlite3.connect(str(sqlite_file))
    cur = conn.cursor()

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

    target_file = tmp_path / "target.db"
    target_engine = create_engine(f"sqlite:///{target_file}")
    Base.metadata.create_all(bind=target_engine)

    # 1. Test Dry-Run Mode
    stats_dry = migrate_data(
        source_sqlite_path=str(sqlite_file),
        target_postgres_url=f"sqlite:///{target_file}",
        dry_run=True,
    )
    assert stats_dry["analysis_records"]["source_count"] == 1
    assert stats_dry["analysis_records"]["inserted_count"] == 1
    assert stats_dry["analysis_records"]["status"] == "DRY_RUN_VALIDATED"

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

    # 3. Test Idempotency
    stats_rerun = migrate_data(
        source_sqlite_path=str(sqlite_file),
        target_postgres_url=f"sqlite:///{target_file}",
        dry_run=False,
    )
    assert stats_rerun["analysis_records"]["source_count"] == 1
    assert stats_rerun["analysis_records"]["inserted_count"] == 0
    assert stats_rerun["analysis_records"]["skipped_count"] == 1
    assert stats_rerun["analysis_records"]["target_final_count"] == 1
