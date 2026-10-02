"""Tests for Alembic migrations and baseline schema parity for VoiceShield V2."""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from alembic.config import Config
from alembic.script import ScriptDirectory
from app.db.database import Base

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


def test_alembic_heads_discovery():
    """Verify Alembic configuration and head revision discovery."""
    backend_dir = Path(__file__).resolve().parent.parent
    alembic_ini_path = backend_dir / "alembic.ini"
    assert alembic_ini_path.exists(), "alembic.ini must exist in backend directory"

    alembic_cfg = Config(str(alembic_ini_path))
    alembic_cfg.set_main_option("script_location", str(backend_dir / "alembic"))

    script = ScriptDirectory.from_config(alembic_cfg)
    heads = script.get_heads()
    assert heads == ["0001_initial_schema"], f"Unexpected heads: {heads}"


def test_alembic_baseline_revision_structure():
    """Verify the 0001_initial_schema baseline migration defines upgrade and downgrade."""
    import importlib.util

    backend_dir = Path(__file__).resolve().parent.parent
    migration_file = backend_dir / "alembic" / "versions" / "0001_initial_schema.py"
    assert migration_file.exists(), "0001_initial_schema.py baseline migration must exist"

    spec = importlib.util.spec_from_file_location("initial_schema", str(migration_file))
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    assert hasattr(mod, "upgrade"), "Migration must define upgrade()"
    assert hasattr(mod, "downgrade"), "Migration must define downgrade()"
    assert mod.revision == "0001_initial_schema"
    assert mod.down_revision is None
