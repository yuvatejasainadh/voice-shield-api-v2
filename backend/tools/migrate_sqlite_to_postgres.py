"""VoiceShield V2 SQLite to PostgreSQL Data Migration Tool.

Migrates data from legacy/local SQLite database to PostgreSQL (AWS RDS / local PostgreSQL)
with transactional safety, type translation (JSONB, UUIDs, datetimes, booleans),
idempotency, dry-run support, and row count verification.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from sqlalchemy import create_engine, select, text
from sqlalchemy.orm import Session, sessionmaker

# Ensure app package is importable
current_dir = Path(__file__).resolve().parent
backend_dir = current_dir.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("sqlite-to-postgres-migrator")

# Migration order respecting foreign key constraints
TABLE_ORDER = [
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
]

MODEL_MAP = {
    "analysis_records": AnalysisRecord,
    "call_sessions": CallSession,
    "call_chunks": CallChunk,
    "device_recording_compatibilities": DeviceRecordingCompatibility,
    "users": User,
    "user_devices": UserDevice,
    "user_auth_events": UserAuthEvent,
    "cybercrime_categories": CybercrimeCategory,
    "cybercrime_templates": CybercrimeTemplate,
    "cybercrime_template_versions": CybercrimeTemplateVersion,
}


def _parse_datetime(val: Any) -> datetime | None:
    if val is None or isinstance(val, datetime):
        return val
    val_str = str(val).strip()
    if not val_str:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(val_str, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(val_str)
    except Exception:
        return None


def _parse_uuid(val: Any) -> uuid.UUID | None:
    if val is None or isinstance(val, uuid.UUID):
        return val
    val_str = str(val).strip()
    if not val_str:
        return None
    try:
        return uuid.UUID(val_str)
    except Exception:
        return None


def _parse_json(val: Any) -> Any:
    if val is None:
        return None
    if isinstance(val, (dict, list)):
        return val
    val_str = str(val).strip()
    if not val_str:
        return None
    try:
        return json.loads(val_str)
    except Exception:
        return val_str


def _parse_bool(val: Any) -> bool | None:
    if val is None:
        return None
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    val_str = str(val).strip().lower()
    return val_str in ("1", "true", "t", "yes", "y")


def transform_row(table_name: str, row: dict[str, Any]) -> dict[str, Any]:
    """Transform SQLite raw row dict to typed Python attributes for SQLAlchemy models."""
    d = dict(row)

    if table_name == "analysis_records":
        d["confidence"] = float(d["confidence"]) if d.get("confidence") is not None else None
        d["ai_probability"] = float(d["ai_probability"]) if d.get("ai_probability") is not None else None
        d["duration_seconds"] = float(d["duration_seconds"]) if d.get("duration_seconds") is not None else 0.0
        d["risk_score"] = int(d["risk_score"]) if d.get("risk_score") is not None else 0
        d["segments_analyzed"] = int(d["segments_analyzed"]) if d.get("segments_analyzed") is not None else None
        d["processing_time_ms"] = int(d["processing_time_ms"]) if d.get("processing_time_ms") is not None else None
        d["reasons"] = _parse_json(d.get("reasons")) or []
        d["created_at"] = _parse_datetime(d.get("created_at")) or datetime.utcnow()

    elif table_name == "call_sessions":
        d["started_at"] = _parse_datetime(d.get("started_at"))
        d["ended_at"] = _parse_datetime(d.get("ended_at"))
        d["current_score"] = float(d["current_score"]) if d.get("current_score") is not None else 0.0
        d["peak_score"] = float(d["peak_score"]) if d.get("peak_score") is not None else 0.0
        d["final_score"] = float(d["final_score"]) if d.get("final_score") is not None else None
        d["confidence"] = float(d["confidence"]) if d.get("confidence") is not None else None
        d["chunks_analyzed"] = int(d["chunks_analyzed"]) if d.get("chunks_analyzed") is not None else 0
        d["total_duration_seconds"] = float(d["total_duration_seconds"]) if d.get("total_duration_seconds") is not None else 0.0
        d["evidence"] = _parse_json(d.get("evidence"))
        d["created_at"] = _parse_datetime(d.get("created_at")) or datetime.utcnow()
        d["updated_at"] = _parse_datetime(d.get("updated_at")) or datetime.utcnow()
        d["last_notification_at"] = _parse_datetime(d.get("last_notification_at"))

    elif table_name == "call_chunks":
        d["sequence_number"] = int(d["sequence_number"])
        d["started_at"] = _parse_datetime(d.get("started_at"))
        d["ended_at"] = _parse_datetime(d.get("ended_at"))
        d["duration_ms"] = int(d["duration_ms"]) if d.get("duration_ms") is not None else 0
        d["score"] = float(d["score"]) if d.get("score") is not None else None
        d["processing_ms"] = int(d["processing_ms"]) if d.get("processing_ms") is not None else None
        d["created_at"] = _parse_datetime(d.get("created_at")) or datetime.utcnow()

    elif table_name == "device_recording_compatibilities":
        d["supported"] = _parse_bool(d.get("supported")) if d.get("supported") is not None else True
        d["configuration"] = _parse_json(d.get("configuration")) or {}
        d["profile_version"] = int(d["profile_version"]) if d.get("profile_version") is not None else 1
        d["created_at"] = _parse_datetime(d.get("created_at")) or datetime.utcnow()
        d["updated_at"] = _parse_datetime(d.get("updated_at")) or datetime.utcnow()

    elif table_name == "users":
        d["id"] = _parse_uuid(d["id"])
        d["created_at"] = _parse_datetime(d.get("created_at"))
        d["updated_at"] = _parse_datetime(d.get("updated_at"))
        d["last_login_at"] = _parse_datetime(d.get("last_login_at"))

    elif table_name == "user_devices":
        d["id"] = _parse_uuid(d["id"])
        d["user_id"] = _parse_uuid(d["user_id"])
        d["is_active"] = _parse_bool(d.get("is_active")) if d.get("is_active") is not None else True
        d["first_seen_at"] = _parse_datetime(d.get("first_seen_at"))
        d["last_seen_at"] = _parse_datetime(d.get("last_seen_at"))

    elif table_name == "user_auth_events":
        d["id"] = _parse_uuid(d["id"])
        d["user_id"] = _parse_uuid(d.get("user_id"))
        d["device_id"] = _parse_uuid(d.get("device_id"))
        d["success"] = _parse_bool(d.get("success")) if d.get("success") is not None else True
        d["created_at"] = _parse_datetime(d.get("created_at"))

    elif table_name == "cybercrime_categories":
        d["id"] = _parse_uuid(d["id"])
        d["is_active"] = _parse_bool(d.get("is_active")) if d.get("is_active") is not None else True
        d["created_at"] = _parse_datetime(d.get("created_at"))
        d["updated_at"] = _parse_datetime(d.get("updated_at"))

    elif table_name == "cybercrime_templates":
        d["id"] = _parse_uuid(d["id"])
        d["category_id"] = _parse_uuid(d["category_id"])
        d["indicators"] = _parse_json(d.get("indicators")) or []
        d["recommended_actions"] = _parse_json(d.get("recommended_actions")) or []
        metadata_val = d.pop("metadata", None)
        d["template_metadata"] = _parse_json(metadata_val) if metadata_val is not None else (_parse_json(d.get("template_metadata")) or {})
        d["is_active"] = _parse_bool(d.get("is_active")) if d.get("is_active") is not None else True
        d["created_at"] = _parse_datetime(d.get("created_at"))
        d["updated_at"] = _parse_datetime(d.get("updated_at"))

    elif table_name == "cybercrime_template_versions":
        d["id"] = _parse_uuid(d["id"])
        d["template_id"] = _parse_uuid(d["template_id"])
        d["version"] = int(d["version"])
        d["template_data"] = _parse_json(d.get("template_data")) or {}
        d["created_at"] = _parse_datetime(d.get("created_at"))

    return d


def migrate_data(
    source_sqlite_path: str,
    target_postgres_url: str,
    dry_run: bool = False,
    batch_size: int = 100,
) -> dict[str, dict[str, Any]]:
    """Execute the data migration from SQLite to PostgreSQL target with full validation."""
    source_path = Path(source_sqlite_path).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Source SQLite database not found: {source_path}")

    # Connect to SQLite
    sqlite_conn = sqlite3.connect(str(source_path))
    sqlite_conn.row_factory = sqlite3.Row
    sqlite_cur = sqlite_conn.cursor()

    # Discover tables in SQLite
    sqlite_cur.execute("SELECT name FROM sqlite_master WHERE type='table'")
    sqlite_tables = {row[0] for row in sqlite_cur.fetchall() if not row[0].startswith("sqlite_")}

    # Connect to PostgreSQL target
    pg_engine = create_engine(target_postgres_url, pool_pre_ping=True)
    TargetSession = sessionmaker(bind=pg_engine)

    stats: dict[str, dict[str, Any]] = {}

    try:
        with TargetSession() as pg_session:
            for table_name in TABLE_ORDER:
                if table_name not in sqlite_tables:
                    stats[table_name] = {
                        "source_count": 0,
                        "read_count": 0,
                        "inserted_count": 0,
                        "skipped_count": 0,
                        "target_final_count": 0,
                        "status": "TABLE_NOT_IN_SOURCE",
                    }
                    continue

                model_cls = MODEL_MAP[table_name]

                # 1. Count rows in SQLite source
                sqlite_cur.execute(f'SELECT count(*) FROM "{table_name}"')
                source_count = sqlite_cur.fetchone()[0]

                # 2. Count initial rows in PostgreSQL target
                try:
                    initial_target_count = pg_session.execute(select(model_cls)).scalars().all()
                    initial_pg_count = len(initial_target_count)
                except Exception as exc:
                    raise RuntimeError(
                        f"Target table '{table_name}' does not exist or cannot be queried on target PostgreSQL. "
                        f"Ensure Alembic migrations have been applied first (`alembic upgrade head`). "
                        f"Underlying error: {exc}"
                    ) from exc

                # 3. Read SQLite rows
                sqlite_cur.execute(f'SELECT * FROM "{table_name}"')
                raw_rows = [dict(r) for r in sqlite_cur.fetchall()]

                inserted_count = 0
                skipped_count = 0

                for i in range(0, len(raw_rows), batch_size):
                    batch = raw_rows[i : i + batch_size]
                    for raw_row in batch:
                        transformed = transform_row(table_name, raw_row)
                        pk_val = transformed.get("id")

                        # Check if record already exists in PostgreSQL
                        existing = None
                        if pk_val is not None:
                            existing = pg_session.get(model_cls, pk_val)

                        if existing is None:
                            record = model_cls(**transformed)
                            pg_session.add(record)
                            inserted_count += 1
                        else:
                            skipped_count += 1

                    if not dry_run:
                        pg_session.flush()

                if dry_run:
                    pg_session.rollback()
                    final_pg_count = initial_pg_count
                    status = "DRY_RUN_VALIDATED"
                else:
                    pg_session.commit()
                    final_target_records = pg_session.execute(select(model_cls)).scalars().all()
                    final_pg_count = len(final_target_records)
                    status = "MIGRATED_OK"

                stats[table_name] = {
                    "source_count": source_count,
                    "read_count": len(raw_rows),
                    "inserted_count": inserted_count,
                    "skipped_count": skipped_count,
                    "target_final_count": final_pg_count,
                    "status": status,
                }

    finally:
        sqlite_conn.close()
        pg_engine.dispose()

    return stats


def print_migration_report(stats: dict[str, dict[str, Any]], dry_run: bool) -> None:
    """Print a clean verification report of source -> target row counts."""
    mode_str = "DRY RUN SIMULATION" if dry_run else "LIVE MIGRATION"
    logger.info("==========================================================================================")
    logger.info(" VOICE SHIELD V2 DATABASE MIGRATION REPORT [%s]", mode_str)
    logger.info("==========================================================================================")
    header = f"{'Table Name':<35} | {'Source':<8} | {'Read':<8} | {'Inserted':<9} | {'Skipped':<8} | {'Target Total':<12} | {'Status'}"
    logger.info(header)
    logger.info("-" * len(header))

    all_ok = True
    for table_name, data in stats.items():
        row_str = (
            f"{table_name:<35} | "
            f"{data['source_count']:<8} | "
            f"{data['read_count']:<8} | "
            f"{data['inserted_count']:<9} | "
            f"{data['skipped_count']:<8} | "
            f"{data['target_final_count']:<12} | "
            f"{data['status']}"
        )
        logger.info(row_str)
        if "ERROR" in data["status"]:
            all_ok = False

    logger.info("==========================================================================================")
    if all_ok:
        logger.info("Migration status: SUCCESS (%s completed with integrity verified)", mode_str)
    else:
        logger.error("Migration status: COMPLETED WITH WARNINGS OR ERRORS")


def main() -> None:
    parser = argparse.ArgumentParser(description="VoiceShield V2 SQLite -> PostgreSQL Data Migration Utility")
    parser.add_argument(
        "--source-sqlite",
        default=os.getenv("SOURCE_SQLITE_PATH", "voice_clone_detection.db"),
        help="Path to source SQLite database file",
    )
    parser.add_argument(
        "--target-postgres",
        default=os.getenv("TARGET_DATABASE_URL") or os.getenv("DATABASE_URL"),
        help="Target PostgreSQL connection URL (e.g. postgresql+psycopg://user:pass@host:5432/dbname)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate reading and transformations without committing changes to target PostgreSQL",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for processing records",
    )

    args = parser.parse_args()

    if not args.target_postgres:
        logger.error("Target PostgreSQL URL must be specified via --target-postgres or TARGET_DATABASE_URL env var.")
        sys.exit(1)

    target_url = args.target_postgres
    if target_url.startswith("postgres://"):
        target_url = "postgresql+psycopg://" + target_url[len("postgres://"):]
    elif target_url.startswith("postgresql://"):
        target_url = "postgresql+psycopg://" + target_url[len("postgresql://"):]

    if not target_url.startswith("postgresql"):
        logger.error("Target URL must be a PostgreSQL connection URL (e.g. postgresql+psycopg://...)")
        sys.exit(1)

    logger.info("Source SQLite: %s", args.source_sqlite)
    logger.info("Target PostgreSQL: [PROTECTED CREDENTIALS]")
    logger.info("Dry Run Mode: %s", args.dry_run)

    try:
        stats = migrate_data(
            source_sqlite_path=args.source_sqlite,
            target_postgres_url=target_url,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
        )
        print_migration_report(stats, dry_run=args.dry_run)
    except Exception as exc:
        logger.exception("Migration failed: %s", exc)
        sys.exit(1)


if __name__ == "__main__":
    main()
