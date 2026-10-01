"""Database engine and session management."""

from collections.abc import Generator
import logging

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, declarative_base, sessionmaker

from app.core.config import get_settings

logger = logging.getLogger("voice-clone-detection")
settings = get_settings()

is_sqlite = settings.database_url.startswith("sqlite")
engine_kwargs: dict = {"pool_pre_ping": True}

if is_sqlite:
    engine_kwargs["connect_args"] = {"check_same_thread": False}
else:
    engine_kwargs["pool_size"] = 10
    engine_kwargs["max_overflow"] = 20
    engine_kwargs["pool_recycle"] = 300

engine = create_engine(settings.database_url, **engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def upgrade_sqlite_schema() -> None:
    """Apply additive SQLite schema upgrades for existing Stage 1 databases."""
    if not settings.database_url.startswith("sqlite"):
        return

    with engine.begin() as connection:
        table_exists = connection.execute(
            text("SELECT name FROM sqlite_master WHERE type='table' AND name='analysis_records'")
        ).fetchone()
        if not table_exists:
            return

        table_sql = connection.execute(
            text("SELECT sql FROM sqlite_master WHERE type='table' AND name='analysis_records'")
        ).scalar()
        if table_sql and "confidence FLOAT NOT NULL" in table_sql:
            from app.db import models  # noqa: F401
            connection.execute(text("DROP INDEX IF EXISTS ix_analysis_records_id"))
            connection.execute(text("ALTER TABLE analysis_records RENAME TO analysis_records_old"))
            Base.metadata.tables["analysis_records"].create(bind=connection)
            connection.execute(text("INSERT INTO analysis_records (id, filename, stored_audio_path, status, classification, risk_score, confidence, ai_probability, duration_seconds, segments_analyzed, processing_time_ms, detector_version, reasons, created_at) SELECT id, filename, stored_audio_path, status, classification, risk_score, confidence, ai_probability, duration_seconds, segments_analyzed, processing_time_ms, detector_version, reasons, created_at FROM analysis_records_old"))
            connection.execute(text("DROP TABLE analysis_records_old"))
            return

        columns = {
            row[1]
            for row in connection.execute(text("PRAGMA table_info('analysis_records')")).fetchall()
        }

        if "ai_probability" not in columns:
            connection.execute(text("ALTER TABLE analysis_records ADD COLUMN ai_probability FLOAT"))
        if "segments_analyzed" not in columns:
            connection.execute(text("ALTER TABLE analysis_records ADD COLUMN segments_analyzed INTEGER"))
        if "processing_time_ms" not in columns:
            connection.execute(text("ALTER TABLE analysis_records ADD COLUMN processing_time_ms INTEGER"))

        # Add realtime session and device compatibility tables if missing without dropping existing data.
        for table_name in ("call_sessions", "call_chunks", "device_recording_compatibilities"):
            exists = connection.execute(
                text("SELECT name FROM sqlite_master WHERE type='table' AND name=:table_name"),
                {"table_name": table_name},
            ).fetchone()
            if not exists:
                from app.db import models  # noqa: F401
                Base.metadata.tables[table_name].create(bind=connection)
            elif table_name == "call_sessions":
                cs_columns = {
                    row[1]
                    for row in connection.execute(text("PRAGMA table_info('call_sessions')")).fetchall()
                }
                if "final_classification" not in cs_columns:
                    connection.execute(text("ALTER TABLE call_sessions ADD COLUMN final_classification VARCHAR(64) DEFAULT 'ANALYZING'"))
                if "confidence" not in cs_columns:
                    connection.execute(text("ALTER TABLE call_sessions ADD COLUMN confidence FLOAT"))
                if "chunks_analyzed" not in cs_columns:
                    connection.execute(text("ALTER TABLE call_sessions ADD COLUMN chunks_analyzed INTEGER DEFAULT 0"))
                if "total_duration_seconds" not in cs_columns:
                    connection.execute(text("ALTER TABLE call_sessions ADD COLUMN total_duration_seconds FLOAT DEFAULT 0.0"))
                if "evidence" not in cs_columns:
                    connection.execute(text("ALTER TABLE call_sessions ADD COLUMN evidence JSON"))
                if "detector_version" not in cs_columns:
                    connection.execute(text("ALTER TABLE call_sessions ADD COLUMN detector_version VARCHAR(64) DEFAULT 'aurigin-realtime'"))
            elif table_name == "device_recording_compatibilities":
                drc_columns = {
                    row[1]
                    for row in connection.execute(text("PRAGMA table_info('device_recording_compatibilities')")).fetchall()
                }
                if "marketing_name" not in drc_columns:
                    connection.execute(text("ALTER TABLE device_recording_compatibilities ADD COLUMN marketing_name VARCHAR(128)"))


def init_database() -> None:
    """Initialize database metadata and seed initial reference data safely."""
    from app.db import models  # noqa: F401
    Base.metadata.create_all(bind=engine)
    upgrade_sqlite_schema()

    # Automatically seed initial validated device recording compatibility rows
    from app.db.seed_compatibility import seed_device_compatibilities
    with SessionLocal() as seed_session:
        try:
            seed_device_compatibilities(seed_session)
        except Exception as exc:
            seed_session.rollback()
            logger.warning("Device compatibility seed warning: %s", exc)


def get_db() -> Generator[Session, None, None]:
    """Provide a DB session for request-scoped dependency injection."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
