"""Readiness endpoint for database and voice clone detection provider checks."""

from __future__ import annotations

from typing import Any
from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.db.database import engine

router = APIRouter(prefix="/ready", tags=["health"])


def _get_database_status() -> dict[str, Any]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
            dialect = engine.dialect.name
            migration_revision: str | None = None
            try:
                res = connection.execute(
                    text("SELECT version_num FROM alembic_version LIMIT 1")
                ).scalar()
                migration_revision = str(res) if res else "uninitialized"
            except Exception:
                migration_revision = "uninitialized"
            return {
                "connected": True,
                "engine": dialect,
                "migration_revision": migration_revision,
            }
    except Exception:
        return {
            "connected": False,
            "engine": engine.dialect.name if hasattr(engine, "dialect") else "unknown",
            "migration_revision": "unavailable",
        }


@router.get("")
def ready() -> JSONResponse:
    settings = get_settings()
    db_status = _get_database_status()
    db_ok = db_status["connected"]

    aurigin_configured = bool(settings.aurigin_api_key) or settings.aurigin_enabled
    rd_configured = bool(settings.reality_defender_api_key)

    aurigin_state = "configured" if aurigin_configured else "not_configured"
    rd_state = "configured" if rd_configured else "not_configured"

    all_ready = db_ok
    status_code = status.HTTP_200_OK if all_ready else status.HTTP_503_SERVICE_UNAVAILABLE

    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ready" if all_ready else "not_ready",
            "version": "v2",
            "database": "connected" if db_ok else "disconnected",
            "database_engine": db_status["engine"],
            "migration_revision": db_status["migration_revision"],
            "providers": {
                "aurigin": aurigin_state,
                "reality_defender": rd_state,
            },
            "primary_realtime_detector": "aurigin",
            "file_analysis_detector": "aurigin",
        },
    )
