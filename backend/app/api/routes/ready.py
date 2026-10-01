"""Readiness endpoint for database and voice clone detection provider checks."""

from __future__ import annotations

from fastapi import APIRouter, status
from fastapi.responses import JSONResponse
from sqlalchemy import text

from app.core.config import get_settings
from app.db.database import engine

router = APIRouter(prefix="/ready", tags=["health"])


def _database_ready() -> bool:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


@router.get("")
def ready() -> JSONResponse:
    settings = get_settings()
    db_ok = _database_ready()

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
            "database": db_ok,
            "providers": {
                "aurigin": aurigin_state,
                "reality_defender": rd_state,
            },
            "primary_realtime_detector": "aurigin",
            "file_analysis_detector": "aurigin",
        },
    )

