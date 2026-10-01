"""Healthcheck endpoint."""

from fastapi import APIRouter

from app.core.config import get_settings


router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health() -> dict[str, str]:
    settings = get_settings()
    return {
        "status": "healthy",
        "service": "voice-clone-detection-api",
        "version": settings.app_version,
    }
