"""Healthcheck endpoint for VoiceShield API V2."""

from fastapi import APIRouter

router = APIRouter(prefix="/health", tags=["health"])


@router.get("")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "version": "v2",
        "service": "voiceshield-api",
    }
