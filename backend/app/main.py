"""FastAPI application entrypoint for VoiceShield API V2."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import (
    analysis,
    auth,
    cybercrime,
    device,
    health,
    history,
    ready,
    realtime,
    realtime_ws,
    reports,
)
from app.core.config import API_PREFIX, API_VERSION, get_settings
from app.core.logging import configure_logging
from app.db.database import init_database
from app.db import models  # noqa: F401


settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("voice-clone-detection")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize persistent storage and database schema at startup."""
    logger.info("Starting VoiceShield API V2 (environment=%s, dialect=%s)", settings.environment, "sqlite" if settings.is_sqlite else "postgresql")
    init_database()

    if settings.aurigin_api_key or settings.aurigin_enabled:
        logger.info("Aurigin real-time detection configured (base_url=%s)", settings.aurigin_api_base_url)
    else:
        logger.info("Aurigin real-time detector initialized")

    if settings.reality_defender_api_key:
        logger.info("Reality Defender RealAPI configured (base_url=%s)", settings.reality_defender_api_base_url)
    else:
        logger.warning("Reality Defender API key is not configured")

    logger.info("VoiceShield API V2 startup complete")
    yield


app = FastAPI(
    title=settings.app_name,
    description="VoiceShield V2 Backend API for real-time and post-call voice clone & deepfake audio detection, device recording compatibility, Firebase auth, and cybercrime intelligence.",
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(_: Request, exc: RequestValidationError) -> JSONResponse:
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content={"detail": "Invalid request", "errors": exc.errors()},
    )


@app.exception_handler(Exception)
async def generic_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error: %s", exc)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal processing error"},
    )


@app.get("/", tags=["meta"])
def root() -> dict[str, str]:
    """Root metadata endpoint identifying VoiceShield API V2."""
    return {
        "name": "VoiceShield API",
        "version": "V2",
        "status": "ok",
        "docs": "/docs",
    }


# Canonical V2 API Route Registrations
app.include_router(health.router, prefix=API_PREFIX)
app.include_router(ready.router, prefix=API_PREFIX)
app.include_router(auth.router, prefix=API_PREFIX)
app.include_router(cybercrime.router, prefix=API_PREFIX)
app.include_router(analysis.router, prefix=API_PREFIX)
app.include_router(history.router, prefix=API_PREFIX)
app.include_router(reports.router, prefix=API_PREFIX)
app.include_router(realtime.router, prefix=API_PREFIX)
app.include_router(realtime_ws.router, prefix=API_PREFIX)
app.include_router(device.router, prefix=API_PREFIX)
