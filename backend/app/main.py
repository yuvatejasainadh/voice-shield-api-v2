"""FastAPI application entrypoint."""

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
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.db.database import init_database
from app.db import models  # noqa: F401


settings = get_settings()
configure_logging(settings.log_level)
logger = logging.getLogger("voice-clone-detection")


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize persistent storage and database schema at startup."""
    init_database()

    if settings.aurigin_api_key or settings.aurigin_enabled:
        logger.info("Aurigin real-time detection configured (base_url=%s)", settings.aurigin_api_base_url)
    else:
        logger.info("Aurigin real-time detector initialized")

    if settings.reality_defender_api_key:
        logger.info("Reality Defender RealAPI configured (base_url=%s)", settings.reality_defender_api_base_url)
    else:
        logger.warning("Reality Defender API key is not configured")

    logger.info("Application startup complete")
    yield


app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    docs_url="/docs",
    redoc_url="/redoc",
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


# V1 API Routes
app.include_router(health.router, prefix="/api/v1")
app.include_router(ready.router, prefix="/api/v1")
app.include_router(analysis.router, prefix="/api/v1")
app.include_router(history.router, prefix="/api/v1")
app.include_router(reports.router, prefix="/api/v1")
app.include_router(realtime.router, prefix="/api/v1")
app.include_router(realtime_ws.router, prefix="/api/v1")
app.include_router(device.router, prefix="/api/v1")
app.include_router(auth.router, prefix="/api/v1")
app.include_router(cybercrime.router, prefix="/api/v1")

# Aliases for direct /api and root web endpoints
app.include_router(auth.router, prefix="/api")
app.include_router(cybercrime.router, prefix="/api")
app.include_router(device.router, prefix="/api")
app.include_router(analysis.router, prefix="/api")
app.include_router(analysis.router)
