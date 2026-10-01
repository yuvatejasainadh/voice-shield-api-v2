"""Device recording compatibility endpoints."""

from __future__ import annotations

import logging
import uuid
from fastapi import APIRouter, Depends, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.device_compatibility import (
    DeviceCompatibilityRequest,
    DeviceCompatibilityResponse,
)
from app.services.device_compatibility_service import DeviceCompatibilityService

router = APIRouter(prefix="/device", tags=["device"])
logger = logging.getLogger("voice-clone-detection")


@router.post(
    "/compatibility",
    response_model=DeviceCompatibilityResponse,
    status_code=status.HTTP_200_OK,
)
def check_device_compatibility(
    payload: DeviceCompatibilityRequest,
    db: Session = Depends(get_db),
) -> DeviceCompatibilityResponse:
    """Resolve recording acquisition compatibility and profile for client device."""
    request_id = f"REQ_COMPAT_{uuid.uuid4().hex[:8]}"
    return DeviceCompatibilityService.resolve_compatibility(
        request=payload,
        db=db,
        request_id=request_id,
    )
