"""Device recording compatibility resolution service."""

from __future__ import annotations

import logging
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.models import DeviceRecordingCompatibility
from app.schemas.device_compatibility import (
    DeviceCompatibilityRequest,
    DeviceCompatibilityResponse,
    DeviceRecordingConfig,
)

logger = logging.getLogger("voice-clone-detection")


class DeviceCompatibilityService:
    """Authoritative resolver for device recording discovery configurations."""

    @staticmethod
    def resolve_compatibility(
        request: DeviceCompatibilityRequest,
        db: Session,
        request_id: str | None = None,
    ) -> DeviceCompatibilityResponse:
        """Resolve compatibility for the requested device against authoritative DB records."""
        req_mfg = request.manufacturer.strip()
        req_model = request.model.strip()

        logger.info(
            "[DEVICE_COMPATIBILITY_REQUEST] request_id=%s manufacturer=%s model=%s os_family=%s os_version=%s android_version=%s",
            request_id or "N/A",
            req_mfg,
            req_model,
            request.os_family,
            request.os_version,
            request.android_version,
        )

        # Case-insensitive query against validated database records using manufacturer + model
        stmt = select(DeviceRecordingCompatibility).where(
            func.lower(DeviceRecordingCompatibility.manufacturer) == req_mfg.lower(),
            func.lower(DeviceRecordingCompatibility.model) == req_model.lower(),
        )
        record = db.execute(stmt).scalar_one_or_none()

        if record is None or not record.supported:
            logger.warning(
                "[DEVICE_COMPATIBILITY_UNSUPPORTED] request_id=%s manufacturer=%s model=%s reason=%s",
                request_id or "N/A",
                req_mfg,
                req_model,
                "NO_MATCHING_RECORD" if record is None else "EXPLICITLY_UNSUPPORTED",
            )
            return DeviceCompatibilityResponse(
                supported=False,
                profile=None,
                configuration=None,
                profile_version=None,
            )

        config_dto = DeviceRecordingConfig(**record.configuration)

        logger.info(
            "[DEVICE_COMPATIBILITY_MATCH] request_id=%s manufacturer=%s model=%s resolved_profile=%s recording_folder=%s supported=true profile_version=%s",
            request_id or "N/A",
            record.manufacturer,
            record.model,
            record.recording_profile,
            config_dto.recording_folder or "unresolved",
            record.profile_version,
        )

        return DeviceCompatibilityResponse(
            supported=True,
            profile=record.recording_profile,
            configuration=config_dto,
            profile_version=record.profile_version,
        )
