"""Seed initial validated OEM device recording compatibility matrix."""

from __future__ import annotations

import logging
import uuid
from typing import TYPE_CHECKING

from sqlalchemy import delete, func, select

if TYPE_CHECKING:
    from sqlalchemy.orm import Session

logger = logging.getLogger("voice-clone-detection")

LEGACY_INCORRECT_KEYS = [
    ("Vivo", "T2 Pro"),
    ("Vivo", "V70 FE"),
    ("Infinix", "Note 40 Pro+"),
    ("Infinix", "X6851B"),
]

VALIDATED_DEVICE_SEEDS = [
    {
        "manufacturer": "Vivo",
        "model": "V2558",
        "marketing_name": "V70 FE",
        "os_family": "Origin OS",
        "os_version": "6",
        "android_version": "16",
        "recording_profile": "ROOT_LEVEL",
        "supported": True,
        "configuration": {
            "search_scope": "ROOT",
            "extensions": [".m4a"],
        },
        "profile_version": 1,
    },
    {
        "manufacturer": "Vivo",
        "model": "V2321",
        "marketing_name": "T2 Pro",
        "os_family": "Funtouch OS",
        "os_version": "15",
        "android_version": "15",
        "recording_profile": "ROOT_LEVEL",
        "supported": True,
        "configuration": {
            "search_scope": "ROOT",
            "extensions": [".m4a"],
        },
        "profile_version": 1,
    },
    {
        "manufacturer": "Infinix",
        "model": "Infinix X6851B",
        "marketing_name": "Note 40 Pro+",
        "os_family": "XOS",
        "os_version": "15",
        "android_version": "15",
        "recording_profile": "SUB_FOLDER",
        "supported": True,
        "configuration": {
            "search_scope": "SUB_FOLDER",
            "recording_folder": "Music/PhoneRecord",
            "extensions": [".aac"],
        },
        "profile_version": 1,
    },
]


def seed_device_compatibilities(session: Session) -> None:
    """Seed initial validated device recording compatibility rows idempotently and purge legacy records."""
    from app.db.models import DeviceRecordingCompatibility

    # Purge legacy / incorrect marketing-name records so they cannot accidentally match
    for legacy_mfg, legacy_model in LEGACY_INCORRECT_KEYS:
        session.execute(
            delete(DeviceRecordingCompatibility).where(
                func.lower(DeviceRecordingCompatibility.manufacturer) == legacy_mfg.lower(),
                func.lower(DeviceRecordingCompatibility.model) == legacy_model.lower(),
            )
        )

    for seed_item in VALIDATED_DEVICE_SEEDS:
        existing = session.execute(
            select(DeviceRecordingCompatibility).where(
                func.lower(DeviceRecordingCompatibility.manufacturer) == seed_item["manufacturer"].lower(),
                func.lower(DeviceRecordingCompatibility.model) == seed_item["model"].lower(),
            )
        ).scalar_one_or_none()

        if existing is None:
            record = DeviceRecordingCompatibility(
                id=str(uuid.uuid4()),
                manufacturer=seed_item["manufacturer"],
                model=seed_item["model"],
                marketing_name=seed_item.get("marketing_name"),
                os_family=seed_item["os_family"],
                os_version=seed_item["os_version"],
                android_version=seed_item.get("android_version"),
                recording_profile=seed_item["recording_profile"],
                supported=seed_item["supported"],
                configuration=seed_item["configuration"],
                profile_version=seed_item.get("profile_version", 1),
            )
            session.add(record)
            logger.info(
                "Seeded device recording compatibility: %s %s (%s) -> %s",
                seed_item["manufacturer"],
                seed_item["model"],
                seed_item.get("marketing_name", "N/A"),
                seed_item["recording_profile"],
            )
        else:
            # Update configuration while preserving existing higher profile version if present
            existing.marketing_name = seed_item.get("marketing_name")
            existing.os_family = seed_item["os_family"]
            existing.os_version = seed_item["os_version"]
            existing.android_version = seed_item.get("android_version")
            existing.recording_profile = seed_item["recording_profile"]
            existing.supported = seed_item["supported"]
            existing.configuration = seed_item["configuration"]
            if existing.profile_version is None:
                existing.profile_version = seed_item.get("profile_version", 1)

    session.commit()
