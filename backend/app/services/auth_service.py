"""Authentication and user management service for Firebase authenticated identities."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.models import User, UserAuthEvent, UserDevice
from app.schemas.auth import AuthSyncRequest, AuthSyncResponse, UserDeviceSchema, UserSchema

logger = logging.getLogger("voice-clone-detection")


class AuthService:
    """Handles synchronization of Firebase authenticated users, devices, and auth audit events."""

    @staticmethod
    def sync_user(db: Session, payload: AuthSyncRequest) -> AuthSyncResponse:
        """Find or create user by firebase_uid or phone_number, update last_login_at, register device, and record auth event."""
        now = datetime.now(timezone.utc)

        # 1. Look up user by firebase_uid
        user = db.execute(
            select(User).where(User.firebase_uid == payload.firebase_uid)
        ).scalar_one_or_none()

        if user is None:
            # Check by phone number in case firebase_uid changed or was linked
            user = db.execute(
                select(User).where(User.phone_number == payload.phone_number)
            ).scalar_one_or_none()

            if user is not None:
                user.firebase_uid = payload.firebase_uid
                if payload.display_name:
                    user.display_name = payload.display_name
                user.last_login_at = now
                user.updated_at = now
            else:
                user = User(
                    id=uuid.uuid4(),
                    firebase_uid=payload.firebase_uid,
                    phone_number=payload.phone_number,
                    display_name=payload.display_name,
                    status="ACTIVE",
                    created_at=now,
                    updated_at=now,
                    last_login_at=now,
                )
                db.add(user)
                db.flush()
        else:
            user.last_login_at = now
            user.updated_at = now
            if payload.display_name:
                user.display_name = payload.display_name
            if payload.phone_number:
                user.phone_number = payload.phone_number

        # 2. Optionally register or update user device
        device_id: uuid.UUID | None = None
        if payload.device_fingerprint:
            device = db.execute(
                select(UserDevice).where(UserDevice.device_fingerprint == payload.device_fingerprint)
            ).scalar_one_or_none()

            if device is None:
                device = UserDevice(
                    id=uuid.uuid4(),
                    user_id=user.id,
                    device_fingerprint=payload.device_fingerprint,
                    manufacturer=payload.manufacturer,
                    model=payload.model,
                    android_version=payload.android_version,
                    app_version=payload.app_version,
                    first_seen_at=now,
                    last_seen_at=now,
                    is_active=True,
                )
                db.add(device)
                db.flush()
            else:
                device.user_id = user.id
                device.last_seen_at = now
                device.is_active = True
                if payload.manufacturer:
                    device.manufacturer = payload.manufacturer
                if payload.model:
                    device.model = payload.model
                if payload.android_version:
                    device.android_version = payload.android_version
                if payload.app_version:
                    device.app_version = payload.app_version

            device_id = device.id

        # 3. Record authentication audit event
        auth_event = UserAuthEvent(
            id=uuid.uuid4(),
            user_id=user.id,
            firebase_uid=payload.firebase_uid,
            phone_number=payload.phone_number,
            event_type=payload.event_type,
            success=True,
            device_id=device_id,
            created_at=now,
        )
        db.add(auth_event)

        db.commit()
        db.refresh(user)

        user_dto = UserSchema.model_validate(user)
        logger.info(
            "[AUTH_SYNC_SUCCESS] user_id=%s firebase_uid=%s phone=%s device_id=%s",
            user.id,
            user.firebase_uid,
            user.phone_number,
            device_id,
        )

        return AuthSyncResponse(
            success=True,
            user=user_dto,
            device_id=device_id,
            message="User authenticated and synchronized successfully",
        )

    @staticmethod
    def get_user_by_firebase_uid(db: Session, firebase_uid: str) -> User | None:
        return db.execute(
            select(User).where(User.firebase_uid == firebase_uid)
        ).scalar_one_or_none()

    @staticmethod
    def get_user_by_id(db: Session, user_id: uuid.UUID) -> User | None:
        return db.execute(
            select(User).where(User.id == user_id)
        ).scalar_one_or_none()
