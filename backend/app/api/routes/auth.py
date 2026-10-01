"""Authentication endpoints for Firebase authenticated identities."""

from __future__ import annotations

import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.auth import AuthSyncRequest, AuthSyncResponse, UserSchema
from app.services.auth_service import AuthService

router = APIRouter(prefix="/auth", tags=["auth"])
logger = logging.getLogger("voice-clone-detection")


@router.post("/sync", response_model=AuthSyncResponse, status_code=status.HTTP_200_OK)
def sync_authenticated_user(
    payload: AuthSyncRequest,
    db: Session = Depends(get_db),
) -> AuthSyncResponse:
    """Synchronize Firebase authenticated identity, maintain profile, register device, and record auth event."""
    return AuthService.sync_user(db=db, payload=payload)


@router.get("/user", response_model=UserSchema)
def get_user_profile(
    firebase_uid: str | None = Query(default=None),
    user_id: uuid.UUID | None = Query(default=None),
    db: Session = Depends(get_db),
) -> UserSchema:
    """Retrieve user profile by firebase_uid or user_id."""
    if firebase_uid:
        user = AuthService.get_user_by_firebase_uid(db=db, firebase_uid=firebase_uid)
    elif user_id:
        user = AuthService.get_user_by_id(db=db, user_id=user_id)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={"code": "MISSING_IDENTIFIER", "message": "Either firebase_uid or user_id must be provided"},
        )

    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "USER_NOT_FOUND", "message": "User not found"},
        )

    return UserSchema.model_validate(user)
