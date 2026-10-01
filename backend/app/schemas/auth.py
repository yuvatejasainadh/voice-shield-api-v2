"""Pydantic schemas for Firebase user authentication and device sync."""

from __future__ import annotations

import uuid
from datetime import datetime
from pydantic import BaseModel, ConfigDict, Field


class AuthSyncRequest(BaseModel):
    firebase_uid: str = Field(..., min_length=1, max_length=128, description="Firebase Authenticated UID")
    phone_number: str = Field(..., min_length=1, max_length=20, description="E.164 verified phone number")
    display_name: str | None = Field(default=None, max_length=128, description="User display name")
    device_fingerprint: str | None = Field(default=None, max_length=255, description="Client hardware/app fingerprint")
    manufacturer: str | None = Field(default=None, max_length=128, description="Device manufacturer")
    model: str | None = Field(default=None, max_length=128, description="Device model")
    android_version: str | None = Field(default=None, max_length=64, description="Android OS version")
    app_version: str | None = Field(default=None, max_length=64, description="App build version")
    event_type: str = Field(default="LOGIN", max_length=32, description="Authentication event type")


class UserDeviceSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    device_fingerprint: str
    manufacturer: str | None = None
    model: str | None = None
    android_version: str | None = None
    app_version: str | None = None
    first_seen_at: datetime | None = None
    last_seen_at: datetime | None = None
    is_active: bool = True


class UserSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    phone_number: str
    display_name: str | None = None
    firebase_uid: str
    status: str = "ACTIVE"
    created_at: datetime | None = None
    updated_at: datetime | None = None
    last_login_at: datetime | None = None
    devices: list[UserDeviceSchema] = []


class AuthSyncResponse(BaseModel):
    success: bool = True
    user: UserSchema
    device_id: uuid.UUID | None = None
    message: str | None = None
