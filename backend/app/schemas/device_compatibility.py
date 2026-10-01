"""Pydantic schemas for device recording compatibility requests and responses."""

from __future__ import annotations

from pydantic import BaseModel, Field


class DeviceRecordingConfig(BaseModel):
    search_scope: str = Field(..., description="Acquisition search scope (e.g., ROOT, SUB_FOLDER)")
    recording_folder: str | None = Field(default=None, description="Device-specific recording folder path (e.g., Music/PhoneRecord) or null")
    extensions: list[str] = Field(..., description="Supported recording audio file extensions")


class DeviceCompatibilityRequest(BaseModel):
    manufacturer: str = Field(..., min_length=1, description="Device manufacturer (e.g., Vivo, Infinix)")
    model: str = Field(..., min_length=1, description="Device model (e.g., T2 Pro, V70 FE, Note 40 Pro+)")
    os_family: str | None = Field(default=None, description="Operating system family (e.g., Funtouch OS, Origin OS, XOS)")
    os_version: str | None = Field(default=None, description="Operating system version string")
    android_version: str | None = Field(default=None, description="Android version string or API level")


class DeviceCompatibilityResponse(BaseModel):
    supported: bool = Field(..., description="Whether device is validated and supported")
    profile: str | None = Field(default=None, description="Validated recording profile (ROOT_LEVEL, SUB_FOLDER) or null")
    configuration: DeviceRecordingConfig | None = Field(default=None, description="Configuration parameters or null")
    profile_version: int | None = Field(default=None, description="Profile schema version or null")
