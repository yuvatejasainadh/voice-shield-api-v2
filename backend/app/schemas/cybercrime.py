"""Pydantic schemas for cybercrime categories, templates, and template versions."""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any
from pydantic import BaseModel, ConfigDict, Field


class CybercrimeCategorySchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    name: str
    description: str | None = None
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None


class CreateCategoryRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=128)
    description: str | None = None
    is_active: bool = True


class CybercrimeTemplateVersionSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    template_id: uuid.UUID
    version: int
    template_data: dict[str, Any]
    change_notes: str | None = None
    created_at: datetime | None = None


class CreateTemplateVersionRequest(BaseModel):
    template_data: dict[str, Any] = Field(...)
    change_notes: str | None = None


class CybercrimeTemplateSchema(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: uuid.UUID
    category_id: uuid.UUID
    category_name: str | None = None
    name: str
    description: str | None = None
    severity: str | None = None
    indicators: list[Any] = []
    recommended_actions: list[Any] = []
    metadata: dict[str, Any] = {}
    is_active: bool = True
    created_at: datetime | None = None
    updated_at: datetime | None = None
    versions: list[CybercrimeTemplateVersionSchema] = []


class CreateTemplateRequest(BaseModel):
    category_id: uuid.UUID
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    severity: str | None = Field(default=None, max_length=32)
    indicators: list[Any] = Field(default_factory=list)
    recommended_actions: list[Any] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    is_active: bool = True


class UpdateTemplateRequest(BaseModel):
    name: str | None = Field(default=None, max_length=255)
    description: str | None = None
    severity: str | None = Field(default=None, max_length=32)
    indicators: list[Any] | None = None
    recommended_actions: list[Any] | None = None
    metadata: dict[str, Any] | None = None
    is_active: bool | None = None
    create_new_version: bool = False
    change_notes: str | None = None
