"""REST endpoints for querying and managing cybercrime categories and templates."""

from __future__ import annotations

import logging
import uuid
from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.schemas.cybercrime import (
    CreateCategoryRequest,
    CreateTemplateRequest,
    CreateTemplateVersionRequest,
    CybercrimeCategorySchema,
    CybercrimeTemplateSchema,
    CybercrimeTemplateVersionSchema,
    UpdateTemplateRequest,
)
from app.services.cybercrime_service import CybercrimeService

router = APIRouter(prefix="/cybercrime", tags=["cybercrime"])
logger = logging.getLogger("voice-clone-detection")


# Categories
@router.get("/categories", response_model=list[CybercrimeCategorySchema])
def list_categories(
    active_only: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> list[CybercrimeCategorySchema]:
    """Retrieve all cybercrime threat categories."""
    return CybercrimeService.list_categories(db=db, active_only=active_only)


@router.post("/categories", response_model=CybercrimeCategorySchema, status_code=status.HTTP_201_CREATED)
def create_category(
    payload: CreateCategoryRequest,
    db: Session = Depends(get_db),
) -> CybercrimeCategorySchema:
    """Create a new cybercrime threat category."""
    return CybercrimeService.create_category(db=db, payload=payload)


# Templates
@router.get("/templates", response_model=list[CybercrimeTemplateSchema])
def list_templates(
    category_id: uuid.UUID | None = Query(default=None),
    severity: str | None = Query(default=None),
    active_only: bool = Query(default=True),
    db: Session = Depends(get_db),
) -> list[CybercrimeTemplateSchema]:
    """Retrieve cybercrime fraud templates with optional category/severity filters."""
    return CybercrimeService.list_templates(
        db=db,
        category_id=category_id,
        severity=severity,
        active_only=active_only,
    )


@router.get("/templates/{template_id}", response_model=CybercrimeTemplateSchema)
def get_template(
    template_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> CybercrimeTemplateSchema:
    """Retrieve a specific cybercrime fraud template by ID."""
    template = CybercrimeService.get_template(db=db, template_id=template_id)
    if template is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TEMPLATE_NOT_FOUND", "message": "Cybercrime template not found"},
        )
    return template


@router.post("/templates", response_model=CybercrimeTemplateSchema, status_code=status.HTTP_201_CREATED)
def create_template(
    payload: CreateTemplateRequest,
    db: Session = Depends(get_db),
) -> CybercrimeTemplateSchema:
    """Create a new cybercrime fraud template with initial version."""
    category = CybercrimeService.get_category(db=db, category_id=payload.category_id)
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "CATEGORY_NOT_FOUND", "message": "Category not found"},
        )
    return CybercrimeService.create_template(db=db, payload=payload)


@router.put("/templates/{template_id}", response_model=CybercrimeTemplateSchema)
def update_template(
    template_id: uuid.UUID,
    payload: UpdateTemplateRequest,
    db: Session = Depends(get_db),
) -> CybercrimeTemplateSchema:
    """Update a cybercrime fraud template, optionally generating a new version."""
    template = CybercrimeService.update_template(db=db, template_id=template_id, payload=payload)
    if template is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TEMPLATE_NOT_FOUND", "message": "Cybercrime template not found"},
        )
    return template


# Versions
@router.get("/templates/{template_id}/versions", response_model=list[CybercrimeTemplateVersionSchema])
def list_template_versions(
    template_id: uuid.UUID,
    db: Session = Depends(get_db),
) -> list[CybercrimeTemplateVersionSchema]:
    """Retrieve all historical versions for a cybercrime template."""
    return CybercrimeService.list_template_versions(db=db, template_id=template_id)


@router.post(
    "/templates/{template_id}/versions",
    response_model=CybercrimeTemplateVersionSchema,
    status_code=status.HTTP_201_CREATED,
)
def create_template_version(
    template_id: uuid.UUID,
    payload: CreateTemplateVersionRequest,
    db: Session = Depends(get_db),
) -> CybercrimeTemplateVersionSchema:
    """Add a new version snapshot for a cybercrime template."""
    template = CybercrimeService.get_template(db=db, template_id=template_id)
    if template is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "TEMPLATE_NOT_FOUND", "message": "Cybercrime template not found"},
        )
    return CybercrimeService.create_template_version(db=db, template_id=template_id, payload=payload)
