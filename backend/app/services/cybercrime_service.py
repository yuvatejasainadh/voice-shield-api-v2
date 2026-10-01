"""Service layer for cybercrime categories, templates, and version management."""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any
from sqlalchemy import func, select
from sqlalchemy.orm import Session, joinedload

from app.db.models import CybercrimeCategory, CybercrimeTemplate, CybercrimeTemplateVersion
from app.schemas.cybercrime import (
    CreateCategoryRequest,
    CreateTemplateRequest,
    CreateTemplateVersionRequest,
    CybercrimeCategorySchema,
    CybercrimeTemplateSchema,
    CybercrimeTemplateVersionSchema,
    UpdateTemplateRequest,
)

logger = logging.getLogger("voice-clone-detection")


class CybercrimeService:
    """Provides lookup, creation, and version management for cybercrime detection templates."""

    # ---------------- Categories ----------------

    @staticmethod
    def list_categories(db: Session, active_only: bool = True) -> list[CybercrimeCategorySchema]:
        stmt = select(CybercrimeCategory)
        if active_only:
            stmt = stmt.where(CybercrimeCategory.is_active.is_(True))
        stmt = stmt.order_by(CybercrimeCategory.name.asc())
        categories = db.execute(stmt).scalars().all()
        return [CybercrimeCategorySchema.model_validate(c) for c in categories]

    @staticmethod
    def get_category(db: Session, category_id: uuid.UUID) -> CybercrimeCategory | None:
        return db.execute(
            select(CybercrimeCategory).where(CybercrimeCategory.id == category_id)
        ).scalar_one_or_none()

    @staticmethod
    def create_category(db: Session, payload: CreateCategoryRequest) -> CybercrimeCategorySchema:
        now = datetime.now(timezone.utc)
        category = CybercrimeCategory(
            id=uuid.uuid4(),
            name=payload.name,
            description=payload.description,
            is_active=payload.is_active,
            created_at=now,
            updated_at=now,
        )
        db.add(category)
        db.commit()
        db.refresh(category)
        logger.info("[CYBERCRIME_CATEGORY_CREATED] id=%s name=%s", category.id, category.name)
        return CybercrimeCategorySchema.model_validate(category)

    # ---------------- Templates ----------------

    @staticmethod
    def _template_to_schema(template: CybercrimeTemplate) -> CybercrimeTemplateSchema:
        version_schemas = [
            CybercrimeTemplateVersionSchema.model_validate(v)
            for v in (template.versions or [])
        ]
        return CybercrimeTemplateSchema(
            id=template.id,
            category_id=template.category_id,
            category_name=template.category.name if template.category else None,
            name=template.name,
            description=template.description,
            severity=template.severity,
            indicators=template.indicators or [],
            recommended_actions=template.recommended_actions or [],
            metadata=template.template_metadata or {},
            is_active=template.is_active,
            created_at=template.created_at,
            updated_at=template.updated_at,
            versions=version_schemas,
        )

    @staticmethod
    def list_templates(
        db: Session,
        category_id: uuid.UUID | None = None,
        severity: str | None = None,
        active_only: bool = True,
    ) -> list[CybercrimeTemplateSchema]:
        stmt = (
            select(CybercrimeTemplate)
            .options(joinedload(CybercrimeTemplate.category), joinedload(CybercrimeTemplate.versions))
        )
        if active_only:
            stmt = stmt.where(CybercrimeTemplate.is_active.is_(True))
        if category_id:
            stmt = stmt.where(CybercrimeTemplate.category_id == category_id)
        if severity:
            stmt = stmt.where(CybercrimeTemplate.severity == severity.upper())

        stmt = stmt.order_by(CybercrimeTemplate.name.asc())
        templates = db.execute(stmt).unique().scalars().all()
        return [CybercrimeService._template_to_schema(t) for t in templates]

    @staticmethod
    def get_template(db: Session, template_id: uuid.UUID) -> CybercrimeTemplateSchema | None:
        stmt = (
            select(CybercrimeTemplate)
            .options(joinedload(CybercrimeTemplate.category), joinedload(CybercrimeTemplate.versions))
            .where(CybercrimeTemplate.id == template_id)
        )
        template = db.execute(stmt).unique().scalar_one_or_none()
        if template is None:
            return None
        return CybercrimeService._template_to_schema(template)

    @staticmethod
    def create_template(db: Session, payload: CreateTemplateRequest) -> CybercrimeTemplateSchema:
        now = datetime.now(timezone.utc)
        template = CybercrimeTemplate(
            id=uuid.uuid4(),
            category_id=payload.category_id,
            name=payload.name,
            description=payload.description,
            severity=payload.severity.upper() if payload.severity else None,
            indicators=payload.indicators,
            recommended_actions=payload.recommended_actions,
            template_metadata=payload.metadata,
            is_active=payload.is_active,
            created_at=now,
            updated_at=now,
        )
        db.add(template)
        db.flush()

        # Create baseline version 1
        initial_version_data = {
            "name": payload.name,
            "description": payload.description,
            "severity": payload.severity,
            "indicators": payload.indicators,
            "recommended_actions": payload.recommended_actions,
            "metadata": payload.metadata,
        }
        v1 = CybercrimeTemplateVersion(
            id=uuid.uuid4(),
            template_id=template.id,
            version=1,
            template_data=initial_version_data,
            change_notes="Initial version",
            created_at=now,
        )
        db.add(v1)
        db.commit()

        return CybercrimeService.get_template(db, template.id)  # type: ignore[return-value]

    @staticmethod
    def update_template(
        db: Session,
        template_id: uuid.UUID,
        payload: UpdateTemplateRequest,
    ) -> CybercrimeTemplateSchema | None:
        template = db.execute(
            select(CybercrimeTemplate).where(CybercrimeTemplate.id == template_id)
        ).scalar_one_or_none()
        if template is None:
            return None

        now = datetime.now(timezone.utc)
        if payload.name is not None:
            template.name = payload.name
        if payload.description is not None:
            template.description = payload.description
        if payload.severity is not None:
            template.severity = payload.severity.upper()
        if payload.indicators is not None:
            template.indicators = payload.indicators
        if payload.recommended_actions is not None:
            template.recommended_actions = payload.recommended_actions
        if payload.metadata is not None:
            template.template_metadata = payload.metadata
        if payload.is_active is not None:
            template.is_active = payload.is_active

        template.updated_at = now

        if payload.create_new_version:
            max_ver = db.execute(
                select(func.coalesce(func.max(CybercrimeTemplateVersion.version), 0))
                .where(CybercrimeTemplateVersion.template_id == template_id)
            ).scalar() or 0
            new_version_num = max_ver + 1
            new_version_data = {
                "name": template.name,
                "description": template.description,
                "severity": template.severity,
                "indicators": template.indicators,
                "recommended_actions": template.recommended_actions,
                "metadata": template.template_metadata,
            }
            new_ver = CybercrimeTemplateVersion(
                id=uuid.uuid4(),
                template_id=template.id,
                version=new_version_num,
                template_data=new_version_data,
                change_notes=payload.change_notes or f"Updated to version {new_version_num}",
                created_at=now,
            )
            db.add(new_ver)

        db.commit()
        return CybercrimeService.get_template(db, template.id)

    # ---------------- Versions ----------------

    @staticmethod
    def create_template_version(
        db: Session,
        template_id: uuid.UUID,
        payload: CreateTemplateVersionRequest,
    ) -> CybercrimeTemplateVersionSchema:
        now = datetime.now(timezone.utc)
        max_ver = db.execute(
            select(func.coalesce(func.max(CybercrimeTemplateVersion.version), 0))
            .where(CybercrimeTemplateVersion.template_id == template_id)
        ).scalar() or 0
        new_version_num = max_ver + 1

        version = CybercrimeTemplateVersion(
            id=uuid.uuid4(),
            template_id=template_id,
            version=new_version_num,
            template_data=payload.template_data,
            change_notes=payload.change_notes or f"Version {new_version_num}",
            created_at=now,
        )
        db.add(version)
        db.commit()
        db.refresh(version)
        return CybercrimeTemplateVersionSchema.model_validate(version)

    @staticmethod
    def list_template_versions(
        db: Session,
        template_id: uuid.UUID,
    ) -> list[CybercrimeTemplateVersionSchema]:
        stmt = (
            select(CybercrimeTemplateVersion)
            .where(CybercrimeTemplateVersion.template_id == template_id)
            .order_by(CybercrimeTemplateVersion.version.asc())
        )
        versions = db.execute(stmt).scalars().all()
        return [CybercrimeTemplateVersionSchema.model_validate(v) for v in versions]
