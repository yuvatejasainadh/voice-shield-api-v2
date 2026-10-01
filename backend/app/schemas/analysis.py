"""Pydantic schemas for analysis API responses."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


Classification = Literal["LIKELY_GENUINE", "SUSPICIOUS", "LIKELY_AI_GENERATED"]


class AnalysisResponse(BaseModel):
    analysis_id: str
    status: str
    classification: Classification
    risk_score: int = Field(ge=0, le=100)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    ai_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    duration_seconds: float = Field(ge=0.0)
    segments_analyzed: int | None = Field(default=None, ge=1)
    processing_time_ms: int | None = Field(default=None, ge=0)
    detector_version: str
    reasons: list[str]
    created_at: datetime


class HistoryResponse(BaseModel):
    page: int
    limit: int
    total: int
    items: list[AnalysisResponse]


class APIError(BaseModel):
    detail: str
