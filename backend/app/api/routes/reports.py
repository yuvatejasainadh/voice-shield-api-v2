"""REST endpoints for querying active and completed real-time Call Reports."""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import CallSession
from app.schemas.realtime import CallReportSchema, WindowEvidenceSchema
from app.services.realtime_session_manager import RealtimeSessionManager

router = APIRouter(prefix="/reports", tags=["reports"])
logger = logging.getLogger("voice-clone-detection")


class PaginatedReportsResponse(BaseModel):
    page: int
    limit: int
    total: int
    items: list[CallReportSchema]


def _call_session_to_report_schema(session: CallSession) -> CallReportSchema:
    # Check if active session is in memory for fresh live state
    mgr_sess = RealtimeSessionManager.get_instance().get_session(session.id)
    if mgr_sess is not None and mgr_sess.status != "CLOSED":
        return mgr_sess.get_report_schema()

    raw_evidence = session.evidence or []
    evidence_models = []
    for item in raw_evidence:
        try:
            evidence_models.append(WindowEvidenceSchema(**item))
        except Exception:
            pass

    return CallReportSchema(
        callSessionId=session.id,
        status=session.status,
        startedAt=session.started_at or session.created_at,
        endedAt=session.ended_at,
        finalClassification=session.final_classification or "ANALYZING",
        riskLevel=session.final_risk or session.current_risk or "LOW",
        confidence=session.confidence,
        score=session.final_score if session.final_score is not None else session.current_score,
        chunksAnalyzed=session.chunks_analyzed,
        totalDurationSeconds=session.total_duration_seconds,
        evidence=evidence_models,
        detectorVersion=session.detector_version or "aurigin-realtime",
        lastUpdatedAt=session.updated_at or session.created_at,
    )


@router.get("", response_model=PaginatedReportsResponse)
def get_reports(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=10, ge=1, le=100),
    status: str | None = Query(default=None),
    db: Session = Depends(get_db),
) -> PaginatedReportsResponse:
    """Retrieve paginated call reports for call history display."""
    query = db.query(CallSession)
    if status:
        query = query.filter(CallSession.status == status.upper())

    query = query.order_by(CallSession.created_at.desc())
    total = query.count()
    records = query.offset((page - 1) * limit).limit(limit).all()

    items = [_call_session_to_report_schema(rec) for rec in records]
    return PaginatedReportsResponse(page=page, limit=limit, total=total, items=items)


@router.get("/{call_session_id}", response_model=CallReportSchema)
def get_report_by_session_id(
    call_session_id: str,
    db: Session = Depends(get_db),
) -> CallReportSchema:
    """Retrieve detailed call report and window evidence for a specific callSessionId."""
    # Check active memory session first
    mgr_sess = RealtimeSessionManager.get_instance().get_session(call_session_id)
    if mgr_sess is not None and mgr_sess.status != "CLOSED":
        return mgr_sess.get_report_schema()

    session = (
        db.query(CallSession)
        .filter((CallSession.id == call_session_id) | (CallSession.client_session_id == call_session_id))
        .first()
    )
    if not session:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"code": "REPORT_NOT_FOUND", "message": f"Call report not found for {call_session_id}"},
        )

    return _call_session_to_report_schema(session)
