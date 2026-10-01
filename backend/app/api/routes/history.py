"""History listing endpoints."""

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.db.database import get_db
from app.db.models import AnalysisRecord
from app.schemas.analysis import AnalysisResponse, HistoryResponse


router = APIRouter(prefix="/history", tags=["history"])


@router.get("", response_model=HistoryResponse)
def get_history(
    page: int = Query(default=1, ge=1),
    limit: int = Query(default=10, ge=1, le=100),
    db: Session = Depends(get_db),
) -> HistoryResponse:
    offset = (page - 1) * limit

    query = db.query(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())
    total = query.count()
    records = query.offset(offset).limit(limit).all()

    items = [
        AnalysisResponse(
            analysis_id=item.id,
            status=item.status,
            classification=item.classification,
            risk_score=item.risk_score,
            confidence=item.confidence,
            ai_probability=item.ai_probability,
            duration_seconds=item.duration_seconds,
            segments_analyzed=item.segments_analyzed,
            processing_time_ms=item.processing_time_ms,
            detector_version=item.detector_version,
            reasons=item.reasons,
            created_at=item.created_at,
        )
        for item in records
    ]

    return HistoryResponse(page=page, limit=limit, total=total, items=items)
