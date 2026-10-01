"""Analysis submission and retrieval endpoints."""

from __future__ import annotations

import logging
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.database import get_db
from app.db.models import AnalysisRecord
from app.schemas.analysis import AnalysisResponse
from app.services.audio_service import AudioService
from app.services.aurigin_service import AuriginService


router = APIRouter(tags=["analysis"])
logger = logging.getLogger("voice-clone-detection")


def to_response(record: AnalysisRecord) -> AnalysisResponse:
    return AnalysisResponse(
        analysis_id=record.id,
        status=record.status,
        classification=record.classification,
        risk_score=record.risk_score,
        confidence=record.confidence,
        ai_probability=record.ai_probability,
        duration_seconds=record.duration_seconds,
        segments_analyzed=record.segments_analyzed,
        processing_time_ms=record.processing_time_ms,
        detector_version=record.detector_version,
        reasons=record.reasons,
        created_at=record.created_at,
    )


@router.post("/analysis", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
async def create_analysis(
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> AnalysisResponse:
    logger.info("Analysis request received for Aurigin detector")
    settings = get_settings()
    audio_service = AudioService()
    aurigin_service = AuriginService()

    stored_path, safe_filename, _, metadata = await audio_service.save_upload(audio)

    try:
        provider_res = await aurigin_service.analyze_audio(
            stored_path,
            filename=safe_filename,
            content_type=audio.content_type or f"audio/{metadata.extension}",
            duration_ms=int(metadata.duration_seconds * 1000),
        )

        raw_status = provider_res.get("raw_provider_status")
        if raw_status in ("NOT_CONFIGURED", "DISABLED"):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "AURIGIN_NOT_CONFIGURED", "message": "Aurigin API key is not configured"},
            )
        elif raw_status == "TIMEOUT":
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail={"code": "AURIGIN_TIMEOUT", "message": "Aurigin analysis request timed out"},
            )
        elif raw_status == "ERROR" or provider_res.get("result") == "UNKNOWN":
            err_reason = provider_res.get("reason") or "Provider analysis error"
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"code": "AURIGIN_ERROR", "message": f"Aurigin analysis failed: {err_reason}"},
            )

        score = provider_res.get("score")
        spoof_prob = float(score) if score is not None else 0.45
        risk_score = int(provider_res.get("risk_score", round(spoof_prob * 100)))

        # Map to standard Voice Shield classifications
        if risk_score <= settings.risk_genuine_max:
            classification = "LIKELY_GENUINE"
        elif risk_score <= settings.risk_suspicious_max:
            classification = "SUSPICIOUS"
        else:
            classification = "LIKELY_AI_GENERATED"

        reasons: list[str] = []
        if provider_res.get("reason"):
            reasons.append(str(provider_res["reason"]))
        if classification == "LIKELY_AI_GENERATED":
            reasons.append("Model detected acoustic characteristics associated with synthetic / cloned speech")
        elif classification == "SUSPICIOUS":
            reasons.append("Model found mixed acoustic indicators between genuine and synthetic speech")
        else:
            reasons.append("Model verified natural vocal acoustic characteristics")

        segments_count = len(provider_res.get("segments", [])) or max(1, int(round(metadata.duration_seconds / 5.0)))
        detector_ver = f"aurigin-{settings.aurigin_provider_version}"

        record = AnalysisRecord(
            id=str(uuid.uuid4()),
            filename=safe_filename,
            stored_audio_path=stored_path,
            status="completed",
            classification=classification,
            risk_score=risk_score,
            confidence=provider_res.get("confidence"),
            ai_probability=spoof_prob,
            duration_seconds=float(metadata.duration_seconds),
            segments_analyzed=segments_count,
            processing_time_ms=int(provider_res.get("processing_ms", 0)),
            detector_version=detector_ver,
            reasons=reasons,
        )

        db.add(record)
        db.commit()
        db.refresh(record)
    except HTTPException:
        audio_service.discard_upload(stored_path)
        raise
    except Exception:
        db.rollback()
        audio_service.discard_upload(stored_path)
        raise

    logger.info(
        "Analysis completed analysis_id=%s classification=%s risk_score=%s",
        record.id,
        record.classification,
        record.risk_score,
    )

    return to_response(record)


@router.get("/analysis/{analysis_id}", response_model=AnalysisResponse)
@router.get("/analyze/{analysis_id}", response_model=AnalysisResponse)
def get_analysis(analysis_id: str, db: Session = Depends(get_db)) -> AnalysisResponse:
    record = db.query(AnalysisRecord).filter(AnalysisRecord.id == analysis_id).first()
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return to_response(record)
