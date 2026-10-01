from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.audio.validator import AudioValidator
from app.core.config import get_settings
from app.db.database import get_db
from app.db.models import CallChunk, CallSession
from app.schemas.realtime import (
    ChunkResponse,
    CompleteSessionRequest,
    CompleteSessionResponse,
    CreateSessionRequest,
    CreateSessionResponse,
    FinalAnalysisResponse,
    SessionStatusResponse,
)
from app.services.aurigin_service import AuriginService
from app.services.final_call_analysis_service import FinalCallAnalysisService
from app.services.realtime_event_publisher import RealtimeEventPublisher
from app.services.realtime_risk_service import RealtimeRiskService, update_peak_risk

router = APIRouter(prefix="/realtime", tags=["realtime"])
logger = logging.getLogger("voice-clone-detection")


def _parse_datetime(value: datetime | str | None) -> datetime | None:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": "INVALID_TIMESTAMP", "message": "Invalid timestamp format"})


@router.post("/sessions", response_model=CreateSessionResponse)
def create_session(payload: CreateSessionRequest, db: Session = Depends(get_db)) -> CreateSessionResponse:
    existing = db.query(CallSession).filter(CallSession.client_session_id == payload.client_session_id).first()
    if existing is not None:
        return CreateSessionResponse(session_id=existing.id, status=existing.status, risk_state=existing.current_risk or "LOW")

    session = CallSession(
        id=str(uuid.uuid4()),
        client_session_id=payload.client_session_id,
        started_at=_parse_datetime(payload.started_at),
        current_risk="LOW",
        peak_risk="LOW",
        current_score=0.0,
        peak_score=0.0,
        realtime_alert_state="LOW",
        status="ACTIVE",
    )
    db.add(session)
    db.commit()
    db.refresh(session)
    return CreateSessionResponse(session_id=session.id, status=session.status, risk_state=session.current_risk)


@router.post("/sessions/{session_id}/chunks", response_model=ChunkResponse)
async def upload_chunk(
    session_id: str,
    audio: UploadFile = File(...),
    sequence_number: int = Form(...),
    started_at: str = Form(...),
    ended_at: str = Form(...),
    duration_ms: int = Form(...),
    idempotency_key: str = Form(...),
    content_type: str | None = Form(default=None),
    db: Session = Depends(get_db),
) -> ChunkResponse:
    request_id = f"REQ_{uuid.uuid4().hex[:8]}"
    window_id = f"W{sequence_number:03d}"
    recv_ts = datetime.now(timezone.utc).isoformat()
    logger.info(
        "[REALTIME] call_id=%s request_id=%s window_id=%s sequence=%s start=%s end=%s duration=%s",
        session_id,
        request_id,
        window_id,
        sequence_number,
        started_at,
        ended_at,
        duration_ms,
    )

    settings = get_settings()
    session = db.query(CallSession).filter(CallSession.id == session_id).first()
    if session is None:
        logger.warning("[REALTIME_REJECTED] call_id=%s request_id=%s reason=SESSION_NOT_FOUND", session_id, request_id)
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "SESSION_NOT_FOUND", "message": "Session not found"})
    if session.status not in {"ACTIVE", "FINALIZING"}:
        logger.warning("[REALTIME_REJECTED] call_id=%s request_id=%s reason=SESSION_NOT_ACTIVE", session_id, request_id)
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail={"code": "SESSION_NOT_ACTIVE", "message": "Session is not active"})
    if sequence_number <= 0:
        logger.warning("[REALTIME_REJECTED] call_id=%s request_id=%s reason=INVALID_SEQUENCE_NUMBER", session_id, request_id)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": "INVALID_SEQUENCE_NUMBER", "message": "Sequence number must be positive"})
    if duration_ms < int(settings.realtime_chunk_min_duration_seconds * 1000) or duration_ms > settings.realtime_chunk_max_duration_ms:
        logger.warning("[REALTIME_REJECTED] call_id=%s request_id=%s reason=INVALID_DURATION duration_ms=%s max_ms=%s", session_id, request_id, duration_ms, settings.realtime_chunk_max_duration_ms)
        raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail={"code": "INVALID_DURATION", "message": "Chunk duration is out of bounds"})
    if not idempotency_key:
        logger.warning("[REALTIME_REJECTED] call_id=%s request_id=%s reason=MISSING_IDEMPOTENCY_KEY", session_id, request_id)
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": "MISSING_IDEMPOTENCY_KEY", "message": "Idempotency key is required"})

    existing = db.query(CallChunk).filter(CallChunk.call_session_id == session_id, CallChunk.idempotency_key == idempotency_key).first()
    if existing is not None:
        logger.info("[REALTIME_CACHED] call_id=%s request_id=%s idempotency_key=%s matched_sequence=%s", session_id, request_id, idempotency_key, existing.sequence_number)
        return ChunkResponse(
            session_id=session_id,
            sequence_number=existing.sequence_number,
            risk_state=existing.risk or "LOW",
            score=existing.score,
            state_changed=False,
            notification_required=False,
            status=existing.status,
        )

    duplicate_sequence = db.query(CallChunk).filter(CallChunk.call_session_id == session_id, CallChunk.sequence_number == sequence_number).first()
    if duplicate_sequence is not None:
        logger.info("[REALTIME_CACHED] call_id=%s request_id=%s sequence_number=%s", session_id, request_id, sequence_number)
        return ChunkResponse(
            session_id=session_id,
            sequence_number=duplicate_sequence.sequence_number,
            risk_state=duplicate_sequence.risk or "LOW",
            score=duplicate_sequence.score,
            state_changed=False,
            notification_required=False,
            status=duplicate_sequence.status,
        )

    if not audio.filename:
        logger.warning("[REALTIME_REJECTED] call_id=%s request_id=%s reason=MISSING_FILENAME", session_id, request_id)
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"code": "MISSING_FILENAME", "message": "Missing audio filename"})

    validator = AudioValidator()
    extension = validator.validate_extension(audio.filename)
    temp_path = Path(settings.storage_path) / "temp" / "realtime" / f"{uuid.uuid4()}{extension}"
    temp_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        content = await audio.read()
        if not content:
            logger.warning("[REALTIME_REJECTED] call_id=%s request_id=%s reason=EMPTY_AUDIO", session_id, request_id)
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail={"code": "EMPTY_AUDIO", "message": "Audio file is empty"})
        temp_path.write_bytes(content)
        metadata = validator.validate_saved_file(temp_path, extension)
        if metadata.duration_seconds < settings.realtime_chunk_min_duration_seconds or metadata.duration_seconds > settings.realtime_chunk_max_duration_seconds:
            logger.warning("[REALTIME_REJECTED] call_id=%s request_id=%s reason=INVALID_SAVED_DURATION duration_s=%s max_s=%s", session_id, request_id, metadata.duration_seconds, settings.realtime_chunk_max_duration_seconds)
            raise HTTPException(status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, detail={"code": "INVALID_DURATION", "message": "Chunk duration is out of bounds"})

        aurigin = AuriginService()
        provider_result = await aurigin.analyze_audio(
            temp_path,
            audio.filename,
            content_type or audio.content_type,
            call_id=session_id,
            request_id=request_id,
            window_id=window_id,
            duration_ms=duration_ms,
        )
        if provider_result.get("result") == "UNKNOWN" and provider_result.get("raw_provider_status") in {"ERROR", "TIMEOUT", "NOT_CONFIGURED"}:
            risk_state = "UNKNOWN"
            score = None
            state_changed = False
            notification_required = False
            status_value = "PROVIDER_ERROR"
            risk_score = None
        else:
            previous_state = session.current_risk
            recent_scores = [
                chunk.score for chunk in db.query(CallChunk)
                .filter(CallChunk.call_session_id == session_id)
                .order_by(CallChunk.sequence_number.asc())
                .all()
                if chunk.score is not None
            ]
            if len(recent_scores) > settings.realtime_window_size:
                recent_scores = recent_scores[-settings.realtime_window_size:]
            risk_engine = RealtimeRiskService()
            decision = risk_engine.evaluate(provider_result, recent_scores, previous_risk=previous_state)
            risk_state = decision["risk_state"]
            score = decision["score"]
            state_changed = decision.get("state_changed", False)
            notification_required = decision.get("notification_required", False)
            status_value = "PROCESSED"
            risk_score = decision["score"]
            session.current_risk = risk_state
            session.current_score = float(score) if score is not None else 0.0
            session.peak_risk = update_peak_risk(session.peak_risk, risk_state)
            if score is not None:
                session.peak_score = max(float(session.peak_score or 0.0), float(score))
            session.realtime_alert_state = risk_state
            session.updated_at = datetime.now(timezone.utc)

        call_chunk = CallChunk(
            id=str(uuid.uuid4()),
            call_session_id=session_id,
            sequence_number=sequence_number,
            started_at=_parse_datetime(started_at),
            ended_at=_parse_datetime(ended_at),
            duration_ms=duration_ms,
            provider="aurigin",
            provider_prediction_id=provider_result.get("prediction_id"),
            score=score,
            risk=risk_state,
            processing_ms=provider_result.get("processing_ms"),
            status=status_value,
            idempotency_key=idempotency_key,
        )
        try:
            db.add(call_chunk)
            db.commit()
            db.refresh(call_chunk)
        except Exception:
            db.rollback()
            existing = db.query(CallChunk).filter(CallChunk.call_session_id == session_id, CallChunk.idempotency_key == idempotency_key).first()
            if existing is None:
                raise
            return ChunkResponse(
                session_id=session_id,
                sequence_number=existing.sequence_number,
                risk_state=existing.risk or "LOW",
                score=existing.score,
                state_changed=False,
                notification_required=False,
                status=existing.status,
            )

        if provider_result.get("result") == "UNKNOWN" and provider_result.get("raw_provider_status") in {"ERROR", "TIMEOUT", "NOT_CONFIGURED"}:
            session.current_risk = "UNKNOWN"
            session.current_score = 0.0
            session.updated_at = datetime.now(timezone.utc)
            db.add(session)
            db.commit()
        else:
            db.add(session)
            db.commit()

        RealtimeEventPublisher().publish_state_change(
            session_id=session_id,
            sequence_number=sequence_number,
            risk_state=risk_state,
            score=score,
            state_changed=state_changed,
            notification_required=notification_required,
        )

        logger.info(
            "[REALTIME_PROCESSED] call_id=%s request_id=%s window_id=%s sequence=%s risk_state=%s score=%s",
            session_id,
            request_id,
            window_id,
            sequence_number,
            risk_state,
            score,
        )

        return ChunkResponse(
            session_id=session_id,
            sequence_number=sequence_number,
            risk_state=risk_state,
            score=score,
            state_changed=state_changed,
            notification_required=notification_required,
            status=status_value,
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass


@router.get("/sessions/{session_id}", response_model=SessionStatusResponse)
def get_session_status(session_id: str, db: Session = Depends(get_db)) -> SessionStatusResponse:
    session = db.query(CallSession).filter(CallSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "SESSION_NOT_FOUND", "message": "Session not found"})
    chunks_processed = db.query(CallChunk).filter(CallChunk.call_session_id == session_id).count()
    last_chunk = db.query(CallChunk).filter(CallChunk.call_session_id == session_id).order_by(CallChunk.sequence_number.desc()).first()
    return SessionStatusResponse(
        session_id=session.id,
        status=session.status,
        current_risk=session.current_risk,
        current_score=session.current_score,
        peak_risk=session.peak_risk,
        peak_score=session.peak_score,
        chunks_processed=chunks_processed,
        last_sequence_number=last_chunk.sequence_number if last_chunk else None,
        last_updated_at=session.updated_at,
    )


@router.post("/sessions/{session_id}/complete", response_model=CompleteSessionResponse)
async def complete_session(session_id: str, payload: CompleteSessionRequest, db: Session = Depends(get_db)) -> CompleteSessionResponse:
    session = db.query(CallSession).filter(CallSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "SESSION_NOT_FOUND", "message": "Session not found"})
    session.ended_at = _parse_datetime(payload.ended_at) or datetime.now(timezone.utc)
    session.status = "CLOSED"
    db.add(session)
    db.commit()
    db.refresh(session)
    # Also notify RealtimeSessionManager to cancel any in-flight background tasks
    from app.services.realtime_session_manager import RealtimeSessionManager
    await RealtimeSessionManager.get_instance().close_session(session_id)
    return CompleteSessionResponse(session_id=session.id, status=session.status, final_analysis_id=None)



@router.post("/sessions/{session_id}/final-audio", response_model=FinalAnalysisResponse)
async def upload_final_audio(
    session_id: str,
    audio: UploadFile = File(...),
    db: Session = Depends(get_db),
) -> FinalAnalysisResponse:
    """Legacy post-call endpoint. Production voice clone detection is handled via real-time in-call pipeline."""
    session = db.query(CallSession).filter(CallSession.id == session_id).first()
    if session is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail={"code": "SESSION_NOT_FOUND", "message": "Session not found"})
    return FinalAnalysisResponse(
        session_id=session.id,
        status="COMPLETED",
        final_risk=session.peak_risk or session.current_risk or "LOW",
        final_score=session.peak_score or session.current_score or 0.0,
        final_provider="aurigin-realtime",
    )

