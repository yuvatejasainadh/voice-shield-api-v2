from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from app.db.database import SessionLocal
from app.db.models import CallSession
from app.schemas.realtime import CallReportSchema, WindowEvidenceSchema
from app.services.detection_decision_engine import DetectionDecisionEngine
from app.services.tced_window_manager import TCEDWindowManager

logger = logging.getLogger("voice-clone-detection")


@dataclass
class ActiveCallSession:
    call_session_id: str
    client_session_id: str
    status: str = "ACTIVE"                       # ACTIVE, ANALYZING, RISK_DETECTED, COMPLETED, FAILED, CANCELLED
    created_at_ms: float = field(default_factory=lambda: time.time() * 1000)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    closed_at_ms: float | None = None
    ended_at: datetime | None = None
    decision_engine: DetectionDecisionEngine = field(default_factory=DetectionDecisionEngine)
    tced_manager: TCEDWindowManager = field(default=None)
    processed_sequences: set[int] = field(default_factory=set)
    first_detection_ms: float | None = None
    final_decision_ms: float | None = None
    active_tasks: set[asyncio.Task] = field(default_factory=set)
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def __post_init__(self) -> None:
        if self.tced_manager is None:
            self.tced_manager = TCEDWindowManager(call_session_id=self.call_session_id)

    def get_report_schema(self) -> CallReportSchema:
        """Construct the live CallReportSchema snapshot."""
        out = self.decision_engine._evaluate_state()
        risk_level = self.decision_engine.get_risk_level(out.call_score, out.decision_state)
        
        # Report status mapping
        if self.status == "CLOSED":
            report_status = "COMPLETED"
        elif out.notification_required or risk_level in ("HIGH", "MEDIUM") or out.decision_state == "confirmed" and out.call_score >= 0.5:
            report_status = "RISK_DETECTED"
        elif len(self.decision_engine.observations) > 0:
            report_status = "ANALYZING"
        else:
            report_status = "ACTIVE"

        evidence_list = [
            WindowEvidenceSchema(**e) for e in self.decision_engine.get_evidence_list()
        ]

        return CallReportSchema(
            callSessionId=self.call_session_id,
            status=report_status,
            startedAt=self.started_at,
            endedAt=self.ended_at,
            finalClassification=out.user_label,
            riskLevel=risk_level,
            confidence=out.call_confidence if out.call_confidence > 0 else None,
            score=out.call_score,
            chunksAnalyzed=out.windows_evaluated,
            totalDurationSeconds=self.decision_engine.get_total_duration_seconds(),
            evidence=evidence_list,
            detectorVersion="aurigin-realtime",
            lastUpdatedAt=datetime.now(timezone.utc),
        )


class RealtimeSessionManager:
    """Singleton-style manager for active call sessions and call reports."""

    _instance: RealtimeSessionManager | None = None

    def __init__(self) -> None:
        self._sessions: dict[str, ActiveCallSession] = {}
        self._lock = asyncio.Lock()

    @classmethod
    def get_instance(cls) -> RealtimeSessionManager:
        if cls._instance is None:
            cls._instance = RealtimeSessionManager()
        return cls._instance

    async def create_session(
        self,
        call_session_id: str,
        client_session_id: str | None = None,
        started_at: datetime | None = None,
    ) -> ActiveCallSession:
        effective_start = started_at or datetime.now(timezone.utc)
        async with self._lock:
            existing = self._sessions.get(call_session_id)
            if existing is not None:
                if existing.status == "CLOSED":
                    # If closed, reopen cleanly
                    existing.status = "ACTIVE"
                    existing.closed_at_ms = None
                    existing.ended_at = None
                    existing.started_at = effective_start
                    existing.processed_sequences.clear()
                    existing.decision_engine = DetectionDecisionEngine()
                    existing.tced_manager = TCEDWindowManager(call_session_id=existing.call_session_id)
                session = existing
            else:
                session = ActiveCallSession(
                    call_session_id=call_session_id,
                    client_session_id=client_session_id or call_session_id,
                    status="ACTIVE",
                    started_at=effective_start,
                )
                self._sessions[call_session_id] = session

        # Sync/Create session in database
        self._sync_db_session(session)
        logger.info("Realtime session created call_session_id=%s", call_session_id)
        return session

    def get_session(self, call_session_id: str) -> ActiveCallSession | None:
        return self._sessions.get(call_session_id)

    def is_session_active(self, call_session_id: str) -> bool:
        session = self._sessions.get(call_session_id)
        return session is not None and session.status != "CLOSED"

    async def handle_client_disconnect(self, call_session_id: str) -> ActiveCallSession | None:
        session = self._sessions.get(call_session_id)
        if session is None:
            return None

        # Wait gracefully for any active in-flight detection tasks without cancelling them
        tasks = list(session.active_tasks)
        if tasks:
            logger.info("Awaiting %d in-flight tasks for disconnected session %s", len(tasks), call_session_id)
            try:
                await asyncio.wait(tasks, timeout=10.0)
            except Exception as exc:
                logger.warning("Error waiting for in-flight tasks on session %s: %s", call_session_id, exc)

        async with session.lock:
            # If session is still ACTIVE, mark as INTERRUPTED rather than forcing normal CLOSED completion
            if session.status == "ACTIVE":
                session.status = "INTERRUPTED"
            session.closed_at_ms = time.time() * 1000

        # Persist final state and processed evidence to DB
        try:
            self._sync_db_session(session)
        except Exception as exc:
            logger.warning("DB sync error during disconnect: %s", exc)
        logger.info("[WS_DISCONNECT] Session disconnected and state preserved call_session_id=%s status=%s", call_session_id, session.status)
        return session

    async def close_session(self, call_session_id: str) -> ActiveCallSession | None:
        session = self._sessions.get(call_session_id)
        if session is None:
            return None

        # Await pending in-flight tasks
        tasks = list(session.active_tasks)
        if tasks:
            try:
                await asyncio.wait(tasks, timeout=5.0)
            except Exception:
                pass

        async with session.lock:
            session.status = "CLOSED"
            session.closed_at_ms = time.time() * 1000
            session.ended_at = datetime.now(timezone.utc)
            session.active_tasks.clear()

        # Finalize and persist report in DB
        try:
            self._sync_db_session(session)
        except Exception as exc:
            logger.warning("DB sync error during close_session: %s", exc)
        logger.info("Realtime session closed and report persisted call_session_id=%s", call_session_id)
        return session

    def sync_report_to_db(self, call_session_id: str) -> None:
        session = self._sessions.get(call_session_id)
        if session:
            try:
                self._sync_db_session(session)
            except Exception as exc:
                logger.warning("DB sync error during sync_report_to_db: %s", exc)

    def _sync_db_session(self, session: ActiveCallSession) -> None:
        """Persist/update the call session and call report state in database."""
        try:
            report = session.get_report_schema()
            with SessionLocal() as db:
                db_session = db.query(CallSession).filter(CallSession.id == session.call_session_id).first()
                if not db_session:
                    db_session = CallSession(
                        id=session.call_session_id,
                        client_session_id=session.client_session_id,
                        started_at=session.started_at,
                        status=report.status,
                    )
                    db.add(db_session)

                db_session.status = report.status
                db_session.ended_at = session.ended_at
                db_session.current_risk = report.riskLevel
                db_session.current_score = report.score or 0.0
                db_session.final_risk = report.riskLevel
                db_session.final_score = report.score
                db_session.final_classification = report.finalClassification
                db_session.confidence = report.confidence
                db_session.chunks_analyzed = report.chunksAnalyzed
                db_session.total_duration_seconds = report.totalDurationSeconds
                db_session.evidence = [e.model_dump() for e in report.evidence]
                db_session.detector_version = report.detectorVersion
                db_session.updated_at = datetime.now(timezone.utc)

                db.commit()
        except Exception as exc:
            logger.exception("Failed to sync call report to DB for session %s: %s", session.call_session_id, exc)

    def register_task(self, call_session_id: str, task: asyncio.Task) -> None:
        session = self._sessions.get(call_session_id)
        if session is not None and session.status != "CLOSED":
            session.active_tasks.add(task)
            task.add_done_callback(lambda t: session.active_tasks.discard(t) if session else None)

    def is_sequence_processed(self, call_session_id: str, sequence_number: int) -> bool:
        session = self._sessions.get(call_session_id)
        if session is None:
            return False
        return sequence_number in session.processed_sequences

    def record_sequence(self, call_session_id: str, sequence_number: int) -> None:
        session = self._sessions.get(call_session_id)
        if session is not None:
            session.processed_sequences.add(sequence_number)

    def clear_all(self) -> None:
        """Helper for test cleanup."""
        self._sessions.clear()

