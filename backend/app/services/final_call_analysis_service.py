from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.db.models import CallSession
from app.services.reality_defender_service import RealityDefenderService

logger = logging.getLogger("voice-clone-detection")


class FinalCallAnalysisService:
    """Runs the final authoritative mismatch check after call completion."""

    def __init__(self, db: Session) -> None:
        self.db = db
        self.settings = get_settings()
        self.reality_defender = RealityDefenderService()

    async def finalize_session(self, session: CallSession, final_audio_path: str | Path | None = None) -> dict[str, Any]:
        if final_audio_path is None:
            session.status = "PARTIAL"
            self.db.add(session)
            self.db.commit()
            return {"status": "PARTIAL", "final_risk": session.final_risk, "final_score": session.final_score}

        path = Path(final_audio_path)
        if not path.exists():
            session.status = "PARTIAL"
            self.db.add(session)
            self.db.commit()
            return {"status": "PARTIAL", "final_risk": session.final_risk, "final_score": session.final_score}

        try:
            result = await self.reality_defender.analyze_file(path)
            final_risk = "REAL" if result.get("classification") in {"LIKELY_GENUINE", "LIKELY_GENUINE"} else "SPOOF"
            final_score = float(result.get("ai_probability") or 0.0)
            session.final_risk = final_risk
            session.final_score = final_score
            session.final_provider = "reality-defender"
            session.status = "COMPLETED"
            session.updated_at = session.updated_at
            self.db.add(session)
            self.db.commit()
            return {
                "status": "COMPLETED",
                "final_risk": final_risk,
                "final_score": final_score,
                "final_provider": "reality-defender",
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("Final analysis failed for session %s: %s", session.id, exc)
            session.status = "PARTIAL"
            self.db.add(session)
            self.db.commit()
            return {"status": "PARTIAL", "final_risk": session.final_risk, "final_score": session.final_score}
