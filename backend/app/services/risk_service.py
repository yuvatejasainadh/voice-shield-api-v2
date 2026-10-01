"""Risk mapping service for classification outputs."""

from __future__ import annotations

from pydantic import BaseModel, Field

from app.core.config import get_settings


class RiskResult(BaseModel):
    risk_score: int = Field(ge=0, le=100)
    classification: str
    confidence: float = Field(ge=0.0, le=1.0)
    ai_probability: float = Field(ge=0.0, le=1.0)
    reasons: list[str]


class RiskService:
    """Converts detector output into risk metadata."""

    def __init__(self) -> None:
        settings = get_settings()
        self.genuine_max = settings.risk_genuine_max
        self.suspicious_max = settings.risk_suspicious_max

    def assess(self, detector_output: dict) -> RiskResult:
        """Map detector output to risk score and classification."""
        ai_probability = float(detector_output.get("ai_probability", 0.0))
        ai_probability = max(0.0, min(1.0, ai_probability))
        risk_score = int(round(ai_probability * 100))

        if risk_score <= self.genuine_max:
            classification = "LIKELY_GENUINE"
        elif risk_score <= self.suspicious_max:
            classification = "SUSPICIOUS"
        else:
            classification = "LIKELY_AI_GENERATED"

        return RiskResult(
            risk_score=risk_score,
            classification=classification,
            confidence=float(detector_output.get("confidence", 0.5)),
            ai_probability=ai_probability,
            reasons=list(detector_output.get("reasons", ["Detector output unavailable"])),
        )
