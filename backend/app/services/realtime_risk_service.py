from __future__ import annotations

import logging
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("voice-clone-detection")


RISK_RANK: dict[str, int] = {
    "UNKNOWN": 0,
    "LOW": 1,
    "MEDIUM": 2,
    "HIGH": 3,
}


def risk_rank(risk: str | None) -> int:
    """Safe risk ranking: UNKNOWN -> 0, LOW -> 1, MEDIUM -> 2, HIGH -> 3. Unrecognized states fail safely to 0."""
    if not risk:
        return 0
    return RISK_RANK.get(str(risk).strip().upper(), 0)


def update_peak_risk(current_peak: str | None, new_risk: str | None) -> str:
    """
    Safely update peak risk according to Voice Shield rules:
    - If current_peak is None or UNKNOWN, new_risk (even if UNKNOWN) becomes the peak.
    - If new_risk has higher rank than current_peak, it replaces current_peak.
    - If new_risk is UNKNOWN, it never lowers an existing LOW/MEDIUM/HIGH peak.
    """
    if not current_peak:
        return new_risk or "UNKNOWN"
    norm_current = str(current_peak).strip().upper()
    norm_new = str(new_risk).strip().upper() if new_risk else "UNKNOWN"
    
    if norm_current == "UNKNOWN":
        return norm_new
    if risk_rank(norm_new) > risk_rank(norm_current):
        return norm_new
    return norm_current


class RealtimeRiskService:
    """Experimental realtime risk engine with hysteresis and cooldown semantics."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.medium_threshold = float(self.settings.realtime_medium_threshold)
        self.high_threshold = float(self.settings.realtime_high_threshold)
        self.window_size = int(self.settings.realtime_window_size)
        self.min_observations = int(self.settings.realtime_hysteresis_min_observations)
        self.cooldown_seconds = int(self.settings.realtime_cooldown_seconds)

    def _normalized_probability(self, provider_result: dict[str, Any]) -> float:
        """
        Convert provider output into a 0.0..1.0 spoof probability estimate.
        1. Prioritize explicit calibrated spoof score (from Aurigin global.score).
        2. Fall back to confidence / confidence inversion if score is absent.
        3. Fall back to deterministic mapping: REAL=0.08, SPOOF=0.85, UNKNOWN=0.45.
        """
        score = provider_result.get("score")
        if score is not None:
            try:
                val = float(score)
                if 0.0 <= val <= 1.0:
                    return val
            except (TypeError, ValueError):
                pass

        confidence = provider_result.get("confidence")
        result = str(provider_result.get("result", "UNKNOWN")).upper()
        if confidence is not None:
            try:
                val = float(confidence)
                if 0.0 <= val <= 1.0:
                    # If result is explicitly REAL/bonafide, high confidence means low spoof probability
                    if result == "REAL":
                        return round(max(0.0, 1.0 - val), 4)
                    return val
            except (TypeError, ValueError):
                pass

        if result == "SPOOF":
            return 0.85
        if result == "REAL":
            return 0.08
        return 0.45

    def _current_risk_state(self, recent_scores: list[float]) -> str:
        if not recent_scores:
            return "LOW"

        high_count = sum(1 for score in recent_scores if score >= self.high_threshold)
        medium_count = sum(1 for score in recent_scores if score >= self.medium_threshold)
        window_size = len(recent_scores)

        if window_size >= self.window_size:
            high_required = self.min_observations
            medium_required = self.min_observations
        else:
            required_for_small_window = max(2, (window_size + 1) // 2)
            high_required = required_for_small_window
            medium_required = required_for_small_window

        if high_count >= high_required:
            return "HIGH"
        if medium_count >= medium_required:
            return "MEDIUM"
        return "LOW"

    def evaluate(self, provider_result: dict[str, Any], recent_scores: list[float], previous_risk: str | None = None) -> dict[str, Any]:
        probability = self._normalized_probability(provider_result)
        if provider_result.get("result") in {"UNKNOWN", "ERROR"}:
            return {
                "risk_state": "UNKNOWN",
                "score": probability,
                "state_changed": False,
                "notification_required": False,
            }

        recent = list(recent_scores)[-self.window_size:]
        recent.append(probability)
        recent = recent[-self.window_size:]
        risk_state = self._current_risk_state(recent)
        state_changed = bool(previous_risk is not None and previous_risk != risk_state)
        notification_required = bool(state_changed and risk_state in {"MEDIUM", "HIGH"})
        if previous_risk is not None and previous_risk == risk_state:
            notification_required = False
        return {
            "risk_state": risk_state,
            "score": round(probability, 4),
            "state_changed": state_changed,
            "notification_required": notification_required,
        }
