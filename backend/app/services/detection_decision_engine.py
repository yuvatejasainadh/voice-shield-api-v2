"""Detection Decision Engine for multi-window cumulative voice-clone detection.

Maintains call-level evidence across windows (e.g. 0-10s, 0-20s, 0-30s+),
distinguishes between individual chunk detector predictions and the call-level
decision state, supports early detection upon high-confidence spoof evidence,
and handles mixed bonafide/spoofed calls without triggering false alarms.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("voice-clone-detection")


@dataclass
class WindowObservation:
    sequence_number: int
    window_start_ms: int
    window_end_ms: int
    raw_result: str              # "bonafide", "spoofed", "partially_spoofed", "unknown"
    normalized_result: str       # "REAL", "SPOOF", "UNKNOWN"
    score: float                 # Calibrated spoof probability 0.0 - 1.0
    confidence: float            # 0.0 - 1.0
    processing_ms: int = 0
    reason: str | None = None
    timestamp_ms: float = 0.0
    window_id: str | None = None


@dataclass
class DecisionStateOutput:
    decision_state: str          # "observing" | "suspicious" | "confirmed" | "uncertain"
    user_label: str              # "ANALYZING" | "SUSPICIOUS" | "VOICE CLONING DETECTED" | "SAFE / LOW RISK" | "INSUFFICIENT EVIDENCE"
    call_score: float            # Aggregate call-level spoof score (0.0 - 1.0)
    call_confidence: float       # Aggregate call-level confidence (0.0 - 1.0)
    state_changed: bool          # True if decisionState changed from previous step
    notification_required: bool  # True if user-facing alert is warranted
    is_early_detection: bool     # True if confirmed before primary decision window
    reason: str                  # Explainability string
    windows_evaluated: int       # Total windows processed


class DetectionDecisionEngine:
    """Aggregates detector window results and computes the call-level decision state."""

    def __init__(
        self,
        spoof_threshold: float | None = None,
        early_spoof_threshold: float | None = None,
        genuine_threshold: float | None = None,
        min_confidence: float | None = None,
        min_windows_for_confirmation: int | None = None,
        allow_early_detection: bool | None = None,
        primary_decision_window_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        self.spoof_threshold = spoof_threshold if spoof_threshold is not None else settings.decision_spoof_threshold
        self.early_spoof_threshold = early_spoof_threshold if early_spoof_threshold is not None else settings.decision_early_spoof_threshold
        self.genuine_threshold = genuine_threshold if genuine_threshold is not None else settings.decision_genuine_threshold
        self.min_confidence = min_confidence if min_confidence is not None else settings.decision_min_confidence
        self.min_windows_for_confirmation = (
            min_windows_for_confirmation if min_windows_for_confirmation is not None else settings.decision_min_windows_for_confirmation
        )
        self.allow_early_detection = (
            allow_early_detection if allow_early_detection is not None else settings.decision_allow_early_detection
        )
        self.primary_decision_window_seconds = (
            primary_decision_window_seconds
            if primary_decision_window_seconds is not None
            else settings.realtime_primary_decision_window_seconds
        )

        self.observations: list[WindowObservation] = []
        self.current_decision_state: str = "observing"
        self.has_notified: bool = False

    def add_observation(
        self,
        sequence_number: int,
        window_start_ms: int,
        window_end_ms: int,
        raw_result: str | None,
        normalized_result: str,
        score: float | None,
        confidence: float | None,
        processing_ms: int = 0,
        reason: str | None = None,
        timestamp_ms: float = 0.0,
        window_id: str | None = None,
    ) -> DecisionStateOutput:
        """Record a new window observation and evaluate updated call decision state."""
        effective_score = float(score) if score is not None else 0.45
        effective_confidence = float(confidence) if confidence is not None else 0.5
        effective_raw = str(raw_result or "unknown").lower()

        obs = WindowObservation(
            sequence_number=sequence_number,
            window_start_ms=window_start_ms,
            window_end_ms=window_end_ms,
            raw_result=effective_raw,
            normalized_result=normalized_result.upper(),
            score=effective_score,
            confidence=effective_confidence,
            processing_ms=processing_ms,
            reason=reason,
            timestamp_ms=timestamp_ms,
            window_id=window_id,
        )
        self.observations.append(obs)
        return self._evaluate_state()

    def _evaluate_state(self) -> DecisionStateOutput:
        prev_state = self.current_decision_state
        total_obs = len(self.observations)

        if total_obs == 0:
            return DecisionStateOutput(
                decision_state="observing",
                user_label="ANALYZING",
                call_score=0.0,
                call_confidence=0.0,
                state_changed=False,
                notification_required=False,
                is_early_detection=False,
                reason="No audio windows evaluated yet",
                windows_evaluated=0,
            )

        # Count unknown / provider error windows
        valid_obs = [o for o in self.observations if o.normalized_result != "UNKNOWN"]
        if not valid_obs:
            self.current_decision_state = "uncertain"
            return DecisionStateOutput(
                decision_state="uncertain",
                user_label="INSUFFICIENT EVIDENCE",
                call_score=0.45,
                call_confidence=0.0,
                state_changed=(prev_state != "uncertain"),
                notification_required=False,
                is_early_detection=False,
                reason="All received audio windows were inconclusive or returned provider errors",
                windows_evaluated=total_obs,
            )

        # Weight cumulative windows: later cumulative windows (0-20s, 0-30s) contain more context
        weights = [max(1.0, o.window_end_ms / 10000.0) for o in valid_obs]
        total_weight = sum(weights)
        weighted_score = sum(o.score * w for o, w in zip(valid_obs, weights)) / total_weight
        weighted_confidence = sum(o.confidence * w for o, w in zip(valid_obs, weights)) / total_weight

        # Check latest observation
        latest = valid_obs[-1]
        latest_duration_s = latest.window_end_ms / 1000.0

        # Check for high-spoof windows
        high_spoof_windows = [
            o for o in valid_obs
            if o.score >= self.spoof_threshold and o.confidence >= self.min_confidence
        ]
        num_high_spoof = len(high_spoof_windows)

        # Evaluate decision state
        is_early = False
        decision_state = "observing"
        reason = "Accumulating audio evidence"
        is_partially_spoofed = False

        # Check for mixed results (partially spoofed)
        has_spoof_or_partial = any(
            o.normalized_result in ("SPOOF", "SPOOFED")
            or o.raw_result in ("spoofed", "partially_spoofed")
            or (o.score >= self.spoof_threshold and o.confidence >= self.min_confidence)
            for o in valid_obs
        )
        has_bonafide = any(
            o.normalized_result == "REAL"
            or o.raw_result in ("bonafide", "authentic", "real")
            or (o.score <= self.genuine_threshold and o.confidence >= self.min_confidence)
            for o in valid_obs
        )
        if has_spoof_or_partial and has_bonafide and len(valid_obs) >= 2:
            is_partially_spoofed = True


        # 1. Early spoof detection check (e.g. 0-10s with very high spoof score & confidence)
        if (
            self.allow_early_detection
            and latest.score >= self.early_spoof_threshold
            and latest.confidence >= self.min_confidence
        ):
            decision_state = "confirmed"
            is_early = True
            reason = f"High-confidence synthetic voice detected early at {latest_duration_s:.1f}s (score={latest.score:.2f}, conf={latest.confidence:.2f})"

        # 2. Confirmed spoof via cumulative repeated evidence
        elif num_high_spoof >= self.min_windows_for_confirmation:
            decision_state = "confirmed"
            reason = f"Voice cloning confirmed across {num_high_spoof} cumulative windows (score={weighted_score:.2f})"

        # 3. Suspicious mixed / elevated spoof activity
        elif is_partially_spoofed or num_high_spoof >= 1 or weighted_score >= (self.spoof_threshold - 0.15):
            decision_state = "suspicious"
            if is_partially_spoofed:
                reason = f"Mixed authentic and synthetic voice patterns observed (score={weighted_score:.2f}, conf={weighted_confidence:.2f})"
            else:
                reason = f"Suspicious acoustic patterns observed in call audio (score={weighted_score:.2f}, conf={weighted_confidence:.2f})"

        # 4. Primary decision window reached (>= 30s)
        elif latest_duration_s >= self.primary_decision_window_seconds:
            if weighted_score <= self.genuine_threshold:
                decision_state = "confirmed"  # Confirmed genuine/safe
                reason = f"Speech verified authentic after {latest_duration_s:.1f}s (score={weighted_score:.2f})"
            elif weighted_confidence < self.min_confidence:
                decision_state = "uncertain"
                reason = "Acoustic evidence inconclusive after 30s due to low model confidence"
            else:
                decision_state = "observing"
                reason = f"Call within normal limits, continuing to monitor (score={weighted_score:.2f})"

        # 5. Still in early monitoring window (< 30s) with low spoof score
        else:
            decision_state = "observing"
            start_s = latest.window_start_ms / 1000.0
            end_s = latest.window_end_ms / 1000.0
            reason = f"Analyzing window {latest.sequence_number} ({start_s:.1f}s–{end_s:.1f}s)..."

        # Determine state change and notification requirement
        state_changed = (decision_state != prev_state)
        self.current_decision_state = decision_state

        # User-facing label mapping
        if decision_state == "confirmed":
            if num_high_spoof >= self.min_windows_for_confirmation or weighted_score >= self.spoof_threshold or is_early:
                user_label = "VOICE CLONING DETECTED"
                notification_required = not self.has_notified
                if notification_required:
                    self.has_notified = True
            else:
                user_label = "SAFE / LOW RISK"
                notification_required = False
        elif decision_state == "suspicious":
            if is_partially_spoofed:
                user_label = "SPOOFED"
            else:
                user_label = "SUSPICIOUS"
            notification_required = state_changed and not self.has_notified
        elif decision_state == "uncertain":
            user_label = "INSUFFICIENT EVIDENCE"
            notification_required = False
        else:
            user_label = "ANALYZING"
            notification_required = False

        return DecisionStateOutput(
            decision_state=decision_state,
            user_label=user_label,
            call_score=round(weighted_score, 4),
            call_confidence=round(weighted_confidence, 4),
            state_changed=state_changed,
            notification_required=notification_required,
            is_early_detection=is_early,
            reason=reason,
            windows_evaluated=total_obs,
        )

    def get_risk_level(self, score: float, decision_state: str) -> str:
        """Map score and decision state to standardized risk level."""
        if decision_state == "uncertain":
            return "UNKNOWN"
        if score >= self.spoof_threshold or decision_state == "confirmed" and score >= 0.5:
            return "HIGH"
        if score >= (self.spoof_threshold - 0.25) or decision_state == "suspicious":
            return "MEDIUM"
        return "LOW"

    def get_evidence_list(self) -> list[dict[str, Any]]:
        """Return raw window observations formatted for persistence and report schemas."""
        evidence: list[dict[str, Any]] = []
        for obs in self.observations:
            win_id = obs.window_id or f"W{obs.sequence_number:02d}"
            evidence.append({
                "windowId": win_id,
                "sequenceNumber": obs.sequence_number,
                "windowStartMs": obs.window_start_ms,
                "windowEndMs": obs.window_end_ms,
                "rawClassification": obs.raw_result,
                "normalizedClassification": obs.normalized_result,
                "score": obs.score,
                "confidence": obs.confidence,
                "processingMs": obs.processing_ms,
                "reason": obs.reason,
                "timestampMs": obs.timestamp_ms,
            })
        return evidence

    def get_total_duration_seconds(self) -> float:
        """Compute max window duration seen so far."""
        if not self.observations:
            return 0.0
        return max(o.window_end_ms for o in self.observations) / 1000.0

