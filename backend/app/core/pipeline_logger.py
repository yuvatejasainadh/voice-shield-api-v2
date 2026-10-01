"""Voice Shield Pipeline Structured Logger.

Provides standardized, numbered, production-grade structured logging for:
Call -> Window -> Chunk -> API request -> Aurigin inference -> Classification -> Final decision.

Format:
  [TIMESTAMP] [LEVEL] [CALL_ID] [WINDOW] [POST_ID] EVENT details...

Never logs raw audio bytes, transcripts, auth tokens, API keys, or personal data.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger("voice-clone-detection")


def format_call_id(raw_call_id: str | None) -> str:
    """Standardize call_id into clean format e.g. CALL-7F3A21."""
    if not raw_call_id:
        return "CALL-UNKNOWN"
    cleaned = raw_call_id.strip()
    if cleaned.upper().startswith("CALL-"):
        return cleaned.upper()
    # Take last 6 characters of cleaned hex/alphanumeric representation
    alphanumeric = "".join(c for c in cleaned if c.isalnum()).upper()
    if len(alphanumeric) >= 6:
        return f"CALL-{alphanumeric[-6:]}"
    return f"CALL-{alphanumeric}" if alphanumeric else "CALL-UNKNOWN"


def format_window_id(seq: int | str | None) -> str:
    """Format window sequence to W01, W02, ..."""
    if seq is None:
        return "W01"
    try:
        if isinstance(seq, str) and seq.upper().startswith("W"):
            num = int(seq[1:])
            return f"W{num:02d}"
        num = int(seq)
        return f"W{num:02d}"
    except (ValueError, TypeError):
        return str(seq)


def format_post_id(seq: int | str | None) -> str:
    """Format POST sequence to POST #001, POST #002, ..."""
    if seq is None:
        return "POST #001"
    try:
        if isinstance(seq, str):
            if "#" in seq:
                num = int(seq.split("#")[-1].strip())
                return f"POST #{num:03d}"
            if seq.upper().startswith("REQ_") or seq.upper().startswith("POST_"):
                return seq
            num = int(seq)
            return f"POST #{num:03d}"
        num = int(seq)
        return f"POST #{num:03d}"
    except (ValueError, TypeError):
        return f"POST #{seq}"


def format_range_ms(start_ms: int, end_ms: int) -> str:
    """Format millisecond range into MM:SS-MM:SS format e.g. 00:00-00:10."""
    start_sec = max(0, start_ms // 1000)
    end_sec = max(0, end_ms // 1000)
    return f"{start_sec // 60:02d}:{start_sec % 60:02d}-{end_sec // 60:02d}:{end_sec % 60:02d}"


class VoiceShieldPipelineLogger:
    """
    Emits clean, structured, low-noise pipeline lifecycle logs.
    
    Categories:
      [CALL]     - Call session lifecycle (start, completion, history summary)
      [TCED]     - Temporal orchestration (config, window creation, tail handling)
      [DETECTOR] - Detector analysis start/completion/failure
      [DECISION] - Risk policy & detection decision updates
    """

    @staticmethod
    def _emit(
        level: int,
        category: str,
        window_id: str | None,
        event: str,
        message: str,
        json_fields: dict[str, Any] | None = None,
    ) -> None:
        wid_str = f" [{format_window_id(window_id)}]" if window_id else ""
        if event and message:
            log_line = f"[{category}]{wid_str} {event} {message}".strip()
        elif event:
            log_line = f"[{category}]{wid_str} {event}".strip()
        else:
            log_line = f"[{category}]{wid_str} {message}".strip()
        logger.log(level, log_line)

        # Optional structured JSON debug payload
        if json_fields is not None:
            now_iso = datetime.now(timezone.utc).isoformat()
            payload = {
                "timestamp": now_iso,
                "level": logging.getLevelName(level),
                "category": category,
                "event": event,
                **json_fields,
            }
            if window_id:
                payload["window_id"] = format_window_id(window_id)
            logger.debug("[JSON_LOG] %s", json.dumps(payload))

    # ── Call Session Lifecycle ───────────────────────────────────────────────

    @classmethod
    def call_started(cls, call_id: str) -> None:
        """Log call session initiation."""
        cid = format_call_id(call_id)
        cls._emit(
            logging.INFO,
            "CALL",
            None,
            "STARTED",
            f"call_id={cid}",
            {"call_id": cid, "status": "ACTIVE"},
        )

    @classmethod
    def call_completed(
        cls,
        duration_s: float,
        windows: int,
        bonafide: int,
        spoofed: int,
        partially_spoofed: int,
        final_risk: str,
        alert_triggered: bool,
        call_id: str | None = None,
    ) -> None:
        """Log high-level call completion summary."""
        summary = (
            f"duration={duration_s:.0f}s "
            f"windows={windows} "
            f"bonafide={bonafide} "
            f"spoofed={spoofed} "
            f"partially_spoofed={partially_spoofed} "
            f"final_risk={final_risk} "
            f"alert_triggered={str(alert_triggered).lower()}"
        )
        cls._emit(
            logging.INFO,
            "CALL",
            None,
            "COMPLETED",
            summary,
            {
                "call_id": format_call_id(call_id) if call_id else None,
                "duration_seconds": duration_s,
                "windows": windows,
                "bonafide": bonafide,
                "spoofed": spoofed,
                "partially_spoofed": partially_spoofed,
                "final_risk": final_risk,
                "alert_triggered": alert_triggered,
            },
        )

    @classmethod
    def call_history(
        cls,
        history_entries: list[tuple[str, str, float | None]],
        call_id: str | None = None,
    ) -> None:
        """Log temporal classification history summary once upon call completion."""
        entries_str = " ".join(
            f"{format_window_id(w)}={c}({conf:.2f})" if conf is not None else f"{format_window_id(w)}={c}"
            for w, c, conf in history_entries
        )
        cls._emit(
            logging.INFO,
            "CALL",
            None,
            "HISTORY",
            entries_str,
            {
                "call_id": format_call_id(call_id) if call_id else None,
                "history": [{"window": format_window_id(w), "result": c, "confidence": conf} for w, c, conf in history_entries],
            },
        )

    # ── TCED Orchestration Lifecycle ────────────────────────────────────────

    @classmethod
    def tced_config(
        cls,
        new_audio_s: float = 2.5,
        stride_s: float = 2.5,
        max_window_s: float = 5.0,
    ) -> None:
        """Log active TCED temporal window configuration once at call start."""
        fmt_new = f"{int(new_audio_s)}s" if float(new_audio_s).is_integer() else f"{new_audio_s:.1f}s"
        fmt_stride = f"{int(stride_s)}s" if float(stride_s).is_integer() else f"{stride_s:.1f}s"
        fmt_max = f"{int(max_window_s)}s" if float(max_window_s).is_integer() else f"{max_window_s:.1f}s"
        msg = f"new_audio={fmt_new} stride={fmt_stride} max_window={fmt_max}"
        cls._emit(
            logging.INFO,
            "TCED",
            None,
            "CONFIG",
            msg,
            {
                "new_audio_interval_seconds": new_audio_s,
                "window_stride_seconds": stride_s,
                "max_window_length_seconds": max_window_s,
            },
        )

    @classmethod
    def window_created(
        cls,
        window_id: str | int,
        start_ms: int,
        end_ms: int,
        sequence: int,
    ) -> None:
        """Log single window creation event."""
        r_str = format_range_ms(start_ms, end_ms)
        dur_s = (end_ms - start_ms) / 1000.0
        wid = format_window_id(window_id)
        msg = f"range={r_str} duration={dur_s:.0f}s sequence={sequence}"
        cls._emit(
            logging.INFO,
            "TCED",
            wid,
            "WINDOW_CREATED",
            msg,
            {"range": r_str, "duration_seconds": dur_s, "sequence": sequence, "start_ms": start_ms, "end_ms": end_ms},
        )

    @classmethod
    def tail_window_created(
        cls,
        window_id: str | int,
        start_ms: int,
        end_ms: int,
        sequence: int,
    ) -> None:
        """Log tail-aligned terminal window creation event."""
        r_str = format_range_ms(start_ms, end_ms)
        dur_s = (end_ms - start_ms) / 1000.0
        wid = format_window_id(window_id)
        msg = f"range={r_str} duration={dur_s:.0f}s sequence={sequence}"
        cls._emit(
            logging.INFO,
            "TCED",
            wid,
            "TAIL_WINDOW_CREATED",
            msg,
            {"range": r_str, "duration_seconds": dur_s, "sequence": sequence, "start_ms": start_ms, "end_ms": end_ms, "is_tail": True},
        )

    @classmethod
    def tced_call_ended(cls, duration_s: float) -> None:
        """Log call duration when TCED receives call termination signal."""
        cls._emit(
            logging.INFO,
            "TCED",
            None,
            "CALL_ENDED",
            f"duration={duration_s:.0f}s",
            {"duration_seconds": duration_s},
        )

    @classmethod
    def tced_tail_detected(cls, start_ms: int, end_ms: int) -> None:
        """Log detection of unprocessed tail audio segment upon call termination."""
        r_str = format_range_ms(start_ms, end_ms)
        cls._emit(
            logging.INFO,
            "TCED",
            None,
            "TAIL_DETECTED",
            f"unprocessed_range={r_str}",
            {"unprocessed_range": r_str, "start_ms": start_ms, "end_ms": end_ms},
        )

    @classmethod
    def tced_tail_suppressed(cls, reason: str, start_ms: int, end_ms: int) -> None:
        """Log suppression of redundant terminal tail window (e.g. duplicate range)."""
        r_str = format_range_ms(start_ms, end_ms)
        cls._emit(
            logging.WARNING,
            "TCED",
            None,
            "TAIL_SUPPRESSED",
            f"reason={reason} range={r_str}",
            {"reason": reason, "range": r_str, "start_ms": start_ms, "end_ms": end_ms},
        )

    # ── Detector Lifecycle ──────────────────────────────────────────────────

    @classmethod
    def detector_started(
        cls,
        window_id: str | int,
        detector: str = "AURIGIN",
        request_id: str | None = None,
    ) -> None:
        """Log dispatch of window audio to detector API."""
        wid = format_window_id(window_id)
        req_str = f"request_id={request_id or 'auto'}"
        cls._emit(
            logging.INFO,
            "DETECTOR",
            wid,
            "ANALYSIS_STARTED",
            f"detector={detector} {req_str}",
            {"detector": detector, "request_id": request_id},
        )

    @classmethod
    def detector_completed(
        cls,
        window_id: str | int,
        result: str,
        confidence: float | None,
        score: float | None,
        latency_ms: int,
        prediction_id: str | None,
        model: str | None = None,
    ) -> None:
        """Log completion of detector inference with key metrics."""
        wid = format_window_id(window_id)
        res_str = result.upper()
        conf_str = f"confidence={confidence:.2f}" if confidence is not None else "confidence=0.00"
        score_str = f"score={score:.2f}" if score is not None else "score=0.00"
        pred_str = f"prediction_id={prediction_id or 'none'}"
        model_str = f" model={model}" if model else ""
        msg = f"result={res_str} {conf_str} {score_str} latency_ms={latency_ms} {pred_str}{model_str}"
        cls._emit(
            logging.INFO,
            "DETECTOR",
            wid,
            "ANALYSIS_COMPLETED",
            msg,
            {
                "result": res_str,
                "confidence": confidence,
                "score": score,
                "latency_ms": latency_ms,
                "prediction_id": prediction_id,
                "model": model,
            },
        )

    @classmethod
    def detector_failed(
        cls,
        window_id: str | int,
        error_type: str,
        message: str,
        retry_count: int = 0,
    ) -> None:
        """Log detector request failure."""
        wid = format_window_id(window_id)
        retry_str = f" retry_count={retry_count}" if retry_count > 0 else ""
        cls._emit(
            logging.ERROR,
            "DETECTOR",
            wid,
            "ANALYSIS_FAILED",
            f"error_type={error_type} message={message}{retry_str}",
            {"error_type": error_type, "message": message, "retry_count": retry_count},
        )

    # ── Decision Engine Lifecycle ───────────────────────────────────────────

    @classmethod
    def decision(
        cls,
        window_id: str | int,
        risk: str,
        result: str,
        confidence: float | None,
        alert: bool = False,
    ) -> None:
        """Log decision engine evaluation for the window."""
        wid = format_window_id(window_id)
        res_str = result.upper()
        conf_str = f"confidence={confidence:.2f}" if confidence is not None else "confidence=0.00"
        alert_str = " alert=TRUE" if alert else ""
        msg = f"risk={risk} result={res_str} {conf_str}{alert_str}"
        cls._emit(
            logging.INFO,
            "DECISION",
            wid,
            "",
            msg,
            {"risk": risk, "result": res_str, "confidence": confidence, "alert": alert},
        )

    @classmethod
    def alert_triggered(
        cls,
        call_id: str | None = None,
        window_id: str | int | None = None,
        reason: str | None = None,
    ) -> None:
        """Log voice clone alert trigger warning."""
        wid = format_window_id(window_id) if window_id else None
        r_str = f"reason={reason}" if reason else ""
        cls._emit(
            logging.WARNING,
            "DECISION",
            wid,
            "ALERT_TRIGGERED",
            r_str,
            {"alert": True, "reason": reason, "call_id": format_call_id(call_id) if call_id else None},
        )

    @classmethod
    def validation_failed(cls, call_id: str | None, window_id: str | int | None, reason: str) -> None:
        """Log audio/request validation error."""
        wid = format_window_id(window_id) if window_id else None
        cls._emit(
            logging.ERROR,
            "CALL",
            wid,
            "AUDIO_VALIDATION_FAILED",
            f"reason={reason}",
            {"reason": reason, "call_id": format_call_id(call_id) if call_id else None},
        )

    # ── Backward Compatibility Aliases (if needed) ──────────────────────────

    @classmethod
    def request_started(cls, call_id: str, window_seq: int | str, request_id: str | None = None) -> None:
        cls.detector_started(window_id=window_seq, detector="AURIGIN", request_id=request_id)

    @classmethod
    def request_sent(cls, call_id: str, window_seq: int | str, attempt: int = 1) -> None:
        pass

    @classmethod
    def aurigin_response(cls, call_id: str, window_seq: int | str, latency_ms: int, status_code: int = 200) -> None:
        pass

    @classmethod
    def classification(
        cls,
        call_id: str,
        window_seq: int | str,
        classification: str,
        confidence: float | None,
        score: float | None = None,
        raw_result: str | None = None,
    ) -> None:
        pass

    @classmethod
    def policy_update(cls, call_id: str, window_seq: int | str, raw_result: str, policy_action: str) -> None:
        pass

    @classmethod
    def decision_update(cls, call_id: str, window_seq: int | str, risk: str, action: str | None = None) -> None:
        pass

    @classmethod
    def request_completed(cls, call_id: str, window_seq: int | str, total_latency_ms: int) -> None:
        pass

    @classmethod
    def classification_history(cls, call_id: str, history_entries: list[tuple[str, str, float | None]]) -> None:
        cls.call_history(history_entries, call_id=call_id)

    @classmethod
    def request_failed(
        cls,
        call_id: str,
        window_seq: int | str,
        error_type: str,
        error_message: str,
        retry_count: int = 0,
    ) -> None:
        cls.detector_failed(window_id=window_seq, error_type=error_type, message=error_message, retry_count=retry_count)

    @classmethod
    def window_detected(cls, call_id: str, window_seq: int, start_ms: int, end_ms: int) -> None:
        pass

    @classmethod
    def window_validated(cls, call_id: str, window_seq: int, duration_ms: int, is_final: bool = False) -> None:
        pass

