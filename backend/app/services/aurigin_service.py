from __future__ import annotations

import asyncio
import logging
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

from app.core.config import get_settings
from app.core.pipeline_logger import VoiceShieldPipelineLogger
from app.services.voice_deepfake_provider import VoiceDeepfakeProvider

logger = logging.getLogger("voice-clone-detection")


@dataclass
class AuriginResult:
    provider: str = "aurigin"
    prediction_id: str | None = None
    result: str = "UNKNOWN"               # Normalized: REAL, SPOOF, UNKNOWN
    score: float | None = None            # Spoof probability: 0.0 - 1.0
    confidence: float | None = None       # Calibrated provider confidence: 0.0 - 1.0
    risk_score: int | None = None         # 0 - 100 integer score
    segments: list[dict[str, Any]] = field(default_factory=list)  # Preserved 5s segment predictions
    processing_ms: int | None = None
    raw_provider_status: str | None = None
    raw_result: str | None = None         # bonafide, spoofed, partially_spoofed, None
    reason: str | None = None


class AuriginResultNormalizer:
    """Normalizes raw Aurigin /v1/predict response payloads into Voice Shield domain models."""

    @staticmethod
    def normalize_result_value(value: Any) -> str:
        """
        Maps official Aurigin result strings to normalized Voice Shield states:
        - bonafide, authentic, real -> REAL
        - spoofed, partially_spoofed, spoof, fake, manipulated -> SPOOFED
        - null, none, unknown -> UNKNOWN

        Aurigin 'partially_spoofed' is normalized to Voice Shield 'SPOOFED' because
        it indicates the presence of spoofed/manipulated audio mixed with genuine voice.
        """
        if value is None:
            return "UNKNOWN"
        normalized = str(value).strip().lower()
        if normalized in {"real", "bonafide", "authentic"}:
            return "REAL"
        if normalized in {"spoof", "spoofed", "partially_spoofed", "fake", "manipulated"}:
            return "SPOOFED"
        return "UNKNOWN"


    @classmethod
    def normalize(cls, payload: dict[str, Any], processing_ms: int = 0) -> AuriginResult:
        global_block = payload.get("global") or {}
        raw_status = payload.get("status") or payload.get("state") or "PROCESSED"
        raw_result_str = global_block.get("result")
        normalized_result = cls.normalize_result_value(raw_result_str)
        reason = global_block.get("reason")
        
        # Aurigin returns:
        # - score: 0.0 - 1.0 (spoof probability)
        # - confidence: 0.0 - 1.0 (confidence in prediction)
        raw_score = global_block.get("score")
        raw_confidence = global_block.get("confidence")
        
        confidence_val: float | None = None
        if raw_confidence is not None:
            try:
                confidence_val = max(0.0, min(1.0, float(raw_confidence)))
            except (TypeError, ValueError):
                pass

        # Determine calibrated spoof score (0.0 to 1.0)
        spoof_prob: float | None = None
        if raw_score is not None:
            try:
                spoof_prob = max(0.0, min(1.0, float(raw_score)))
            except (TypeError, ValueError):
                pass

        # If spoof_score is missing, deduce from confidence and result value
        if spoof_prob is None:
            if normalized_result == "REAL":
                conf = confidence_val if confidence_val is not None else 0.95
                spoof_prob = round(max(0.0, 1.0 - conf), 4)
            elif normalized_result in ("SPOOF", "SPOOFED"):
                conf = confidence_val if confidence_val is not None else 0.95
                spoof_prob = round(max(0.5, conf), 4)
            else:
                spoof_prob = 0.45


        risk_score_int = int(round(spoof_prob * 100)) if spoof_prob is not None else 0
        raw_segments = payload.get("segments") or payload.get("segment_predictions") or []
        segments_list = raw_segments if isinstance(raw_segments, list) else []

        return AuriginResult(
            provider="aurigin",
            prediction_id=str(payload.get("prediction_id") or payload.get("id") or "unknown"),
            result=normalized_result,
            score=spoof_prob,
            confidence=confidence_val,
            risk_score=risk_score_int,
            segments=segments_list,
            processing_ms=processing_ms,
            raw_provider_status=str(raw_status),
            raw_result=str(raw_result_str) if raw_result_str is not None else None,
            reason=str(reason) if reason is not None else None,
        )


class AuriginService(VoiceDeepfakeProvider):
    """REST implementation of the official Aurigin /v1/predict contract."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.normalizer = AuriginResultNormalizer()

    def _normalize_response(self, payload: dict[str, Any], processing_ms: int) -> AuriginResult:
        return self.normalizer.normalize(payload, processing_ms)

    async def analyze_audio(
        self,
        audio_path_or_bytes: str | bytes | Any,
        filename: str,
        content_type: str | None = None,
        call_id: str | None = None,
        request_id: str | None = None,
        window_id: str | None = None,
        duration_ms: int | None = None,
    ) -> dict[str, Any]:
        if not self.settings.aurigin_enabled:
            return {
                "provider": "aurigin",
                "prediction_id": None,
                "result": "UNKNOWN",
                "confidence": None,
                "risk_score": None,
                "segments": [],
                "processing_ms": 0,
                "raw_provider_status": "DISABLED",
            }

        api_key = self.settings.aurigin_api_key
        if not api_key:
            logger.warning("Aurigin API key is not configured; realtime detection disabled")
            return {
                "provider": "aurigin",
                "prediction_id": None,
                "result": "UNKNOWN",
                "confidence": None,
                "risk_score": None,
                "segments": [],
                "processing_ms": 0,
                "raw_provider_status": "NOT_CONFIGURED",
            }

        if isinstance(audio_path_or_bytes, (str, Path)):
            file_path = Path(audio_path_or_bytes)
            if not file_path.exists():
                raise FileNotFoundError(f"Audio file not found: {file_path}")
            payload = file_path.read_bytes()
            effective_filename = filename or file_path.name
        else:
            payload = bytes(audio_path_or_bytes)
            effective_filename = filename or "chunk.wav"

        url = f"{self.settings.aurigin_api_base_url.rstrip('/')}/v1/predict"
        started = time.perf_counter()
        temp_path: Path | None = None
        if not payload:
            return {
                "provider": "aurigin",
                "prediction_id": None,
                "result": "UNKNOWN",
                "confidence": None,
                "risk_score": None,
                "segments": [],
                "processing_ms": 0,
                "raw_provider_status": "EMPTY_AUDIO",
            }

        calc_duration_ms = duration_ms if duration_ms is not None else 0

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=Path(effective_filename).suffix or ".wav") as temp_file:
                temp_file.write(payload)
                temp_path = Path(temp_file.name)

            headers = {"x-api-key": api_key}
            mime = content_type or "audio/wav"
            async with httpx.AsyncClient(timeout=self.settings.aurigin_timeout_seconds) as client:
                response = await client.post(
                    url,
                    headers=headers,
                    files={"file": (effective_filename, temp_path.read_bytes(), mime)},
                )
            latency_ms = int((time.perf_counter() - started) * 1000)
            response.raise_for_status()
            json_payload = response.json()
            result = self._normalize_response(json_payload, latency_ms)
            logger.debug("[AURIGIN_RAW_RESPONSE] prediction_id=%s payload=%s", result.prediction_id, json_payload)

            return {
                "provider": result.provider,
                "prediction_id": result.prediction_id,
                "result": result.result,
                "score": result.score,
                "confidence": result.confidence,
                "risk_score": result.risk_score,
                "segments": result.segments,
                "processing_ms": result.processing_ms,
                "raw_provider_status": result.raw_provider_status,
                "raw_result": result.raw_result,
                "reason": result.reason,
            }
        except httpx.TimeoutException:
            latency_ms = int((time.perf_counter() - started) * 1000)
            VoiceShieldPipelineLogger.detector_failed(
                window_id=window_id or "W01",
                error_type="TIMEOUT",
                message="Aurigin request timed out",
                retry_count=0,
            )
            return {
                "provider": "aurigin",
                "prediction_id": None,
                "result": "UNKNOWN",
                "confidence": None,
                "risk_score": None,
                "segments": [],
                "processing_ms": latency_ms,
                "raw_provider_status": "TIMEOUT",
            }
        except Exception as exc:  # noqa: BLE001
            latency_ms = int((time.perf_counter() - started) * 1000)
            VoiceShieldPipelineLogger.detector_failed(
                window_id=window_id or "W01",
                error_type="PROVIDER_ERROR",
                message=str(exc),
                retry_count=0,
            )
            return {
                "provider": "aurigin",
                "prediction_id": None,
                "result": "UNKNOWN",
                "confidence": None,
                "risk_score": None,
                "segments": [],
                "processing_ms": latency_ms,
                "raw_provider_status": "ERROR",
            }
        finally:
            if temp_path and temp_path.exists():
                try:
                    temp_path.unlink()
                except OSError:
                    pass


class AuriginStreamingProvider:
    """Placeholder for future Aurigin WebSocket streaming support. Implement when Aurigin streaming credentials/documentation are available."""

    def __init__(self) -> None:
        logger.warning("TODO: Implement when Aurigin streaming credentials/documentation are available.")

    async def analyze_audio(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("TODO: Implement when Aurigin streaming credentials/documentation are available.")
