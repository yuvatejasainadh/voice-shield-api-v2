"""Service for deepfake and synthetic voice detection using the official Reality Defender RealAPI."""

from __future__ import annotations

import logging
import os
from pathlib import Path
import time
from typing import Any

from fastapi import HTTPException, status
from realitydefender import RealityDefender
from realitydefender.core.constants import DEFAULT_MAX_ATTEMPTS, DEFAULT_POLLING_INTERVAL
from realitydefender.errors import RealityDefenderError

from app.core.config import get_settings

logger = logging.getLogger("voice-clone-detection")


class RealityDefenderService:
    """Encapsulates all communication with the Reality Defender RealAPI for voice clone detection."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def _get_client(self) -> RealityDefender:
        """Instantiate and return the configured RealityDefender SDK client."""
        api_key = self.settings.reality_defender_api_key
        if not api_key:
            logger.error("Reality Defender API key is not configured")
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={
                    "code": "REALITY_DEFENDER_NOT_CONFIGURED",
                    "message": "Reality Defender API key is not configured",
                },
            )
        base_url = self.settings.reality_defender_api_base_url or None
        return RealityDefender(api_key=api_key, base_url=base_url)

    async def analyze_file(self, file_path: Path | str) -> dict[str, Any]:
        """Upload and analyze an audio file with Reality Defender, polling for final results."""
        file_path = Path(file_path)
        if not file_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"code": "AUDIO_NOT_FOUND", "message": f"Audio file {file_path.name} not found"},
            )

        client = self._get_client()
        start_time = time.perf_counter()

        # Calculate max polling attempts based on timeout setting (default 2s interval)
        polling_interval_ms = DEFAULT_POLLING_INTERVAL
        timeout_ms = self.settings.reality_defender_timeout_seconds * 1000
        max_attempts = max(5, timeout_ms // polling_interval_ms)

        logger.info(
            "Reality Defender analysis started file=%s timeout=%ds max_attempts=%d",
            file_path.name,
            self.settings.reality_defender_timeout_seconds,
            max_attempts,
        )

        # Check native Reality Defender formats
        native_exts = {"wav", "mp3", "m4a", "flac", "ogg"}
        ext = file_path.suffix.lstrip(".").lower()

        normalized_temp_path: Path | None = None
        target_file_path = file_path

        if ext not in native_exts:
            from app.utils.audio_normalizer import AudioNormalizer
            logger.info("Format .%s not natively supported by Reality Defender; normalizing to WAV", ext)
            normalizer = AudioNormalizer()
            normalized_temp_path = normalizer.normalize_to_wav(file_path)
            target_file_path = normalized_temp_path

        try:
            # 1. Upload audio file
            upload_res = await client.upload(file_path=str(target_file_path))
            request_id = upload_res.get("request_id")
            if not request_id:
                raise RealityDefenderError("No request_id returned from upload", "upload_failed")

            logger.info("Reality Defender upload successful request_id=%s", request_id)

            # 2. Poll for detection result
            detection_result = await client.get_result(
                request_id=request_id,
                max_attempts=max_attempts,
                polling_interval=polling_interval_ms,
            )

            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            return self._normalize_result(detection_result, processing_time_ms)

        except RealityDefenderError as rde:
            logger.error("Reality Defender SDK error: %s (code=%s)", rde.message, rde.code)
            if rde.code == "unauthorized":
                raise HTTPException(
                    status_code=status.HTTP_502_BAD_GATEWAY,
                    detail={
                        "code": "REALITY_DEFENDER_AUTH_ERROR",
                        "message": "Reality Defender API authentication failed",
                    },
                )
            if rde.code == "timeout":
                raise HTTPException(
                    status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                    detail={
                        "code": "REALITY_DEFENDER_TIMEOUT",
                        "message": "Reality Defender detection timed out",
                    },
                )
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"code": "REALITY_DEFENDER_ERROR", "message": rde.message},
            )
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Unexpected error during Reality Defender analysis: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail={"code": "REALITY_DEFENDER_ERROR", "message": f"Reality Defender analysis failed: {exc}"},
            ) from exc
        finally:
            if normalized_temp_path and normalized_temp_path.exists():
                try:
                    normalized_temp_path.unlink()
                except Exception as exc:
                    logger.warning("Failed to remove temporary normalized WAV file %s: %s", normalized_temp_path, exc)
            try:
                await client.cleanup()
            except Exception:
                pass

    def _normalize_result(self, result: dict[str, Any], processing_time_ms: int) -> dict[str, Any]:
        """Map Reality Defender detection result into the backend's internal normalized structure."""
        request_id = str(result.get("request_id", "unknown"))
        raw_status = str(result.get("status", "UNKNOWN")).upper()
        raw_score = result.get("score")  # float 0.0 - 1.0 or None

        # Determine AI manipulation probability
        ai_probability: float | None = None
        if raw_score is not None:
            try:
                ai_probability = max(0.0, min(1.0, float(raw_score)))
            except (ValueError, TypeError):
                ai_probability = None

        # Derive classification based on Reality Defender status and risk thresholds
        if raw_status in ("MANIPULATED", "FAKE"):
            classification = "LIKELY_AI_GENERATED"
        elif raw_status in ("AUTHENTIC", "REAL"):
            classification = "LIKELY_GENUINE"
        elif ai_probability is not None:
            if ai_probability >= (self.settings.risk_suspicious_max / 100.0):
                classification = "LIKELY_AI_GENERATED"
            elif ai_probability <= (self.settings.risk_genuine_max / 100.0):
                classification = "LIKELY_GENUINE"
            else:
                classification = "SUSPICIOUS"
        else:
            classification = "SUSPICIOUS"

        # Risk score 0-100
        risk_score = round(ai_probability * 100) if ai_probability is not None else 0

        # Build explainability reasons from individual sub-model outputs
        reasons: list[str] = []
        models_data = result.get("models", [])
        for model in models_data:
            m_name = model.get("name", "Detector")
            m_status = model.get("status", "UNKNOWN")
            m_score = model.get("score")
            if m_status in ("MANIPULATED", "FAKE", "SUSPICIOUS"):
                score_str = f" (score: {m_score})" if m_score is not None else ""
                reasons.append(f"{m_name} detected manipulation indicators{score_str}")

        if not reasons:
            if classification == "LIKELY_AI_GENERATED":
                reasons.append("Reality Defender detected characteristics associated with synthetic speech")
            elif classification == "LIKELY_GENUINE":
                reasons.append("No significant deepfake or synthetic voice manipulation detected")
            else:
                reasons.append("Analysis inconclusive; audio exhibits mixed or ambiguous acoustic indicators")

        logger.info(
            "Reality Defender analysis normalized request_id=%s classification=%s risk_score=%d ai_prob=%s",
            request_id,
            classification,
            risk_score,
            ai_probability,
        )

        return {
            "analysis_id": request_id,
            "status": "completed",
            "classification": classification,
            "risk_score": risk_score,
            "confidence": None,  # Provider does not return a separate calibrated confidence
            "ai_probability": ai_probability,
            "duration_seconds": 0.0,
            "segments_analyzed": len(models_data) if models_data else 1,
            "detector_version": "reality-defender",
            "processing_time_ms": processing_time_ms,
            "reasons": reasons,
        }
