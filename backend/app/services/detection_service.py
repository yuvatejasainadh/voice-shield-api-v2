"""Detection service abstraction for real anti-spoof model inference."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
from fastapi import HTTPException, status

from app.audio.preprocessor import AudioPreprocessor
from app.ml.model_loader import detector_ready, detector_version, get_detector, initialize_detector


logger = logging.getLogger("voice-clone-detection")

class DetectionService:
    """Coordinates preprocessing and model inference for one analysis request."""

    def analyze(self, audio_path: Path) -> dict:
        """Run real model inference and return normalized detector output."""
        if not detector_ready():
            initialize_detector()

        if not detector_ready():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "MODEL_NOT_READY", "message": "Detection model is not ready"},
            )

        detector = get_detector()
        if detector is None:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail={"code": "MODEL_NOT_READY", "message": "Detection model is not ready"},
            )

        start = time.perf_counter()
        try:
            logger.info("Analysis started path=%s", audio_path.name)
            preprocessor = AudioPreprocessor()
            prepared = preprocessor.prepare(str(audio_path))

            segment_probs: list[float] = []
            segment_confidences: list[float] = []
            for segment in prepared.segments:
                result = detector.predict(segment, prepared.sample_rate)
                segment_probs.append(float(result["spoof_probability"]))
                segment_confidences.append(float(result["confidence"]))

            ai_probability = self._aggregate_segment_probabilities(segment_probs)
            confidence = self._aggregate_confidence(segment_confidences, segment_probs)
            elapsed_ms = int((time.perf_counter() - start) * 1000)

            reasons = self._build_reasons(ai_probability, prepared.near_silence)
            logger.info(
                "Inference completed segments=%s ai_probability=%.4f confidence=%.4f processing_time_ms=%s",
                len(prepared.segments),
                ai_probability,
                confidence,
                elapsed_ms,
            )

            return {
                "ai_probability": ai_probability,
                "confidence": confidence,
                "duration_seconds": round(prepared.duration_seconds, 3),
                "segments_analyzed": len(prepared.segments),
                "processing_time_ms": elapsed_ms,
                "detector_version": detector_version() or "unknown",
                "reasons": reasons,
            }
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("Detection failed: %s", exc)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail={"code": "DETECTION_FAILED", "message": "Model inference failed"},
            ) from exc

    @staticmethod
    def _aggregate_segment_probabilities(segment_probs: list[float]) -> float:
        """Robustly aggregate segment-level spoof probabilities.

        Strategy:
        - Use p75 to reduce sensitivity to isolated outliers.
        - Mix with max score to retain sensitivity to concentrated spoof evidence.
        """
        if not segment_probs:
            return 0.0
        if len(segment_probs) == 1:
            return float(np.clip(segment_probs[0], 0.0, 1.0))

        scores = np.array(segment_probs, dtype=np.float32)
        p75 = float(np.percentile(scores, 75))
        max_score = float(np.max(scores))
        combined = (0.7 * p75) + (0.3 * max_score)
        return float(np.clip(combined, 0.0, 1.0))

    @staticmethod
    def _aggregate_confidence(segment_confidences: list[float], segment_probs: list[float]) -> float:
        if not segment_confidences:
            return 0.5

        base = float(np.mean(segment_confidences))
        spread = float(np.std(np.array(segment_probs, dtype=np.float32))) if len(segment_probs) > 1 else 0.0
        calibrated = base * (1.0 - min(spread, 0.4))
        return float(np.clip(calibrated, 0.0, 1.0))

    @staticmethod
    def _build_reasons(ai_probability: float, near_silence: bool) -> list[str]:
        reasons: list[str] = []
        if ai_probability >= 0.7:
            reasons.append("Model detected characteristics associated with synthetic speech")
        elif ai_probability >= 0.3:
            reasons.append("Model found mixed indicators between genuine and synthetic speech")
        else:
            reasons.append("Model did not detect strong synthetic speech indicators")

        if near_silence:
            reasons.append("Low-energy audio may reduce confidence in detection reliability")

        return reasons
