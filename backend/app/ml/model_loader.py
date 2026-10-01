"""Singleton detector loading and readiness tracking."""

from __future__ import annotations

import logging

from app.ml.base_detector import BaseDetector
from app.ml.spoof_detector import SpoofDetector


logger = logging.getLogger("voice-clone-detection")

_detector_instance: BaseDetector | None = None
_detector_error: str | None = None


def initialize_detector() -> None:
    """Load the configured detector and keep it in memory for future requests."""
    global _detector_instance, _detector_error
    if _detector_instance is not None:
        return

    try:
        detector = SpoofDetector()
        detector.load()
        _detector_instance = detector
        _detector_error = None
    except Exception as exc:
        _detector_instance = None
        _detector_error = str(exc)
        logger.exception("Detector initialization failed: %s", exc)


def get_detector() -> BaseDetector | None:
    return _detector_instance


def detector_ready() -> bool:
    return _detector_instance is not None


def detector_error() -> str | None:
    return _detector_error


def detector_version() -> str | None:
    if _detector_instance is None:
        return None
    return _detector_instance.model_version
