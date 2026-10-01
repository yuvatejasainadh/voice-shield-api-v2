"""Base interface for spoof/AI speech detectors."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseDetector(ABC):
    """Contract used by DetectionService to keep ML implementation replaceable."""

    @property
    @abstractmethod
    def model_version(self) -> str:
        """Return the detector/model version string."""

    @abstractmethod
    def load(self) -> None:
        """Load model weights and preprocessing artifacts into memory."""

    @abstractmethod
    def predict(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        """Return probability outputs for one audio segment.

        Expected keys in the returned dictionary:
        - spoof_probability
        - bonafide_probability
        - confidence
        """
