"""Concrete pretrained spoof detector implementation."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from app.core.config import get_settings
from app.ml.base_detector import BaseDetector

try:
    import torch
except ImportError:
    torch = None  # type: ignore[assignment]


logger = logging.getLogger("voice-clone-detection")


class SpoofDetector(BaseDetector):
    """Wrapper around a pretrained HF AST anti-spoof model."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self._device = "cuda" if (self.settings.enable_gpu and torch is not None and torch.cuda.is_available()) else "cpu"
        self._model = None
        self._feature_extractor = None
        self._model_version = self.settings.detector_model_id
        self._spoof_label_id: int | None = None
        self._bonafide_label_id: int | None = None

    @property
    def model_version(self) -> str:
        return self._model_version

    @property
    def device(self) -> str:
        return self._device

    def load(self) -> None:
        """Load model and feature extractor once.

        Determines label mapping dynamically from model.config.id2label.
        """
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification
        
        cache_dir = Path(self.settings.model_path)
        cache_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Loading detector model_id=%s device=%s", self.settings.detector_model_id, self._device)
        self._feature_extractor = AutoFeatureExtractor.from_pretrained(
            self.settings.detector_model_id,
            cache_dir=str(cache_dir),
        )
        self._model = AutoModelForAudioClassification.from_pretrained(
            self.settings.detector_model_id,
            cache_dir=str(cache_dir),
        )
        self._model.eval()
        self._model.to(self._device)

        id2label = getattr(self._model.config, "id2label", {}) or {}
        self._spoof_label_id = self._find_label_id(id2label, keywords=["spoof", "synthetic", "fake", "ai"])
        self._bonafide_label_id = self._find_label_id(id2label, keywords=["bonafide", "genuine", "human", "real"])

        if self._spoof_label_id is None or self._bonafide_label_id is None:
            raise RuntimeError(
                "Could not map model labels to spoof/bonafide classes. id2label="
                f"{id2label}"
            )

        logger.info(
            "Detector loaded model_id=%s version=%s device=%s spoof_label_id=%s bonafide_label_id=%s",
            self.settings.detector_model_id,
            self.model_version,
            self._device,
            self._spoof_label_id,
            self._bonafide_label_id,
        )

    def predict(self, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
        """Run inference for one preprocessed audio segment."""
        if self._model is None or self._feature_extractor is None:
            raise RuntimeError("Detector model is not loaded")

        with torch.no_grad():
            inputs = self._feature_extractor(
                audio,
                sampling_rate=sample_rate,
                return_tensors="pt",
            )
            inputs = {key: value.to(self._device) for key, value in inputs.items()}
            logits = self._model(**inputs).logits
            probs = torch.softmax(logits, dim=-1)[0].detach().cpu().numpy()

        spoof_probability = float(probs[self._spoof_label_id])
        bonafide_probability = float(probs[self._bonafide_label_id])
        confidence = float(max(spoof_probability, bonafide_probability))

        return {
            "spoof_probability": spoof_probability,
            "bonafide_probability": bonafide_probability,
            "confidence": confidence,
        }

    @staticmethod
    def _find_label_id(id2label: dict, keywords: list[str]) -> int | None:
        for key, label in id2label.items():
            normalized = str(label).strip().lower()
            if any(token in normalized for token in keywords):
                return int(key)
        return None
