from __future__ import annotations

import os

os.environ.setdefault("PRELOAD_MODEL_ON_STARTUP", "false")

from app.ml import model_loader


def test_model_load_failure_reports_not_ready(monkeypatch) -> None:
    class _FailingDetector:
        def load(self) -> None:
            raise RuntimeError("weights unavailable")

    monkeypatch.setattr(model_loader, "SpoofDetector", _FailingDetector)
    monkeypatch.setattr(model_loader, "_detector_instance", None)
    monkeypatch.setattr(model_loader, "_detector_error", None)

    model_loader.initialize_detector()

    assert model_loader.detector_ready() is False
    assert model_loader.detector_version() is None
    assert model_loader.detector_error() == "weights unavailable"
