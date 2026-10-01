from __future__ import annotations

import os

import pytest

from app.ml.model_loader import detector_ready, initialize_detector, get_detector
from tests.helpers import generate_wav_bytes


@pytest.mark.integration
def test_real_model_inference_optional(tmp_path) -> None:
    if os.getenv("RUN_REAL_MODEL_TESTS", "0") != "1":
        pytest.skip("Set RUN_REAL_MODEL_TESTS=1 to run real model integration test")

    file_path = tmp_path / "integration.wav"
    file_path.write_bytes(generate_wav_bytes(duration_seconds=1.0, sample_rate=16000, channels=1))

    initialize_detector()
    assert detector_ready() is True

    detector = get_detector()
    assert detector is not None

    import librosa

    audio, sr = librosa.load(str(file_path), sr=16000, mono=True)
    output = detector.predict(audio, sr)

    assert 0.0 <= output["spoof_probability"] <= 1.0
    assert 0.0 <= output["bonafide_probability"] <= 1.0
    assert 0.0 <= output["confidence"] <= 1.0
