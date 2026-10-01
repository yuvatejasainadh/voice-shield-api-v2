from __future__ import annotations

import os

import numpy as np
import soundfile as sf

os.environ.setdefault("PRELOAD_MODEL_ON_STARTUP", "false")

from app.audio.preprocessor import AudioPreprocessor
from tests.helpers import generate_wav_bytes


def test_preprocessor_mono_and_resample(tmp_path) -> None:
    file_path = tmp_path / "stereo_8k.wav"
    wav_bytes = generate_wav_bytes(duration_seconds=1.0, sample_rate=8000, channels=2)
    file_path.write_bytes(wav_bytes)

    preprocessor = AudioPreprocessor()
    output = preprocessor.prepare(str(file_path))

    assert output.sample_rate == 16000
    assert len(output.segments) == 1
    assert output.segments[0].ndim == 1


def test_preprocessor_output_shape_and_short_audio(tmp_path) -> None:
    file_path = tmp_path / "short.wav"
    file_path.write_bytes(generate_wav_bytes(duration_seconds=0.15, sample_rate=16000, channels=1))

    preprocessor = AudioPreprocessor()
    output = preprocessor.prepare(str(file_path))

    assert len(output.segments) == 1
    assert output.segments[0].shape[0] > 0


def test_preprocessor_silence_flag(tmp_path) -> None:
    file_path = tmp_path / "silence.wav"
    silence = np.zeros(16000, dtype=np.float32)
    sf.write(file_path, silence, 16000)

    preprocessor = AudioPreprocessor()
    output = preprocessor.prepare(str(file_path))

    assert output.near_silence is True


def test_preprocessor_segments_long_audio(tmp_path) -> None:
    file_path = tmp_path / "long.wav"
    file_path.write_bytes(generate_wav_bytes(duration_seconds=17.0, sample_rate=16000, channels=1))

    output = AudioPreprocessor().prepare(str(file_path))

    assert len(output.segments) == 3
    assert all(segment.ndim == 1 for segment in output.segments)
