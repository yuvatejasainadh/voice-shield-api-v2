"""Audio preprocessing and segmentation for model inference."""

from __future__ import annotations

import numpy as np
import soundfile as sf
from scipy import signal

from app.audio.metadata import PreprocessedAudio
from app.core.config import get_settings


class AudioPreprocessor:
    """Preprocesses audio with model-compatible sampling and segmentation."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.target_sample_rate = self.settings.model_sample_rate

    def prepare(self, audio_path: str) -> PreprocessedAudio:
        """Decode, convert to mono, resample, normalize, and segment audio.

        Steps:
        1. Decode the source audio with soundfile (falling back to librosa if needed).
        2. Convert to mono using averaging when source is multi-channel.
        3. Resample to model sample rate.
        4. Normalize peak amplitude to avoid clipping while preserving dynamics.
        5. Split into fixed-length segments for long audio.
        """
        try:
            waveform, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
            # waveform shape is (frames, channels)
            waveform = waveform.T
        except Exception:
            import librosa
            waveform, sample_rate = librosa.load(audio_path, sr=None, mono=False)

        if waveform.ndim > 1:
            mono = np.mean(waveform, axis=0)
        else:
            mono = waveform

        if sample_rate != self.target_sample_rate:
            num_target_samples = int(round(len(mono) * self.target_sample_rate / sample_rate))
            mono = signal.resample(mono, num_target_samples)

        mono = mono.astype(np.float32)
        peak = float(np.max(np.abs(mono))) if mono.size else 0.0
        if peak > 0:
            mono = mono / peak

        near_silence = self._is_near_silence(mono)
        segments = self._segment_audio(mono)
        duration_seconds = float(len(mono) / self.target_sample_rate) if self.target_sample_rate > 0 else 0.0

        return PreprocessedAudio(
            sample_rate=self.target_sample_rate,
            duration_seconds=duration_seconds,
            segments=segments,
            near_silence=near_silence,
        )

    def _segment_audio(self, mono_waveform: np.ndarray) -> list[np.ndarray]:
        segment_samples = int(self.settings.analysis_segment_seconds * self.target_sample_rate)
        if segment_samples <= 0:
            return [mono_waveform]

        total = len(mono_waveform)
        if total <= segment_samples:
            return [mono_waveform]

        segments: list[np.ndarray] = []
        for start in range(0, total, segment_samples):
            end = min(start + segment_samples, total)
            chunk = mono_waveform[start:end]
            if len(chunk) < int(0.2 * self.target_sample_rate):
                continue
            segments.append(chunk)

        if not segments:
            segments = [mono_waveform]

        return segments[: self.settings.max_segments_per_analysis]

    @staticmethod
    def _is_near_silence(waveform: np.ndarray) -> bool:
        if waveform.size == 0:
            return True
        rms = float(np.sqrt(np.mean(np.square(waveform))))
        return rms < 0.005
