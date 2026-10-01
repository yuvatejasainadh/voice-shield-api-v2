"""Decode-aware audio validation utilities for multi-format audio (.wav, .mp3, .m4a, .aac, .ogg, .flac, .webm)."""

from __future__ import annotations

from pathlib import Path
import logging

import numpy as np
import soundfile as sf
from fastapi import HTTPException, status

from app.audio.metadata import AudioMetadata
from app.core.config import get_settings
from app.utils.audio_utils import get_extension, is_allowed_extension

logger = logging.getLogger("voice-clone-detection")


class AudioValidator:
    """Validates uploaded audio files beyond extension checks using soundfile and PyAV."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def validate_extension(self, filename: str) -> str:
        if not is_allowed_extension(filename):
            raise HTTPException(
                status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
                detail={"code": "UNSUPPORTED_AUDIO_FORMAT", "message": "Unsupported audio format"},
            )
        return get_extension(filename)

    def _decode_audio_info(self, file_path: Path) -> tuple[int, int, int, float]:
        """Attempt decoding with soundfile, falling back to PyAV for container formats.

        Returns (sample_rate, channels, num_samples, duration_seconds).
        """
        # 1. Primary decoder: soundfile (fast C libsndfile for WAV, MP3, OGG, FLAC)
        try:
            info = sf.info(str(file_path))
            if info.samplerate > 0 and info.frames > 0:
                duration = float(info.duration)
                return int(info.samplerate), int(info.channels), int(info.frames), duration
        except Exception:
            pass

        # 2. Secondary decoder: PyAV (ffmpeg C bindings for M4A, AAC, WEBM, etc.)
        try:
            import av
            container = av.open(str(file_path))
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is None:
                raise ValueError("No audio stream found")

            sr = int(audio_stream.codec_context.sample_rate or audio_stream.rate or 16000)
            channels = int(audio_stream.codec_context.channels or 1)

            total_samples = 0
            for frame in container.decode(audio_stream):
                total_samples += frame.samples

            if total_samples <= 0:
                raise ValueError("Decoded zero audio samples")

            duration = float(total_samples / sr) if sr > 0 else 0.0
            return sr, channels, total_samples, duration
        except Exception as exc:
            logger.warning("Audio decoding failed for %s: %s", file_path.name, exc)
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "INVALID_AUDIO", "message": "Audio could not be decoded"},
            ) from exc

    def validate_saved_file(self, file_path: Path, extension: str) -> AudioMetadata:
        if not file_path.exists() or not file_path.is_file():
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "INVALID_AUDIO", "message": "Audio file is missing"},
            )

        size = file_path.stat().st_size
        if size <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "EMPTY_AUDIO", "message": "Audio file is empty"},
            )

        if size > self.settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={"code": "FILE_TOO_LARGE", "message": "File too large"},
            )

        sample_rate, channels, num_samples, duration_seconds = self._decode_audio_info(file_path)

        if duration_seconds <= 0.0 or num_samples <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "INVALID_AUDIO", "message": "Audio duration is invalid"},
            )

        if duration_seconds > self.settings.max_audio_duration_seconds:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={
                    "code": "AUDIO_TOO_LONG",
                    "message": (
                        "Audio duration exceeds maximum allowed duration "
                        f"({self.settings.max_audio_duration_seconds}s)"
                    ),
                },
            )

        return AudioMetadata(
            sample_rate=sample_rate,
            channels=channels,
            duration_seconds=round(duration_seconds, 3),
            num_samples=num_samples,
            extension=extension,
        )
