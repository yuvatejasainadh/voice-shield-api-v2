"""Audio normalization utilities for provider compatibility."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile

import av
import numpy as np
import soundfile as sf

from app.core.config import get_settings

logger = logging.getLogger("voice-clone-detection")


class AudioNormalizer:
    """Normalizes arbitrary audio formats (.m4a, .aac, .webm, etc.) into standard 16kHz mono WAV."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def normalize_to_wav(
        self,
        input_path: Path | str,
        target_sample_rate: int = 16000,
        output_dir: Path | str | None = None,
    ) -> Path:
        """Decode input audio file and convert to 16kHz mono PCM WAV.

        Returns path to the newly created temporary WAV file.
        Caller is responsible for unlinking output_path in a try/finally block.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Audio file {input_path} does not exist")

        target_dir = Path(output_dir or self.settings.storage_path) / "temp"
        target_dir.mkdir(parents=True, exist_ok=True)

        fd, temp_path_str = tempfile.mkstemp(suffix="_normalized.wav", dir=target_dir)
        import os
        os.close(fd)
        output_path = Path(temp_path_str)

        try:
            # 1. Try decoding with PyAV (supports all containers and codecs)
            container = av.open(str(input_path))
            audio_stream = next((s for s in container.streams if s.type == "audio"), None)
            if audio_stream is None:
                raise ValueError("No audio stream found in file")

            resampler = av.AudioResampler(format="flt", layout="mono", rate=target_sample_rate)
            samples_list: list[np.ndarray] = []

            for frame in container.decode(audio_stream):
                resampled_frames = resampler.resample(frame)
                for rframe in resampled_frames:
                    samples_list.append(rframe.to_ndarray())

            if not samples_list:
                raise ValueError("Decoded audio produced zero samples")

            audio_data = np.concatenate(samples_list, axis=1).squeeze(0)
            sf.write(str(output_path), audio_data, target_sample_rate, subtype="PCM_16")

            logger.info(
                "Normalized audio input=%s (%s) -> 16kHz mono WAV size=%d bytes",
                input_path.name,
                input_path.suffix,
                output_path.stat().st_size,
            )
            return output_path

        except Exception as exc:
            if output_path.exists():
                try:
                    output_path.unlink()
                except Exception:
                    pass
            logger.error("Audio normalization failed for %s: %s", input_path.name, exc)
            raise RuntimeError(f"Failed to normalize audio {input_path.name}: {exc}") from exc
