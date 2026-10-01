"""TCED (Temporal Cascaded Evaluation/Detection) Window Manager.

Orchestrates multi-window temporal audio analysis around the detection engine (Aurigin),
handling:
- Context accumulation (e.g. 0-10s, 0-20s, 0-30s)
- Regular sliding windows (e.g. 10-40s, 20-50s)
- Tail-aligned terminal window on call termination (e.g. 24-54s for 54s call)
- Duplicate terminal window suppression (e.g. for 30s call)
- Timeline and window metadata preservation
"""

from __future__ import annotations

import io
import logging
import wave
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.core.pipeline_logger import VoiceShieldPipelineLogger

logger = logging.getLogger("voice-clone-detection")


def _pcm_to_wav(pcm_bytes: bytes, sample_rate: int = 16000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit mono PCM samples into standard WAV container bytes."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(pcm_bytes)
    return buf.getvalue()


def _strip_wav_header_if_present(data: bytes) -> bytes:
    """Extract raw PCM bytes from WAV container if RIFF header is present, else return as-is."""
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WAVE":
        try:
            with wave.open(io.BytesIO(data), "rb") as wf:
                return wf.readframes(wf.getnframes())
        except Exception:
            # If wave parsing fails, fallback to 44-byte standard header strip if long enough
            if len(data) > 44:
                return data[44:]
    return data


@dataclass
class TCEDConfig:
    """Centralized configuration for TCED windowing."""
    new_audio_interval_seconds: float = 2.5
    window_stride_seconds: float = 2.5
    max_window_length_seconds: float = 5.0
    sample_rate: int = 16000
    channels: int = 1
    bytes_per_sample: int = 2  # 16-bit PCM

    @property
    def bytes_per_ms(self) -> float:
        return (self.sample_rate * self.channels * self.bytes_per_sample) / 1000.0

    @property
    def new_audio_interval_ms(self) -> int:
        return int(self.new_audio_interval_seconds * 1000)

    @property
    def window_stride_ms(self) -> int:
        return int(self.window_stride_seconds * 1000)

    @property
    def max_window_length_ms(self) -> int:
        return int(self.max_window_length_seconds * 1000)

    @classmethod
    def from_settings(cls) -> TCEDConfig:
        settings = get_settings()
        return cls(
            new_audio_interval_seconds=settings.tced_new_audio_interval_seconds,
            window_stride_seconds=settings.tced_window_stride_seconds,
            max_window_length_seconds=settings.tced_max_window_length_seconds,
            sample_rate=settings.model_sample_rate,
            channels=1,
            bytes_per_sample=2,
        )


@dataclass
class TCEDWindow:
    """Represents an evaluation window extracted from the continuous call audio timeline."""
    window_id: str              # e.g. "W01", "W02", "W05"
    sequence: int               # 1, 2, 3, ...
    window_start_ms: int        # Timeline start in ms
    window_end_ms: int          # Timeline end in ms
    duration_ms: int            # Window duration in ms
    audio_bytes: bytes          # Raw 16-bit PCM audio bytes for this window
    is_tail: bool = False       # True if this is a tail-aligned terminal window
    sample_rate: int = 16000
    channels: int = 1

    def to_wav_bytes(self) -> bytes:
        """Wrap this window's PCM samples in standard WAV headers for detector consumption."""
        return _pcm_to_wav(self.audio_bytes, sample_rate=self.sample_rate, channels=self.channels)


class TCEDWindowManager:
    """
    Temporal Orchestration Layer for Real-Time Voice Clone Detection.
    
    Ingests continuous incoming call audio, buffers PCM stream, generates context accumulation
    and sliding windows at configured intervals, and emits a tail-aligned terminal window upon
    call end without duplicating already-analyzed windows.
    """

    def __init__(
        self,
        call_session_id: str,
        config: TCEDConfig | None = None,
    ) -> None:
        self.call_session_id = call_session_id
        self.config = config or TCEDConfig.from_settings()
        self.pcm_buffer = bytearray()
        self.generated_windows: list[TCEDWindow] = []
        self._generated_ranges: set[tuple[int, int]] = set()
        self._next_regular_interval_ms: int = self.config.new_audio_interval_ms

    @property
    def total_buffered_duration_ms(self) -> int:
        """Total duration of buffered PCM audio in milliseconds."""
        return int(len(self.pcm_buffer) / self.config.bytes_per_ms)

    def add_audio(self, audio_data: bytes) -> list[TCEDWindow]:
        """
        Append incoming audio data (PCM or WAV) to the session buffer and generate
        any newly eligible TCED windows.
        
        Returns:
            list[TCEDWindow]: Any newly generated windows in chronological sequence.
        """
        if not audio_data:
            return []

        # Extract raw PCM bytes if WAV headers are present
        raw_pcm = _strip_wav_header_if_present(audio_data)
        self.pcm_buffer.extend(raw_pcm)

        new_windows: list[TCEDWindow] = []
        total_ms = self.total_buffered_duration_ms

        while total_ms >= self._next_regular_interval_ms:
            end_ms = self._next_regular_interval_ms
            
            # Context accumulation vs sliding window calculation
            if end_ms <= self.config.max_window_length_ms:
                start_ms = 0
            else:
                start_ms = end_ms - self.config.max_window_length_ms

            dur_ms = end_ms - start_ms
            seq = len(self.generated_windows) + 1
            win_id = f"W{seq:02d}"

            start_byte = int(start_ms * self.config.bytes_per_ms)
            end_byte = int(end_ms * self.config.bytes_per_ms)
            window_pcm = bytes(self.pcm_buffer[start_byte:end_byte])

            win = TCEDWindow(
                window_id=win_id,
                sequence=seq,
                window_start_ms=start_ms,
                window_end_ms=end_ms,
                duration_ms=dur_ms,
                audio_bytes=window_pcm,
                is_tail=False,
                sample_rate=self.config.sample_rate,
                channels=self.config.channels,
            )

            self.generated_windows.append(win)
            self._generated_ranges.add((start_ms, end_ms))
            new_windows.append(win)

            self._next_regular_interval_ms += self.config.new_audio_interval_ms

        return new_windows

    def finalize_call(self, call_end_ms: int | None = None) -> TCEDWindow | None:
        """
        Handle call termination by generating a tail-aligned terminal window covering
        the final unprocessed audio segment.
        
        Formula:
            tail_start = max(0, call_end - MAX_WINDOW_LENGTH)
            tail_end   = call_end
            
        Duplicate prevention:
            If (tail_start, tail_end) has already been generated as a regular window,
            returns None to avoid duplicate evaluation.
        """
        effective_end_ms = call_end_ms if call_end_ms is not None else self.total_buffered_duration_ms
        if effective_end_ms <= 0 or len(self.pcm_buffer) == 0:
            return None

        # Detect unprocessed tail audio range
        last_regular_end = max(0, self._next_regular_interval_ms - self.config.new_audio_interval_ms)
        if effective_end_ms > last_regular_end:
            VoiceShieldPipelineLogger.tced_tail_detected(start_ms=last_regular_end, end_ms=effective_end_ms)

        tail_start_ms = max(0, effective_end_ms - self.config.max_window_length_ms)
        tail_end_ms = effective_end_ms

        # Prevent duplicate terminal window
        if (tail_start_ms, tail_end_ms) in self._generated_ranges:
            VoiceShieldPipelineLogger.tced_tail_suppressed(
                reason="DUPLICATE",
                start_ms=tail_start_ms,
                end_ms=tail_end_ms,
            )
            return None

        if tail_end_ms <= tail_start_ms:
            return None

        dur_ms = tail_end_ms - tail_start_ms
        seq = len(self.generated_windows) + 1
        win_id = f"W{seq:02d}"

        start_byte = int(tail_start_ms * self.config.bytes_per_ms)
        end_byte = int(tail_end_ms * self.config.bytes_per_ms)
        window_pcm = bytes(self.pcm_buffer[start_byte:end_byte])

        tail_win = TCEDWindow(
            window_id=win_id,
            sequence=seq,
            window_start_ms=tail_start_ms,
            window_end_ms=tail_end_ms,
            duration_ms=dur_ms,
            audio_bytes=window_pcm,
            is_tail=True,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
        )

        self.generated_windows.append(tail_win)
        self._generated_ranges.add((tail_start_ms, tail_end_ms))

        VoiceShieldPipelineLogger.tail_window_created(
            window_id=win_id,
            start_ms=tail_start_ms,
            end_ms=tail_end_ms,
            sequence=seq,
        )
        return tail_win
