"""Metadata structures for decoded audio."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class AudioMetadata:
    sample_rate: int
    channels: int
    duration_seconds: float
    num_samples: int
    extension: str


@dataclass(slots=True)
class PreprocessedAudio:
    sample_rate: int
    duration_seconds: float
    segments: list
    near_silence: bool
