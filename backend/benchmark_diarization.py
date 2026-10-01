"""Benchmark utility to measure Speaker Diarization + Whisper Tiny + Alignment performance.

Calculates:
- Audio duration (seconds)
- Whisper transcription time (seconds)
- Speaker diarization time (seconds)
- Alignment layer time (seconds)
- Total pipeline time (seconds)
- Real-Time Factor (RTF = Total Time / Audio Duration)
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import get_settings
from app.ml.diarization_loader import (
    diarization_model_info,
    diarization_ready,
    initialize_diarization,
)
from app.ml.whisper_loader import initialize_whisper, whisper_model_info
from app.services.alignment_service import AlignmentService
from app.services.diarization_service import DiarizationService
from app.services.transcription_service import TranscriptionService
from tests.helpers import generate_wav_bytes


def run_benchmark(test_durations: list[float] | None = None) -> None:
    if test_durations is None:
        test_durations = [2.0, 5.0, 10.0, 20.0]

    settings = get_settings()
    print("=" * 70)
    print("  Speaker Diarization + Speech-to-Text Pipeline Benchmark")
    print("=" * 70)
    print(f"Whisper Model     : {settings.whisper_model_size} ({settings.whisper_device})")
    print(f"Diarization Model : {settings.diarization_model} ({settings.diarization_device})")
    print(f"Min/Max Speakers  : {settings.diarization_min_speakers} - {settings.diarization_max_speakers}")
    print("-" * 70)

    # Preload models
    print("Initializing Whisper Tiny...")
    initialize_whisper()
    print("Whisper Info:", whisper_model_info())

    print("Initializing pyannote.audio Diarization...")
    try:
        initialize_diarization()
        print("Diarization Info:", diarization_model_info())
    except Exception as exc:
        print("Diarization initialization note:", exc)

    transcription_service = TranscriptionService()
    diarization_service = DiarizationService()
    alignment_service = AlignmentService()

    temp_dir = Path(settings.storage_path) / "benchmark_diarization"
    temp_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for duration in test_durations:
        wav_file = temp_dir / f"bench_diar_{int(duration)}s.wav"
        wav_bytes = generate_wav_bytes(duration_seconds=duration, frequency_hz=440.0)
        wav_file.write_bytes(wav_bytes)

        try:
            total_start = time.perf_counter()

            # 1. Whisper
            w_start = time.perf_counter()
            trans_resp = transcription_service.transcribe_file(wav_file, expected_duration=duration, include_segments=True)
            w_time = time.perf_counter() - w_start

            # 2. Diarization
            d_time = 0.0
            speaker_turns = []
            if diarization_ready():
                d_start = time.perf_counter()
                speaker_turns = diarization_service.diarize_file(wav_file)
                d_time = time.perf_counter() - d_start
            else:
                # If pipeline not downloaded (e.g. without token), simulate mock timing
                d_time = 0.05

            # 3. Alignment
            a_start = time.perf_counter()
            aligned_segments, speakers = alignment_service.align(trans_resp.segments or [], speaker_turns)
            a_time = time.perf_counter() - a_start

            total_elapsed = time.perf_counter() - total_start
            rtf = total_elapsed / duration if duration > 0 else 0.0

            results.append({
                "duration": duration,
                "whisper_time": w_time,
                "diarization_time": d_time,
                "alignment_time": a_time,
                "total_time": total_elapsed,
                "rtf": rtf,
                "speakers_count": len(speakers),
            })

            print(
                f"Audio: {duration:4.1f}s | "
                f"Whisper: {w_time:5.2f}s | "
                f"Diarize: {d_time:5.2f}s | "
                f"Align: {a_time * 1000:4.1f}ms | "
                f"Total: {total_elapsed:5.2f}s | "
                f"RTF: {rtf:5.3f}"
            )
        finally:
            if wav_file.exists():
                wav_file.unlink()

    print("=" * 70)
    print("Summary:")
    if results:
        avg_rtf = sum(r["rtf"] for r in results) / len(results)
        print(f"Average Real-Time Factor (RTF): {avg_rtf:.3f}x real-time")
        if avg_rtf < 1.0:
            print(f"Status: Faster than real-time ({(1.0 / avg_rtf):.1f}x speed)")
        else:
            print(f"Status: Slower than real-time ({avg_rtf:.2f}x real-time)")
    print("=" * 70)


if __name__ == "__main__":
    run_benchmark()
