"""Benchmark utility to measure Whisper Tiny transcription performance.

Calculates:
- Audio duration (seconds)
- Transcription inference time (seconds)
- Real-Time Factor (RTF = Inference Time / Audio Duration)
- Memory usage and device info
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

# Add backend directory to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.core.config import get_settings
from app.ml.whisper_loader import get_whisper_model, initialize_whisper, whisper_model_info
from app.services.transcription_service import TranscriptionService
from tests.helpers import generate_wav_bytes


def run_benchmark(test_durations: list[float] | None = None) -> None:
    if test_durations is None:
        test_durations = [2.0, 5.0, 10.0, 30.0]

    settings = get_settings()
    model_label = settings.whisper_model_size.capitalize()
    print("=" * 60)
    print(f"  Whisper {model_label} Transcription Performance Benchmark")
    print("=" * 60)
    print(f"Model Size     : {settings.whisper_model_size}")
    print(f"Target Device  : {settings.whisper_device}")
    print(f"Compute Type   : {settings.whisper_compute_type}")
    print(f"VAD Filter     : {settings.whisper_vad_filter}")
    print("-" * 60)

    # Initialize model
    print("Loading Whisper model...")
    load_start = time.perf_counter()
    initialize_whisper()
    load_time = time.perf_counter() - load_start
    print(f"Model loaded in {load_time:.2f} seconds.")
    print(f"Runtime Info   : {whisper_model_info()}")
    print("-" * 60)

    service = TranscriptionService()
    temp_dir = Path(settings.storage_path) / "benchmark"
    temp_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for duration in test_durations:
        wav_file = temp_dir / f"bench_{int(duration)}s.wav"
        wav_bytes = generate_wav_bytes(duration_seconds=duration, frequency_hz=440.0)
        wav_file.write_bytes(wav_bytes)

        try:
            # Warm-up / Run
            start_t = time.perf_counter()
            response = service.transcribe_file(wav_file, expected_duration=duration)
            elapsed = time.perf_counter() - start_t

            rtf = elapsed / duration if duration > 0 else 0.0
            results.append({
                "duration": duration,
                "inference_time": elapsed,
                "rtf": rtf,
                "detected_lang": response.language,
            })

            print(
                f"Audio: {duration:5.1f}s | "
                f"Inference: {elapsed:5.2f}s | "
                f"RTF: {rtf:5.3f} | "
                f"Lang: {response.language}"
            )
        finally:
            if wav_file.exists():
                wav_file.unlink()

    print("=" * 60)
    print("Summary:")
    avg_rtf = sum(r["rtf"] for r in results) / len(results) if results else 0
    print(f"Average Real-Time Factor (RTF): {avg_rtf:.3f}x real-time")
    if avg_rtf < 1.0:
        print(f"Status: Faster than real-time ({(1.0 / avg_rtf):.1f}x speed)")
    else:
        print(f"Status: Slower than real-time ({avg_rtf:.2f}x real-time)")
    print("=" * 60)


if __name__ == "__main__":
    run_benchmark()
