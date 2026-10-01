"""Benchmark script comparing Sarvam Saaras v3 vs v4 on Telugu audio recordings."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
import sys
import time
from typing import Any

from app.core.config import get_settings
from app.services.sarvam_transcription_service import SarvamTranscriptionService
from app.services.transcript_quality_evaluator import evaluate_transcription_quality

# Configure UTF-8 stdout
sys.stdout.reconfigure(encoding="utf-8")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sarvam-benchmark")


async def evaluate_model(
    service: SarvamTranscriptionService,
    audio_path: Path,
    model_name: str,
    duration: float,
) -> dict[str, Any]:
    """Transcribe a sample with a specific model and compute quality benchmarks."""
    logger.info("Evaluating model %s on %s...", model_name, audio_path.name)
    t0 = time.perf_counter()
    
    # Override model setting dynamically
    service.settings.sarvam_model = model_name
    res = await service.transcribe_file(
        file_path=audio_path,
        language="te-IN",
        output_mode="codemix",
        with_diarization=True,
        expected_duration=duration,
    )
    latency_ms = int((time.perf_counter() - t0) * 1000)

    quality = evaluate_transcription_quality(
        res,
        expected_duration=duration,
        requested_language="te-IN",
    )

    return {
        "model": model_name,
        "language": res.get("language_code") or "te-IN",
        "duration_seconds": duration,
        "latency_ms": latency_ms,
        "quality_score": quality.score,
        "quality_status": quality.status,
        "quality_reasons": quality.reasons,
        "telugu_ratio": quality.metrics.get("telugu_ratio", 0.0),
        "latin_ratio": quality.metrics.get("latin_ratio", 0.0),
        "foreign_script_ratio": quality.metrics.get("foreign_script_ratio", 0.0),
        "replacement_chars": quality.metrics.get("replacement_chars", 0),
        "repetition_ratio": quality.metrics.get("repetition_ratio", 0.0),
        "has_excessive_repetition": quality.metrics.get("has_excessive_repetition", False),
        "speaker_count": len(res.get("speakers", [])),
        "utterance_count": len(res.get("speaker_transcript", [])),
        "timestamp_valid": len(quality.metrics.get("timestamps", {}).get("details", [])) == 0,
        "transcript_preview": res.get("transcript", "")[:200],
    }


async def main() -> None:
    settings = get_settings()
    if not settings.sarvam_api_key:
        print("SARVAM_API_KEY is not configured.")
        return

    storage = Path("storage")
    wav_files = sorted(storage.glob("*.wav"), key=lambda f: f.stat().st_size, reverse=True)
    if not wav_files:
        print("No .wav files found in storage/.")
        return

    sample = wav_files[0]
    print(f"Benchmarking on {sample.name} ({sample.stat().st_size / 1024 / 1024:.2f} MB)...")

    service = SarvamTranscriptionService()

    # Model evaluation
    v4_res = await evaluate_model(service, sample, "saaras:v4", duration=158.78)
    v3_res = await evaluate_model(service, sample, "saaras:v3", duration=158.78)

    print("\n=======================================================")
    print("SARVAM SAARAS v3 vs v4 TELUGU BENCHMARK RESULTS")
    print("=======================================================")
    print(json.dumps({"saaras:v4": v4_res, "saaras:v3": v3_res}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
