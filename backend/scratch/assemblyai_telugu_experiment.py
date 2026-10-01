#!/usr/bin/env python3
"""
AssemblyAI Telugu/Multilingual Controlled Experiment Script
Tests A, B, C, D as specified.

Usage:
    .venv\Scripts\python.exe scratch/assemblyai_telugu_experiment.py
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import unicodedata
from pathlib import Path

# Load .env before importing app modules
from dotenv import load_dotenv
load_dotenv()

import assemblyai as aai

# ── Unicode diagnostics ──────────────────────────────────────────────────────

def unicode_diagnostics(text: str) -> dict:
    """Count character distribution by Unicode block. Never log the raw text."""
    if not text:
        return {"telugu": 0, "devanagari": 0, "latin": 0, "replacement": 0, "other": 0, "total": 0}
    counts = {"telugu": 0, "devanagari": 0, "latin": 0, "replacement": 0, "other": 0, "total": 0}
    for ch in text:
        counts["total"] += 1
        cp = ord(ch)
        if 0x0C00 <= cp <= 0x0C7F:
            counts["telugu"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["devanagari"] += 1
        elif (0x0041 <= cp <= 0x005A) or (0x0061 <= cp <= 0x007A):
            counts["latin"] += 1
        elif cp == 0xFFFD:
            counts["replacement"] += 1
        else:
            counts["other"] += 1
    return counts


def log_utterances(utterances, label: str):
    """Log utterance metadata without logging transcript text (PII protection)."""
    if not utterances:
        print(f"  {label}: 0 utterances")
        return
    print(f"  {label}: {len(utterances)} utterances")
    for i, u in enumerate(utterances):
        spk = getattr(u, 'speaker', 'N/A')
        start = getattr(u, 'start', 0) / 1000.0
        end = getattr(u, 'end', 0) / 1000.0
        conf = getattr(u, 'confidence', None)
        diag = unicode_diagnostics(getattr(u, 'text', ''))
        print(f"    [{i}] speaker={spk} start={start:.3f}s end={end:.3f}s conf={conf}")
        print(f"         unicode: te={diag['telugu']} deva={diag['devanagari']} lat={diag['latin']} repl={diag['replacement']}")


def run_experiment(label: str, config: aai.TranscriptionConfig, audio_path: Path, api_key: str):
    """Run one AssemblyAI transcription experiment synchronously (wraps async via asyncio.run)."""
    print(f"\n{'='*60}")
    print(f"TEST {label}")
    print(f"{'='*60}")

    async def _run():
        t0 = time.perf_counter()
        async with aai.AsyncTranscriber(api_key=api_key, config=config) as transcriber:
            transcript = await transcriber.transcribe(str(audio_path))
        elapsed_ms = int((time.perf_counter() - t0) * 1000)
        return transcript, elapsed_ms

    try:
        transcript, elapsed_ms = asyncio.run(_run())
    except Exception as exc:
        print(f"  ERROR: {exc}")
        return None

    # Check for error status
    error = getattr(transcript, 'error', None)
    if error:
        print(f"  PROVIDER ERROR: {error}")
        return None

    text = getattr(transcript, 'text', '') or ''
    language_code = getattr(transcript, 'language_code', None)
    audio_duration = getattr(transcript, 'audio_duration', None)
    utterances = getattr(transcript, 'utterances', None) or []

    diag = unicode_diagnostics(text)
    speakers_seen = sorted(set(getattr(u, 'speaker', '?') for u in utterances))

    print(f"  Latency:      {elapsed_ms} ms")
    print(f"  Language:     {language_code}")
    print(f"  Duration:     {audio_duration}s")
    print(f"  Speakers:     {speakers_seen} ({len(utterances)} utterances)")
    print(f"  Unicode dist: te={diag['telugu']} deva={diag['devanagari']} lat={diag['latin']} repl={diag['replacement']} total={diag['total']}")
    print(f"  RAW TEXT (first 200 chars repr): {repr(text[:200])}")
    log_utterances(utterances, "Utterances")

    # Gap analysis
    if len(utterances) > 1:
        print("  Gaps between utterances:")
        for i in range(len(utterances) - 1):
            u1_end = getattr(utterances[i], 'end', 0) / 1000.0
            u2_start = getattr(utterances[i+1], 'start', 0) / 1000.0
            gap = u2_start - u1_end
            if gap > 1.0:
                print(f"    GAP: end={u1_end:.3f}s → next_start={u2_start:.3f}s gap={gap:.3f}s  (investigate: silence/noise/omission?)")

    return {
        "language_code": language_code,
        "transcript_len": len(text),
        "unicode": diag,
        "speakers": speakers_seen,
        "utterance_count": len(utterances),
        "latency_ms": elapsed_ms,
        "has_replacement_chars": diag["replacement"] > 0,
        "has_devanagari": diag["devanagari"] > 0,
    }


def main():
    api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: ASSEMBLYAI_API_KEY not set in .env")
        sys.exit(1)

    # Find audio files
    backend_dir = Path(__file__).resolve().parents[1]
    english_path = backend_dir / "testvoices" / "genuine" / "LJ001-0004.wav"

    # Find the largest file in storage as the real Telugu candidate
    storage = backend_dir / "storage"
    wav_files = sorted(storage.glob("*.wav"), key=lambda f: f.stat().st_size, reverse=True)
    telugu_path = wav_files[0] if wav_files else None

    print("\n" + "="*70)
    print("ASSEMBLYAI TELUGU/MULTILINGUAL EXPERIMENT")
    print("="*70)
    print(f"SDK VERSION: {aai.__version__}")
    print(f"API KEY: {'*'*8 + api_key[-4:] if api_key else 'NOT SET'}")
    print(f"English audio: {english_path} (exists={english_path.exists()})")
    print(f"Telugu audio: {telugu_path} (exists={telugu_path.exists() if telugu_path else False})")

    # ── API parameter verification ──────────────────────────────────────────
    print("\n--- SDK PARAMETERS VERIFIED ---")
    print("speech_model values: best, nano, slam_1, universal")
    print("speech_models: Optional[List[str]] — model priority list e.g. ['slam-1', 'universal']")
    print("prompt: Optional[str] — supported by Universal model")
    print("language_code: Optional[str] — 'te' as raw string (NOT in LanguageCode enum)")
    print("language_detection: Optional[bool] — enables auto-detection")
    print("speaker_labels: Optional[bool] — enables diarization")
    print("NOTE: Telugu NOT in LanguageCode enum — must pass 'te' as raw string")
    print("NOTE: Universal model supports 99 languages incl. Telugu, Hindi, Tamil, etc.")
    print("NOTE: Slam-1 = English only. Universal = 99 languages.")
    print("NOTE: 'best' speech_model defaults to Slam-1 for English, Universal for others")

    if not telugu_path:
        print("\nERROR: No audio found in storage. Cannot run Telugu experiments.")
        sys.exit(1)

    # ─── TEST A: Current production config (auto-detect, universal, speaker labels) ───
    config_a = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=2,
        language_detection=True,
        speech_model=aai.SpeechModel.universal,
    )
    result_a = run_experiment("A — Current Production (universal, language_detection=True)", config_a, telugu_path, api_key)

    # ─── TEST B: Explicit Telugu, universal model ─────────────────────────────────
    # Note: 'te' must be passed as raw string — not in LanguageCode enum
    config_b = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=2,
        language_detection=False,
        speech_model=aai.SpeechModel.universal,
        # language_code as raw string since 'te' is not in SDK enum
    )
    # Inject language_code via raw_transcription_config workaround
    config_b.raw.language_code = "te"
    result_b = run_experiment("B — Explicit Telugu (language_code=te, universal, no auto-detect)", config_b, telugu_path, api_key)

    # ─── TEST C: Auto-detect with language_preservation prompt ────────────────────
    PRESERVATION_PROMPT = (
        "Transcribe the conversation exactly as spoken. "
        "Preserve the original language and script of every word. "
        "Do not translate Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, or English into another language or script. "
        "Preserve natural Telugu-English code-switching. "
        "Do not transliterate Telugu into Devanagari or Latin script."
    )
    config_c = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=2,
        language_detection=True,
        speech_model=aai.SpeechModel.universal,
        prompt=PRESERVATION_PROMPT,
    )
    result_c = run_experiment("C — Auto-detect + Language Preservation Prompt", config_c, telugu_path, api_key)

    # ─── TEST D: speech_models priority list [slam-1, universal] ─────────────────
    config_d = aai.TranscriptionConfig(
        speaker_labels=True,
        speakers_expected=2,
        language_detection=True,
        speech_models=["slam-1", "universal"],
    )
    result_d = run_experiment("D — speech_models priority [slam-1, universal]", config_d, telugu_path, api_key)

    # ── English sanity check ─────────────────────────────────────────────────
    if english_path.exists():
        print("\n" + "="*60)
        print("ENGLISH SANITY CHECK — LJ001-0004")
        print("="*60)
        config_en = aai.TranscriptionConfig(
            speaker_labels=True,
            speakers_expected=1,
            language_detection=True,
            speech_model=aai.SpeechModel.universal,
        )
        result_en = run_experiment("ENGLISH", config_en, english_path, api_key)

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "="*70)
    print("EXPERIMENT SUMMARY")
    print("="*70)
    for label, result in [("A", result_a), ("B", result_b), ("C", result_c), ("D", result_d)]:
        if result:
            te = result["unicode"]["telugu"]
            deva = result["unicode"]["devanagari"]
            repl = result["unicode"]["replacement"]
            print(f"  Test {label}: lang={result['language_code']} speakers={result['speakers']} "
                  f"te={te} deva={deva} repl={repl} latency={result['latency_ms']}ms")
        else:
            print(f"  Test {label}: FAILED")

    print("\nDONE.")


if __name__ == "__main__":
    main()
