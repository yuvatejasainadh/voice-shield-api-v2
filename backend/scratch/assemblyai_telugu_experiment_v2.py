#!/usr/bin/env python3
"""
AssemblyAI Corrected Experiment Script v2
Uses speech_models=[...] with the ACTUAL valid values from API error responses.

Valid speech_models values: "universal-3-5-pro", "universal-2", "universal-3-pro"
"""
from __future__ import annotations

import asyncio
import os
import sys
import time
import unicodedata
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import assemblyai as aai


def unicode_diagnostics(text: str) -> dict:
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


def log_utterances(utterances, label="Utterances"):
    if not utterances:
        print(f"  {label}: 0 utterances")
        return
    print(f"  {label}: {len(utterances)} total")
    for i, u in enumerate(utterances):
        spk = getattr(u, "speaker", "N/A")
        start = getattr(u, "start", 0) / 1000.0
        end = getattr(u, "end", 0) / 1000.0
        conf = getattr(u, "confidence", None)
        text = getattr(u, "text", "") or ""
        diag = unicode_diagnostics(text)
        print(f"    [{i}] spk={spk} {start:.3f}s-{end:.3f}s conf={conf}")
        print(f"         unicode: te={diag['telugu']} deva={diag['devanagari']} lat={diag['latin']} repl={diag['replacement']}")

    # Gap analysis
    if len(utterances) > 1:
        gaps = []
        for i in range(len(utterances) - 1):
            u1_end = getattr(utterances[i], "end", 0) / 1000.0
            u2_start = getattr(utterances[i + 1], "start", 0) / 1000.0
            gap = u2_start - u1_end
            if gap > 1.0:
                gaps.append((u1_end, u2_start, gap))
        if gaps:
            print("  GAPS > 1s:")
            for end, start, gap in gaps:
                print(f"    end={end:.3f}s -> next_start={start:.3f}s  gap={gap:.3f}s")


def run_test(label, speech_models_list, language_code_raw, language_detection,
             prompt_text, audio_path, api_key):
    print(f"\n{'='*60}")
    print(f"TEST {label}")
    cfg_info = f"speech_models={speech_models_list!r} lang={language_code_raw!r} lang_detect={language_detection}"
    if prompt_text:
        cfg_info += " prompt=YES"
    print(f"Config: {cfg_info}")
    print(f"{'='*60}")

    # Build config without deprecated speech_model enum
    kwargs = {
        "speaker_labels": True,
        "speakers_expected": 2,
    }
    if speech_models_list:
        kwargs["speech_models"] = speech_models_list
    if language_detection is not None:
        kwargs["language_detection"] = language_detection
    if prompt_text:
        kwargs["prompt"] = prompt_text

    config = aai.TranscriptionConfig(**kwargs)

    # Inject language_code as raw string (SDK enum lacks 'te')
    if language_code_raw:
        config.raw.language_code = language_code_raw

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

    error = getattr(transcript, "error", None)
    if error:
        print(f"  PROVIDER ERROR: {error}")
        return None

    text = getattr(transcript, "text", "") or ""
    language_code = getattr(transcript, "language_code", None)
    audio_duration = getattr(transcript, "audio_duration", None)
    utterances = getattr(transcript, "utterances", None) or []
    speakers_seen = sorted(set(getattr(u, "speaker", "?") for u in utterances))
    diag = unicode_diagnostics(text)

    print(f"  Latency:      {elapsed_ms} ms")
    print(f"  Language:     {language_code}")
    print(f"  Duration:     {audio_duration}s")
    print(f"  Speakers:     {speakers_seen}  ({len(utterances)} utterances)")
    print(f"  Unicode:      te={diag['telugu']} deva={diag['devanagari']} lat={diag['latin']} repl={diag['replacement']} total={diag['total']}")
    print(f"  RAW TEXT (first 300): {repr(text[:300])}")
    log_utterances(utterances)

    return {
        "language_code": language_code,
        "unicode": diag,
        "speakers": speakers_seen,
        "utterance_count": len(utterances),
        "latency_ms": elapsed_ms,
    }


def main():
    api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
    if not api_key:
        print("ERROR: ASSEMBLYAI_API_KEY not set")
        sys.exit(1)

    backend_dir = Path(__file__).resolve().parents[1]
    english_path = backend_dir / "testvoices" / "genuine" / "LJ001-0004.wav"
    storage = backend_dir / "storage"
    wav_files = sorted(storage.glob("*.wav"), key=lambda f: f.stat().st_size, reverse=True)
    telugu_path = wav_files[0] if wav_files else None

    print("\n" + "="*70)
    print("ASSEMBLYAI CORRECTED EXPERIMENTS v2")
    print("="*70)
    print(f"SDK VERSION: {aai.__version__}")
    print(f"Valid speech_models values: universal-3-5-pro, universal-2, universal-3-pro")
    print(f"Deprecated: SpeechModel enum (slam_1, universal, best, nano)")
    print(f"Telugu audio: {telugu_path}")

    PRESERVATION_PROMPT = (
        "Transcribe the conversation exactly as spoken. "
        "Preserve the original language and script of every word. "
        "Do not translate Telugu, Hindi, Tamil, Kannada, Malayalam, Bengali, or English. "
        "Do not transliterate Telugu into Devanagari or Latin script. "
        "Preserve Telugu-English code-switching as spoken."
    )

    # TEST A — universal-3-5-pro (latest), language_detection=True (no explicit lang)
    result_a = run_test(
        "A  universal-3-5-pro + language_detection=True",
        speech_models_list=["universal-3-5-pro"],
        language_code_raw=None,
        language_detection=True,
        prompt_text=None,
        audio_path=telugu_path,
        api_key=api_key,
    )

    # TEST B — universal-3-5-pro + explicit te
    result_b = run_test(
        "B  universal-3-5-pro + language_code=te",
        speech_models_list=["universal-3-5-pro"],
        language_code_raw="te",
        language_detection=False,
        prompt_text=None,
        audio_path=telugu_path,
        api_key=api_key,
    )

    # TEST C — universal-2 (broader multilingual) + explicit te
    result_c = run_test(
        "C  universal-2 + language_code=te",
        speech_models_list=["universal-2"],
        language_code_raw="te",
        language_detection=False,
        prompt_text=None,
        audio_path=telugu_path,
        api_key=api_key,
    )

    # TEST D — universal-3-5-pro fallback to universal-2 + language_detection + preservation prompt
    result_d = run_test(
        "D  [universal-3-5-pro, universal-2] + language_detection + preservation prompt",
        speech_models_list=["universal-3-5-pro", "universal-2"],
        language_code_raw=None,
        language_detection=True,
        prompt_text=PRESERVATION_PROMPT,
        audio_path=telugu_path,
        api_key=api_key,
    )

    # English sanity check
    if english_path.exists():
        result_en = run_test(
            "ENGLISH universal-3-5-pro",
            speech_models_list=["universal-3-5-pro"],
            language_code_raw=None,
            language_detection=True,
            prompt_text=None,
            audio_path=english_path,
            api_key=api_key,
        )

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    for label, result in [("A", result_a), ("B", result_b), ("C", result_c), ("D", result_d)]:
        if result:
            d = result["unicode"]
            print(f"  {label}: lang={result['language_code']} spk={result['speakers']} "
                  f"te={d['telugu']} deva={d['devanagari']} repl={d['replacement']} latency={result['latency_ms']}ms")
        else:
            print(f"  {label}: FAILED")

    print("\nDONE.")


if __name__ == "__main__":
    main()
