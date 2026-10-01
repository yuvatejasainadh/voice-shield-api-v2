"""
AssemblyAI Tests C, D, E — Tests remaining after A/B results.
Tests universal-2 for Telugu script preservation.
Run: .venv\Scripts\python.exe scratch/run_tests_cde.py
"""
import os, asyncio, time, sys
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()
import assemblyai as aai

api_key = os.getenv("ASSEMBLYAI_API_KEY", "").strip()
backend_dir = Path(__file__).resolve().parents[1]
storage = backend_dir / "storage"
wav_files = sorted(storage.glob("*.wav"), key=lambda f: f.stat().st_size, reverse=True)
telugu_path = wav_files[0]
english_path = backend_dir / "testvoices" / "genuine" / "LJ001-0004.wav"

PRESERVATION_PROMPT = (
    "Transcribe the conversation exactly as spoken. "
    "Preserve the original language and script of every word. "
    "Do not translate Telugu or transliterate it into Devanagari or Latin script. "
    "Preserve Telugu-English code-switching as spoken."
)


def unicode_diagnostics(text: str) -> dict:
    counts = {"te": 0, "deva": 0, "lat": 0, "repl": 0, "total": 0}
    for ch in text:
        cp = ord(ch)
        counts["total"] += 1
        if 0x0C00 <= cp <= 0x0C7F:
            counts["te"] += 1
        elif 0x0900 <= cp <= 0x097F:
            counts["deva"] += 1
        elif (0x41 <= cp <= 0x5A) or (0x61 <= cp <= 0x7A):
            counts["lat"] += 1
        elif cp == 0xFFFD:
            counts["repl"] += 1
    return counts


async def run_test(label, speech_models, lang_code, lang_detect, prompt, audio_path):
    print(f"\n{'='*60}")
    print(f"TEST {label}")
    print(f"  speech_models={speech_models} lang_code={lang_code!r} lang_detect={lang_detect} prompt={'YES' if prompt else 'NO'}")
    print(f"{'='*60}")
    kwargs = {"speaker_labels": True, "speakers_expected": 2}
    if speech_models:
        kwargs["speech_models"] = speech_models
    if lang_detect is not None:
        kwargs["language_detection"] = lang_detect
    if prompt:
        kwargs["prompt"] = prompt
    config = aai.TranscriptionConfig(**kwargs)
    if lang_code:
        config.raw.language_code = lang_code
    try:
        t0 = time.perf_counter()
        async with aai.AsyncTranscriber(api_key=api_key, config=config) as tr:
            result = await tr.transcribe(str(audio_path))
        ms = int((time.perf_counter() - t0) * 1000)
    except Exception as exc:
        print(f"  SDK ERROR: {exc}")
        return None
    err = getattr(result, "error", None)
    if err:
        print(f"  PROVIDER ERROR: {err}")
        return None
    text = getattr(result, "text", "") or ""
    lang = getattr(result, "language_code", None)
    utts = getattr(result, "utterances", None) or []
    diag = unicode_diagnostics(text)
    speakers = sorted(set(getattr(u, "speaker", "?") for u in utts))
    print(f"  Latency:   {ms} ms")
    print(f"  Language:  {lang}")
    print(f"  Speakers:  {speakers}  ({len(utts)} utterances)")
    print(f"  Unicode:   te={diag['te']} deva={diag['deva']} lat={diag['lat']} repl={diag['repl']} total={diag['total']}")
    # Safe text display (unicode_escape avoids cp1252 errors on Windows terminal)
    safe = text[:300].encode("unicode_escape").decode("ascii")
    print(f"  Text (escape): {safe}")
    for i, u in enumerate(utts):
        spk = getattr(u, "speaker", "?")
        s = getattr(u, "start", 0) / 1000.0
        e = getattr(u, "end", 0) / 1000.0
        ud = unicode_diagnostics(getattr(u, "text", "") or "")
        utext_safe = (getattr(u, "text", "") or "")[:80].encode("unicode_escape").decode("ascii")
        print(f"    [{i}] spk={spk} {s:.3f}s-{e:.3f}s te={ud['te']} deva={ud['deva']} lat={ud['lat']}  {utext_safe}")
    if len(utts) > 1:
        for i in range(len(utts) - 1):
            gap = (getattr(utts[i + 1], "start", 0) - getattr(utts[i], "end", 0)) / 1000.0
            if gap > 1.0:
                s = getattr(utts[i], "end", 0) / 1000.0
                ns = getattr(utts[i + 1], "start", 0) / 1000.0
                print(f"  GAP: {s:.3f}s -> {ns:.3f}s  gap={gap:.3f}s")
    return {"lang": lang, "diag": diag, "speakers": speakers, "ms": ms, "utterances": len(utts)}


async def main():
    print("\n" + "=" * 70)
    print("ASSEMBLYAI TESTS C/D/E/F + ENGLISH")
    print("=" * 70)
    print(f"SDK: {aai.__version__}  Audio: {telugu_path.name}  Size: {telugu_path.stat().st_size // 1024}KB")

    # Test C: universal-2 + explicit te (most likely to preserve Telugu script)
    c = await run_test("C  universal-2 + language_code=te", ["universal-2"], "te", False, None, telugu_path)

    # Test D: [u35p, u2] + auto-detect + preservation prompt
    d = await run_test("D  [u-3-5-pro, u-2] + auto + prompt", ["universal-3-5-pro", "universal-2"], None, True, PRESERVATION_PROMPT, telugu_path)

    # Test E: universal-2 + auto-detect (no prompt)
    e = await run_test("E  universal-2 + auto-detect", ["universal-2"], None, True, None, telugu_path)

    # Test F: universal-2 + explicit te + preservation prompt
    f = await run_test("F  universal-2 + te + prompt", ["universal-2"], "te", False, PRESERVATION_PROMPT, telugu_path)

    # English sanity check
    if english_path.exists():
        en = await run_test("ENGLISH  u-3-5-pro + auto", ["universal-3-5-pro"], None, True, None, english_path)
    else:
        en = None
        print("  English audio not found - skipping")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    for lbl, r in [("A (prev)", {"lang": "ml", "diag": {"te": 0, "deva": 0, "lat": 288, "repl": 0}, "speakers": ["A","B"], "ms": 14832, "utterances": 7}),
                   ("B (prev)", {"lang": "te", "diag": {"te": 30, "deva": 2, "lat": 238, "repl": 0}, "speakers": ["A","B"], "ms": 20063, "utterances": 6}),
                   ("C", c), ("D", d), ("E", e), ("F", f)]:
        if r:
            d2 = r["diag"]
            print(f"  {lbl}: lang={r['lang']} spk={r['speakers']} utts={r['utterances']} te={d2['te']} deva={d2['deva']} lat={d2['lat']} ms={r['ms']}")
        else:
            print(f"  {lbl}: FAILED")


asyncio.run(main())
