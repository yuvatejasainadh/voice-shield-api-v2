# Dataset Audit Report

## Summary

| Metric | Value |
|--------|-------|
| Total files | 2007 |
| Genuine files | 1007 |
| Spoof files | 1000 |
| Corrupted files | 0 |
| All readable | Yes |
| Sample rate | 16000 Hz (100%) |
| Channels | Mono (100%) |

## Duration Statistics

| Stat | Genuine | Spoof | All |
|------|---------|-------|-----|
| Min | 4.00 s | 4.01 s | 4.00 s |
| Max | 10.10 s | 10.09 s | 10.10 s |
| Mean | 7.22 s | 7.14 s | 7.18 s |
| Median | 7.23 s | 7.17 s | 7.19 s |
| P5 | 4.43 s | 4.43 s | 4.43 s |
| P95 | 9.83 s | 9.72 s | 9.80 s |

## Class Balance

- Genuine: 1007 files (50.2%)
- Spoof: 1000 files (49.8%)
- **Well balanced** — no action required.

## Duplicate Files (7 hash-groups found)

All duplicates are in the `genuine/` folder and are Windows "Copy" duplicates (e.g. `LJ001-0004 - Copy.wav` vs `LJ001-0004.wav`). These should be excluded from benchmarking to avoid inflating genuine accuracy. They are noted but **not deleted** from disk.

Duplicate pairs:
- LJ001-0004 - Copy.wav ↔ LJ001-0004.wav
- LJ001-0024 - Copy.wav ↔ LJ001-0024.wav
- LJ001-0075 - Copy.wav ↔ LJ001-0075.wav
- LJ002-0269 - Copy.wav ↔ LJ002-0269.wav
- LJ003-0191 - Copy.wav ↔ LJ003-0191.wav
- LJ003-0205 - Copy.wav ↔ LJ003-0205.wav
- LJ003-0264 - Copy.wav ↔ LJ003-0264.wav

> **Recommendation:** Exclude the `* - Copy.wav` files from benchmark evaluation. This reduces genuine count to **1000**, giving a perfectly balanced 1000/1000 dataset.

## Audio Properties

- **Format:** WAV, 16-bit PCM
- **Sample rate:** 16,000 Hz (all files — matches WavLM expected input exactly)
- **Channels:** Mono (all files)
- **Duration range:** 4–10.1 seconds

## Observations

- Dataset is very clean: 0 corrupted files.
- All files have uniform preprocessing (16kHz mono), which is ideal.
- Duration range 4–10s is consistent with ASVspoof-style datasets.
- 7 genuine duplicate pairs exist (Windows file copies) — excluded from benchmarks.
- **Effective dataset after deduplication:** 1000 genuine + 1000 spoof = 2000 files.
