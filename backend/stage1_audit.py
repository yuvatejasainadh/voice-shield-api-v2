"""
Stage 1: Dataset Audit
Inspects all WAV files in testvoices/genuine and testvoices/spoof.
"""
import json
import csv
import hashlib
import os
import sys
import warnings
from pathlib import Path
from collections import defaultdict

import librosa
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")

RESULTS_DIR = Path("benchmark_results")
RESULTS_DIR.mkdir(exist_ok=True)

def sha256_small(path, max_bytes=65536):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        h.update(f.read(max_bytes))
    return h.hexdigest()

def audit_files():
    entries = []
    errors = []
    hash_map = defaultdict(list)

    for label in ["genuine", "spoof"]:
        folder = Path("testvoices") / label
        wav_files = sorted(folder.glob("*.wav"))
        print(f"  Scanning {label}: {len(wav_files)} files...")
        for path in wav_files:
            entry = {
                "filename": path.name,
                "label": label,
                "path": str(path),
                "size_bytes": path.stat().st_size,
            }
            try:
                info = sf.info(str(path))
                entry["sample_rate"] = info.samplerate
                entry["channels"] = info.channels
                entry["duration_sec"] = info.duration
                entry["frames"] = info.frames
                entry["format"] = info.format
                entry["readable"] = True
                # Quick hash for duplicate detection (first 64KB only)
                h = sha256_small(path)
                hash_map[h].append(str(path))
            except Exception as e:
                entry["readable"] = False
                entry["error"] = str(e)
                errors.append({"path": str(path), "error": str(e)})
            entries.append(entry)

    return entries, errors, hash_map

def compute_stats(values):
    arr = np.array(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p5": float(np.percentile(arr, 5)),
        "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)),
        "p95": float(np.percentile(arr, 95)),
        "std": float(np.std(arr)),
    }

def main():
    print("Running dataset audit...")
    entries, errors, hash_map = audit_files()

    readable = [e for e in entries if e.get("readable")]
    corrupted = [e for e in entries if not e.get("readable")]

    genuine_entries = [e for e in readable if e["label"] == "genuine"]
    spoof_entries   = [e for e in readable if e["label"] == "spoof"]

    # Duration stats
    gen_dur  = [e["duration_sec"] for e in genuine_entries]
    spf_dur  = [e["duration_sec"] for e in spoof_entries]
    all_dur  = [e["duration_sec"] for e in readable]

    # Sample rate distribution
    sr_dist = defaultdict(int)
    for e in readable:
        sr_dist[e["sample_rate"]] += 1

    # Channel distribution
    ch_dist = defaultdict(int)
    for e in readable:
        ch_dist[e["channels"]] += 1

    # Duplicates (same hash)
    duplicates = {k: v for k, v in hash_map.items() if len(v) > 1}

    # File size stats
    sizes = [e["size_bytes"] for e in readable]

    audit = {
        "total_files": len(entries),
        "genuine_count": len(genuine_entries),
        "spoof_count": len(spoof_entries),
        "corrupted_count": len(corrupted),
        "corrupted_files": corrupted,
        "readable_count": len(readable),
        "sample_rate_distribution": {str(k): v for k, v in sr_dist.items()},
        "channel_distribution": {str(k): v for k, v in ch_dist.items()},
        "duration_stats_all": compute_stats(all_dur) if all_dur else {},
        "duration_stats_genuine": compute_stats(gen_dur) if gen_dur else {},
        "duration_stats_spoof": compute_stats(spf_dur) if spf_dur else {},
        "file_size_stats": compute_stats(sizes) if sizes else {},
        "duplicate_hash_groups": len(duplicates),
        "duplicate_details": list(duplicates.values())[:20],  # cap at 20
    }

    with open(RESULTS_DIR / "dataset_audit.json", "w") as f:
        json.dump(audit, f, indent=2)

    # CSV of all entries
    if readable:
        keys = ["filename", "label", "size_bytes", "sample_rate", "channels",
                "duration_sec", "readable"]
        with open(RESULTS_DIR / "dataset_audit_files.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(readable)

    # Print summary
    print(f"\nDATASET AUDIT SUMMARY")
    print(f"  Total files    : {audit['total_files']}")
    print(f"  Genuine        : {audit['genuine_count']}")
    print(f"  Spoof          : {audit['spoof_count']}")
    print(f"  Corrupted      : {audit['corrupted_count']}")
    print(f"  Sample rates   : {dict(sr_dist)}")
    print(f"  Channels       : {dict(ch_dist)}")
    if all_dur:
        ds = audit["duration_stats_all"]
        print(f"  Duration (all) : min={ds['min']:.2f}s  max={ds['max']:.2f}s  "
              f"mean={ds['mean']:.2f}s  median={ds['median']:.2f}s  "
              f"p5={ds['p5']:.2f}s  p95={ds['p95']:.2f}s")
    print(f"  Duplicate groups: {audit['duplicate_hash_groups']}")
    if errors:
        print(f"\n  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    {e['path']}: {e['error']}")

    return audit, readable

if __name__ == "__main__":
    main()
