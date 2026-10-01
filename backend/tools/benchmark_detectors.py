from __future__ import annotations

import argparse
import csv
import html
import json
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.ml.benchmarking import BenchmarkSampleResult, BenchmarkSummary, classify_probability, compute_metrics, compute_threshold_metrics, discover_audio_files
from app.ml.model_registry import get_model_spec, repo_exists
from app.ml.spoof_detector import SpoofDetector

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "benchmark_results"
DATASET_ROOT = ROOT / "testvoices"
THRESHOLDS = [0.30, 0.50, 0.70]


@dataclass
class BenchmarkFileRecord:
    model: str
    filename: str
    ground_truth: str
    predicted_class: str
    correct: bool
    spoof_probability: float
    bonafide_probability: float
    confidence: float
    latency_ms: float
    sample_rate: int
    duration_seconds: float
    segments_analyzed: int
    near_silence: bool
    low_energy: bool
    invalid: bool
    too_short: bool
    file_format: str


def _safe_float(value: float | None, default: float = 0.0) -> float:
    return float(value) if value is not None else default


def _dataset_metadata() -> dict:
    genuine_files = sorted((DATASET_ROOT / "genuine").glob("*.wav"))
    spoof_files = sorted((DATASET_ROOT / "spoof").glob("*.wav"))
    metadata = {
        "dataset_path": str(DATASET_ROOT),
        "genuine_count": len(genuine_files),
        "spoof_count": len(spoof_files),
        "total_count": len(genuine_files) + len(spoof_files),
        "audio_format": "wav",
        "sample_rates": [],
        "channels": [],
        "durations": [],
        "readable": [],
        "files": [],
    }
    for label in ("genuine", "spoof"):
        for file_path in sorted((DATASET_ROOT / label).glob("*.wav")):
            try:
                data, sr = sf.read(file_path)
                channels = 1 if data.ndim == 1 else int(data.shape[-1])
                duration = float(len(data)) / sr if sr else 0.0
                readable = True
            except Exception as exc:  # pragma: no cover - dataset-specific failure path
                data = None
                sr = None
                channels = None
                duration = 0.0
                readable = False
            metadata["files"].append({
                "label": label,
                "filename": file_path.name,
                "path": str(file_path),
                "sample_rate": sr,
                "channels": channels,
                "duration_seconds": duration,
                "readable": readable,
                "error": str(exc) if not readable else None,
            })
            if sr is not None:
                metadata["sample_rates"].append(sr)
            if channels is not None:
                metadata["channels"].append(channels)
            if readable:
                metadata["readable"].append(True)
            metadata["durations"].append(duration)
    metadata["sample_rate_summary"] = {
        "min": min(metadata["sample_rates"]) if metadata["sample_rates"] else None,
        "max": max(metadata["sample_rates"]) if metadata["sample_rates"] else None,
        "unique": sorted(set(metadata["sample_rates"])),
        "all_same": len(set(metadata["sample_rates"])) == 1,
    }
    return metadata


def _file_quality_flags(audio: np.ndarray, sample_rate: int, duration_seconds: float) -> tuple[bool, bool, bool, bool, bool]:
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    near_silence = rms < 0.005
    low_energy = rms < 0.05
    invalid = not np.isfinite(audio).all() or audio.size == 0
    too_short = duration_seconds < 0.3
    return near_silence, low_energy, invalid, too_short, bool(invalid or near_silence or too_short or low_energy)


def _run_ast_benchmark(model_name: str, dataset: dict[str, list[Path]]) -> tuple[list[dict], dict, dict, str]:
    detector = SpoofDetector()
    start = time.perf_counter()
    detector.load()
    load_time_ms = (time.perf_counter() - start) * 1000.0

    rows: list[dict] = []
    results: list[BenchmarkSampleResult] = []
    summary_payload: dict[str, float | None] = {}

    for ground_truth_key in ("genuine", "spoof"):
        files = dataset.get(ground_truth_key, [])
        for file_path in files:
            start_infer = time.perf_counter()
            audio, sr = librosa.load(str(file_path), sr=None, mono=False)
            mono = np.mean(audio, axis=0) if audio.ndim > 1 else audio
            if sr != 16000:
                mono = librosa.resample(mono, orig_sr=sr, target_sr=16000)
            duration_seconds = float(len(mono)) / 16000.0 if 16000 else 0.0
            near_silence, low_energy, invalid, too_short, _ = _file_quality_flags(mono.astype(np.float32), 16000, duration_seconds)

            output = detector.predict(mono.astype(np.float32), 16000)
            elapsed_ms = (time.perf_counter() - start_infer) * 1000.0
            spoof_probability = float(output["spoof_probability"])
            bonafide_probability = float(output["bonafide_probability"])
            confidence = float(output["confidence"])
            predicted_class = classify_probability(spoof_probability, 0.5)
            correct = predicted_class == ("SPOOF" if ground_truth_key == "spoof" else "GENUINE")

            record = BenchmarkFileRecord(
                model=model_name,
                filename=file_path.name,
                ground_truth=("SPOOF" if ground_truth_key == "spoof" else "GENUINE"),
                predicted_class=predicted_class,
                correct=correct,
                spoof_probability=spoof_probability,
                bonafide_probability=bonafide_probability,
                confidence=confidence,
                latency_ms=elapsed_ms,
                sample_rate=int(sr),
                duration_seconds=duration_seconds,
                segments_analyzed=1,
                near_silence=near_silence,
                low_energy=low_energy,
                invalid=invalid,
                too_short=too_short,
                file_format="wav",
            )
            row = asdict(record)
            rows.append(row)
            results.append(
                BenchmarkSampleResult(
                    model=model_name,
                    filename=file_path.name,
                    ground_truth=record.ground_truth,
                    predicted_class=record.predicted_class,
                    correct=record.correct,
                    spoof_probability=record.spoof_probability,
                    bonafide_probability=record.bonafide_probability,
                    confidence=record.confidence,
                    latency_ms=record.latency_ms,
                    sample_rate=record.sample_rate,
                    duration_seconds=record.duration_seconds,
                    segments=record.segments_analyzed,
                )
            )

    metrics = compute_metrics(results)
    threshold_metrics = compute_threshold_metrics(results, THRESHOLDS)
    avg_genuine = _safe_float(metrics.get("average_genuine_spoof_probability"))
    avg_spoof = _safe_float(metrics.get("average_spoof_spoof_probability"))
    summary_payload = {
        "model": model_name,
        "status": "PASS" if results else "EMPTY_DATASET",
        "repository": "MattyB95/AST-ASVspoof2019-Synthetic-Voice-Detection",
        "device": detector.device,
        "sample_rate": 16000,
        "number_of_files": len(results),
        "successful_files": len(results),
        "failed_files": 0,
        "accuracy": _safe_float(metrics.get("accuracy")),
        "precision": _safe_float(metrics.get("precision")),
        "recall": _safe_float(metrics.get("recall")),
        "f1": _safe_float(metrics.get("f1")),
        "genuine_accuracy": _safe_float(metrics.get("genuine_accuracy")),
        "spoof_accuracy": _safe_float(metrics.get("spoof_accuracy")),
        "false_positive_rate": _safe_float(metrics.get("false_positive_rate")),
        "false_negative_rate": _safe_float(metrics.get("false_negative_rate")),
        "average_latency_ms": _safe_float(metrics.get("average_latency_ms")),
        "median_latency_ms": _safe_float(metrics.get("median_latency_ms")),
        "min_latency_ms": min(item.latency_ms for item in results) if results else 0.0,
        "max_latency_ms": max(item.latency_ms for item in results) if results else 0.0,
        "p95_latency_ms": float(np.percentile([item.latency_ms for item in results], 95)) if len(results) >= 2 else (_safe_float(stats_median([item.latency_ms for item in results])) if results else 0.0),
        "load_time_ms": load_time_ms,
        "average_genuine_spoof_probability": avg_genuine,
        "average_spoof_spoof_probability": avg_spoof,
        "score_separation_difference": avg_spoof - avg_genuine,
        "threshold_metrics": threshold_metrics,
        "message": "AST baseline benchmark completed on real dataset",
    }
    return rows, summary_payload, threshold_metrics, detector.device


def stats_median(values: list[float]) -> float:
    return float(statistics.median(values)) if values else 0.0


def _failed_model_summary(model_key: str, dataset: dict[str, list[Path]], reason: str) -> dict:
    summary = {
        "model": model_key,
        "status": "FAILED",
        "repository": get_model_spec(model_key).repo_id,
        "device": "cpu",
        "sample_rate": 16000,
        "number_of_files": len(dataset.get("genuine", [])) + len(dataset.get("spoof", [])),
        "successful_files": 0,
        "failed_files": len(dataset.get("genuine", [])) + len(dataset.get("spoof", [])),
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "genuine_accuracy": None,
        "spoof_accuracy": None,
        "false_positive_rate": None,
        "false_negative_rate": None,
        "average_latency_ms": None,
        "median_latency_ms": None,
        "load_time_ms": None,
        "average_genuine_spoof_probability": None,
        "average_spoof_spoof_probability": None,
        "score_separation_difference": None,
        "threshold_metrics": None,
        "message": reason,
    }
    return summary


def _build_markdown_report(dataset_info: dict, model_summaries: list[dict], rows_by_model: dict[str, list[dict]]) -> str:
    genuine_count = dataset_info["genuine_count"]
    spoof_count = dataset_info["spoof_count"]
    total_count = dataset_info["total_count"]
    lines = [
        "# Voice Anti-Spoofing Benchmark",
        "",
        "## Dataset",
        "",
        f"- Genuine samples: {genuine_count}",
        f"- Spoof samples: {spoof_count}",
        f"- Total: {total_count}",
        f"- Dataset path: {dataset_info['dataset_path']}",
        f"- Audio format: {dataset_info['audio_format']}",
        f"- Sample-rate information: {dataset_info['sample_rate_summary']}",
        "",
        "## Models Tested",
        "",
        "| Model | Repository | Status | Accuracy | Genuine Acc | Spoof Acc | FPR | FNR | Precision | Recall | F1 | Avg Latency | Median Latency | Load Time |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for model in model_summaries:
        acc = model.get("accuracy") if model.get("accuracy") is not None else 0.0
        genuine_acc = model.get("genuine_accuracy") if model.get("genuine_accuracy") is not None else 0.0
        spoof_acc = model.get("spoof_accuracy") if model.get("spoof_accuracy") is not None else 0.0
        fpr = model.get("false_positive_rate") if model.get("false_positive_rate") is not None else 0.0
        fnr = model.get("false_negative_rate") if model.get("false_negative_rate") is not None else 0.0
        precision = model.get("precision") if model.get("precision") is not None else 0.0
        recall = model.get("recall") if model.get("recall") is not None else 0.0
        f1 = model.get("f1") if model.get("f1") is not None else 0.0
        avg_lat = model.get("average_latency_ms") if model.get("average_latency_ms") is not None else 0.0
        med_lat = model.get("median_latency_ms") if model.get("median_latency_ms") is not None else 0.0
        load_time = model.get("load_time_ms") if model.get("load_time_ms") is not None else 0.0
        lines.append(f"| {model['model']} | {model['repository']} | {model['status']} | {acc:.4f} | {genuine_acc:.4f} | {spoof_acc:.4f} | {fpr:.4f} | {fnr:.4f} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {avg_lat:.2f} | {med_lat:.2f} | {load_time:.2f} |")
    lines.append("")

    for model in model_summaries:
        lines.append(f"## Detailed Per-Model Results: {model['model']}")
        lines.append("")
        rows = rows_by_model.get(model["model"], [])
        if not rows:
            lines.append(f"Model {model['model']} produced no benchmark rows because it failed to load or was unavailable.")
            lines.append("")
            continue
        lines.append("| File | Ground Truth | Prediction | Correct | Spoof Probability | Bonafide Probability | Confidence | Latency |")
        lines.append("| --- | --- | --- | --- | ---: | ---: | ---: | ---: |")
        for row in rows:
            lines.append(f"| {row['filename']} | {row['ground_truth']} | {row['predicted_class']} | {'Yes' if row['correct'] else 'No'} | {row['spoof_probability']:.4f} | {row['bonafide_probability']:.4f} | {row['confidence']:.4f} | {row['latency_ms']:.2f} |")
        lines.append("")

    lines.append("## Score Separation")
    lines.append("")
    for model in model_summaries:
        if model.get("average_genuine_spoof_probability") is None:
            continue
        lines.append(f"- {model['model']}: average genuine spoof probability = {model['average_genuine_spoof_probability']:.4f}; average spoof spoof probability = {model['average_spoof_spoof_probability']:.4f}; difference = {model['score_separation_difference']:.4f}")
    lines.append("")

    lines.append("## Threshold Comparison")
    lines.append("")
    lines.append("| Model | Threshold | Accuracy | FPR | FNR | Precision | Recall | F1 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for model in model_summaries:
        if not model.get("threshold_metrics"):
            continue
        for threshold, metrics in model["threshold_metrics"].items():
            lines.append(f"| {model['model']} | {threshold} | {metrics['accuracy']:.4f} | {metrics['false_positive_rate']:.4f} | {metrics['false_negative_rate']:.4f} | {metrics['precision']:.4f} | {metrics['recall']:.4f} | {metrics['f1']:.4f} |")
    lines.append("")

    lines.append("## Latency Comparison")
    lines.append("")
    lines.append("| Model | Average Latency | Median Latency | Load Time |")
    lines.append("| --- | ---: | ---: | ---: |")
    for model in model_summaries:
        avg_lat = model.get("average_latency_ms") if model.get("average_latency_ms") is not None else 0.0
        med_lat = model.get("median_latency_ms") if model.get("median_latency_ms") is not None else 0.0
        load_time = model.get("load_time_ms") if model.get("load_time_ms") is not None else 0.0
        lines.append(f"| {model['model']} | {avg_lat:.2f} | {med_lat:.2f} | {load_time:.2f} |")
    lines.append("")

    lines.append("## Failure Report")
    lines.append("")
    for model in model_summaries:
        if model["status"] == "FAILED":
            lines.append(f"- {model['model']}: {model['message']}")
    lines.append("")

    lines.append("## Model Ranking")
    lines.append("")
    lines.append("### Detection performance")
    ranked = sorted([m for m in model_summaries if m.get("f1") is not None], key=lambda m: (-m["f1"], -m["accuracy"], -m["genuine_accuracy"], -m["spoof_accuracy"]))
    for idx, model in enumerate(ranked, start=1):
        lines.append(f"{idx}. {model['model']} - F1 {model['f1']:.4f}, accuracy {model['accuracy']:.4f}")
    lines.append("")
    lines.append("### Practical performance")
    ranked2 = sorted([m for m in model_summaries if m.get("f1") is not None], key=lambda m: (-m["f1"], -(1.0 - (m['false_positive_rate'] or 0.0)) if m.get('false_positive_rate') is not None else 0.0, -(1.0 - (m['false_negative_rate'] or 0.0)) if m.get('false_negative_rate') is not None else 0.0, (m['average_latency_ms'] or 0.0), (m['load_time_ms'] or 0.0)))
    for idx, model in enumerate(ranked2, start=1):
        lines.append(f"{idx}. {model['model']} - F1 {model['f1']:.4f}, FPR {model['false_positive_rate']:.4f}, FNR {model['false_negative_rate']:.4f}, avg latency {model['average_latency_ms']:.2f} ms")
    lines.append("")

    lines.append("## Recommendation")
    lines.append("")
    if ranked:
        best = ranked[0]
        lines.append(f"- Best overall model: {best['model']} (F1 {best['f1']:.4f}, accuracy {best['accuracy']:.4f})")
    else:
        lines.append("- Best overall model: no successfully tested model")
    lines.append("- This benchmark uses a 20-sample prototype dataset and is not statistically reliable for production claims.")
    lines.append("- Production thresholds and detector selection were not changed.")
    return "\n".join(lines)


def _build_html_report(dataset_info: dict, model_summaries: list[dict], rows_by_model: dict[str, list[dict]]) -> str:
    body = [
        "<html><head><meta charset='utf-8'><title>Voice Anti-Spoofing Benchmark</title>",
        "<style>body{font-family:Arial,sans-serif;padding:20px;}table{border-collapse:collapse;width:100%;margin-bottom:20px;}th,td{border:1px solid #ddd;padding:8px;text-align:left;}th{background:#f2f2f2;}caption{font-weight:bold;margin-bottom:8px;}</style>",
        "</head><body>",
        "<h1>Voice Anti-Spoofing Benchmark</h1>",
        f"<p><strong>Dataset:</strong> {dataset_info['genuine_count']} genuine + {dataset_info['spoof_count']} spoof = {dataset_info['total_count']} files</p>",
        f"<p><strong>Dataset path:</strong> {html.escape(dataset_info['dataset_path'])}</p>",
        "<h2>Models Tested</h2>",
        "<table><tr><th>Model</th><th>Repository</th><th>Status</th><th>Accuracy</th><th>Genuine Acc</th><th>Spoof Acc</th><th>FPR</th><th>FNR</th><th>Precision</th><th>Recall</th><th>F1</th><th>Avg Latency</th><th>Median Latency</th><th>Load Time</th></tr>",
    ]
    for model in model_summaries:
        acc = model.get("accuracy") if model.get("accuracy") is not None else 0.0
        gacc = model.get("genuine_accuracy") if model.get("genuine_accuracy") is not None else 0.0
        sacc = model.get("spoof_accuracy") if model.get("spoof_accuracy") is not None else 0.0
        fpr = model.get("false_positive_rate") if model.get("false_positive_rate") is not None else 0.0
        fnr = model.get("false_negative_rate") if model.get("false_negative_rate") is not None else 0.0
        precision = model.get("precision") if model.get("precision") is not None else 0.0
        recall = model.get("recall") if model.get("recall") is not None else 0.0
        f1 = model.get("f1") if model.get("f1") is not None else 0.0
        avg_lat = model.get("average_latency_ms") if model.get("average_latency_ms") is not None else 0.0
        med_lat = model.get("median_latency_ms") if model.get("median_latency_ms") is not None else 0.0
        load_time = model.get("load_time_ms") if model.get("load_time_ms") is not None else 0.0
        body.append(f"<tr><td>{model['model']}</td><td>{html.escape(model['repository'])}</td><td>{model['status']}</td><td>{acc:.4f}</td><td>{gacc:.4f}</td><td>{sacc:.4f}</td><td>{fpr:.4f}</td><td>{fnr:.4f}</td><td>{precision:.4f}</td><td>{recall:.4f}</td><td>{f1:.4f}</td><td>{avg_lat:.2f}</td><td>{med_lat:.2f}</td><td>{load_time:.2f}</td></tr>")
    body.append("</table>")

    for model in model_summaries:
        rows = rows_by_model.get(model["model"], [])
        body.append(f"<h2>{html.escape(model['model'])} per-file results</h2>")
        if not rows:
            body.append("<p>No rows available because the model did not load or was unavailable.</p>")
            continue
        body.append("<table><tr><th>File</th><th>Ground Truth</th><th>Prediction</th><th>Correct</th><th>Spoof Prob</th><th>Bonafide Prob</th><th>Confidence</th><th>Latency</th></tr>")
        for row in rows:
            body.append(f"<tr><td>{html.escape(row['filename'])}</td><td>{row['ground_truth']}</td><td>{row['predicted_class']}</td><td>{'Yes' if row['correct'] else 'No'}</td><td>{row['spoof_probability']:.4f}</td><td>{row['bonafide_probability']:.4f}</td><td>{row['confidence']:.4f}</td><td>{row['latency_ms']:.2f}</td></tr>")
        body.append("</table>")
    body.append("</body></html>")
    return "".join(body)


def _write_csv(rows: list[dict], output_path: Path) -> None:
    if not rows:
        output_path.write_text("model,filename,ground_truth,predicted_class,correct,spoof_probability,bonafide_probability,confidence,latency_ms,sample_rate,duration_seconds,segments_analyzed,near_silence,low_energy,invalid,too_short,file_format\n", encoding="utf-8")
        return
    fieldnames = [
        "model",
        "filename",
        "ground_truth",
        "predicted_class",
        "correct",
        "spoof_probability",
        "bonafide_probability",
        "confidence",
        "latency_ms",
        "sample_rate",
        "duration_seconds",
        "segments_analyzed",
        "near_silence",
        "low_energy",
        "invalid",
        "too_short",
        "file_format",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark all configured anti-spoofing models against the repository testvoice set.")
    parser.add_argument("--model", choices=["ast", "w2v2-aasist", "spectra-aasist3", "pellav2"], default=None)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    dataset = discover_audio_files(ROOT)
    metadata = _dataset_metadata()
    model_keys = [args.model] if args.model else ["ast", "w2v2-aasist", "spectra-aasist3", "pellav2"]

    rows_by_model: dict[str, list[dict]] = {}
    model_summaries: list[dict] = []

    for model_key in model_keys:
        if model_key == "ast":
            rows, summary, threshold_metrics, device = _run_ast_benchmark(model_key, dataset)
            rows_by_model[model_key] = rows
            model_summaries.append({**summary, "threshold_metrics": threshold_metrics})
            continue

        spec = get_model_spec(model_key)
        if not repo_exists(spec.repo_id):
            reason = f"Repository unavailable or invalid in registry: {spec.repo_id}"
            rows_by_model[model_key] = []
            model_summaries.append(_failed_model_summary(model_key, dataset, reason))
            continue

        rows_by_model[model_key] = []
        model_summaries.append({
            "model": model_key,
            "status": "NOT_IMPLEMENTED",
            "repository": spec.repo_id,
            "device": "cpu",
            "sample_rate": 16000,
            "number_of_files": len(dataset.get("genuine", [])) + len(dataset.get("spoof", [])),
            "successful_files": 0,
            "failed_files": len(dataset.get("genuine", [])) + len(dataset.get("spoof", [])),
            "accuracy": None,
            "precision": None,
            "recall": None,
            "f1": None,
            "genuine_accuracy": None,
            "spoof_accuracy": None,
            "false_positive_rate": None,
            "false_negative_rate": None,
            "average_latency_ms": None,
            "median_latency_ms": None,
            "load_time_ms": None,
            "average_genuine_spoof_probability": None,
            "average_spoof_spoof_probability": None,
            "score_separation_difference": None,
            "threshold_metrics": None,
            "message": "Benchmark adapter not implemented for registry model; repo exists but no loader was configured.",
        })

    all_rows = [row for rows in rows_by_model.values() for row in rows]
    _write_csv(all_rows, RESULTS_DIR / "benchmark_results.csv")
    (RESULTS_DIR / "benchmark_results.json").write_text(json.dumps({"dataset": metadata, "models": model_summaries, "rows": all_rows}, indent=2), encoding="utf-8")
    (RESULTS_DIR / "benchmark_summary.json").write_text(json.dumps({"dataset": metadata, "models": model_summaries}, indent=2), encoding="utf-8")
    markdown = _build_markdown_report(metadata, model_summaries, rows_by_model)
    (RESULTS_DIR / "benchmark_report.md").write_text(markdown, encoding="utf-8")
    (RESULTS_DIR / "benchmark_report.html").write_text(_build_html_report(metadata, model_summaries, rows_by_model), encoding="utf-8")

    print(json.dumps({"dataset": metadata, "models": model_summaries}, indent=2))


if __name__ == "__main__":
    main()
