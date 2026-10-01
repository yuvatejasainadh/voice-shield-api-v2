from __future__ import annotations

import csv
import json
import statistics
from dataclasses import asdict, dataclass
from pathlib import Path


VALID_AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".ogg"}


@dataclass
class BenchmarkSampleResult:
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
    segments: int


@dataclass
class BenchmarkSummary:
    model: str
    status: str
    accuracy: float | None = None
    genuine_accuracy: float | None = None
    spoof_accuracy: float | None = None
    false_positive_rate: float | None = None
    false_negative_rate: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    average_latency_ms: float | None = None
    median_latency_ms: float | None = None
    average_genuine_spoof_probability: float | None = None
    average_spoof_spoof_probability: float | None = None
    threshold_metrics: dict[str, dict[str, float]] | None = None
    load_time_ms: float | None = None
    message: str | None = None
    sample_count: int = 0
    genuine_count: int = 0
    spoof_count: int = 0


def discover_audio_files(root: Path) -> dict[str, list[Path]]:
    dataset_dir = root / "testvoices"
    if not dataset_dir.exists():
        return {"genuine": [], "spoof": []}

    genuine_files = sorted(
        p for p in (dataset_dir / "genuine").glob("*") if p.is_file() and p.suffix.lower() in VALID_AUDIO_EXTENSIONS
    )
    spoof_files = sorted(
        p for p in (dataset_dir / "spoof").glob("*") if p.is_file() and p.suffix.lower() in VALID_AUDIO_EXTENSIONS
    )
    return {"genuine": genuine_files, "spoof": spoof_files}


def classify_probability(spoof_probability: float, threshold: float = 0.5) -> str:
    return "SPOOF" if spoof_probability >= threshold else "GENUINE"


def compute_threshold_metrics(results: list[BenchmarkSampleResult], thresholds: list[float]) -> dict[str, dict[str, float]]:
    metrics_by_threshold: dict[str, dict[str, float]] = {}
    for threshold in thresholds:
        predicted = []
        for item in results:
            predicted.append(
                {
                    "ground_truth": item.ground_truth,
                    "predicted": classify_probability(item.spoof_probability, threshold),
                }
            )

        tp = sum(1 for row in predicted if row["ground_truth"] == "SPOOF" and row["predicted"] == "SPOOF")
        fp = sum(1 for row in predicted if row["ground_truth"] == "GENUINE" and row["predicted"] == "SPOOF")
        tn = sum(1 for row in predicted if row["ground_truth"] == "GENUINE" and row["predicted"] == "GENUINE")
        fn = sum(1 for row in predicted if row["ground_truth"] == "SPOOF" and row["predicted"] == "GENUINE")

        genuine_total = sum(1 for row in predicted if row["ground_truth"] == "GENUINE")
        spoof_total = sum(1 for row in predicted if row["ground_truth"] == "SPOOF")

        accuracy = (tp + tn) / len(predicted) if predicted else 0.0
        genuine_accuracy = (tn / genuine_total) if genuine_total else 0.0
        spoof_accuracy = (tp / spoof_total) if spoof_total else 0.0
        false_positive_rate = fp / genuine_total if genuine_total else 0.0
        false_negative_rate = fn / spoof_total if spoof_total else 0.0
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        metrics_by_threshold[str(threshold)] = {
            "accuracy": accuracy,
            "genuine_accuracy": genuine_accuracy,
            "spoof_accuracy": spoof_accuracy,
            "false_positive_rate": false_positive_rate,
            "false_negative_rate": false_negative_rate,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return metrics_by_threshold


def compute_metrics(results: list[BenchmarkSampleResult]) -> dict[str, float | None]:
    total = len(results)
    if total == 0:
        return {
            "accuracy": 0.0,
            "genuine_accuracy": 0.0,
            "spoof_accuracy": 0.0,
            "false_positive_rate": 0.0,
            "false_negative_rate": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "average_latency_ms": 0.0,
            "median_latency_ms": 0.0,
            "average_genuine_spoof_probability": 0.0,
            "average_spoof_spoof_probability": 0.0,
        }

    correct = sum(1 for item in results if item.correct)
    genuine = [item for item in results if item.ground_truth == "GENUINE"]
    spoof = [item for item in results if item.ground_truth == "SPOOF"]

    genuine_correct = sum(1 for item in genuine if item.predicted_class == "GENUINE")
    spoof_correct = sum(1 for item in spoof if item.predicted_class == "SPOOF")

    genuine_total = len(genuine)
    spoof_total = len(spoof)
    fp = sum(1 for item in genuine if item.predicted_class == "SPOOF")
    fn = sum(1 for item in spoof if item.predicted_class == "GENUINE")
    tp = spoof_correct
    tn = genuine_correct

    accuracy = correct / total if total else 0.0
    genuine_accuracy = genuine_correct / genuine_total if genuine_total else 0.0
    spoof_accuracy = spoof_correct / spoof_total if spoof_total else 0.0
    false_positive_rate = fp / genuine_total if genuine_total else 0.0
    false_negative_rate = fn / spoof_total if spoof_total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    avg_latency = sum(item.latency_ms for item in results) / total if total else 0.0
    median_latency = statistics.median(item.latency_ms for item in results) if results else 0.0

    avg_genuine_spoof_probability = (
        sum(item.spoof_probability for item in genuine) / genuine_total if genuine_total else 0.0
    )
    avg_spoof_spoof_probability = (
        sum(item.spoof_probability for item in spoof) / spoof_total if spoof_total else 0.0
    )

    return {
        "accuracy": accuracy,
        "genuine_accuracy": genuine_accuracy,
        "spoof_accuracy": spoof_accuracy,
        "false_positive_rate": false_positive_rate,
        "false_negative_rate": false_negative_rate,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "average_latency_ms": avg_latency,
        "median_latency_ms": median_latency,
        "average_genuine_spoof_probability": avg_genuine_spoof_probability,
        "average_spoof_spoof_probability": avg_spoof_spoof_probability,
    }


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text(
            "model,filename,ground_truth,predicted_class,correct,spoof_probability,bonafide_probability,confidence,latency_ms,sample_rate,duration_seconds,segments\n",
            encoding="utf-8",
        )
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
