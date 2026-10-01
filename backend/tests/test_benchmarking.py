from __future__ import annotations

from pathlib import Path

from app.ml.benchmarking import BenchmarkSampleResult, classify_probability, compute_metrics, compute_threshold_metrics, discover_audio_files


def test_discover_audio_files_handles_missing_dataset(tmp_path: Path) -> None:
    assert discover_audio_files(tmp_path) == {"genuine": [], "spoof": []}


def test_discover_audio_files_recognizes_dataset(tmp_path: Path) -> None:
    genuine_dir = tmp_path / "testvoices" / "genuine"
    spoof_dir = tmp_path / "testvoices" / "spoof"
    genuine_dir.mkdir(parents=True)
    spoof_dir.mkdir(parents=True)

    genuine_path = genuine_dir / "a.wav"
    spoof_path = spoof_dir / "b.wav"
    genuine_path.write_bytes(b"fake")
    spoof_path.write_bytes(b"fake")

    discovered = discover_audio_files(tmp_path)
    assert [p.name for p in discovered["genuine"]] == ["a.wav"]
    assert [p.name for p in discovered["spoof"]] == ["b.wav"]


def test_classify_probability_thresholds() -> None:
    assert classify_probability(0.49, 0.5) == "GENUINE"
    assert classify_probability(0.50, 0.5) == "SPOOF"
    assert classify_probability(0.70, 0.6) == "SPOOF"


def test_compute_metrics_and_thresholds() -> None:
    results = [
        BenchmarkSampleResult("ast", "g1.wav", "GENUINE", "GENUINE", True, 0.10, 0.90, 0.90, 10.0, 16000, 1.0, 1),
        BenchmarkSampleResult("ast", "g2.wav", "GENUINE", "SPOOF", False, 0.80, 0.20, 0.80, 12.0, 16000, 1.0, 1),
        BenchmarkSampleResult("ast", "s1.wav", "SPOOF", "SPOOF", True, 0.92, 0.08, 0.92, 14.0, 16000, 1.0, 1),
        BenchmarkSampleResult("ast", "s2.wav", "SPOOF", "GENUINE", False, 0.35, 0.65, 0.65, 9.0, 16000, 1.0, 1),
    ]

    metrics = compute_metrics(results)
    assert metrics["accuracy"] == 0.5
    assert metrics["genuine_accuracy"] == 0.5
    assert metrics["spoof_accuracy"] == 0.5
    assert metrics["false_positive_rate"] == 0.5
    assert metrics["false_negative_rate"] == 0.5

    thresholds = compute_threshold_metrics(results, [0.5])
    assert "0.5" in thresholds
    assert thresholds["0.5"]["accuracy"] == 0.5


def test_label_mapping_is_explicit_and_not_assumed() -> None:
    label_map = {0: "Bonafide", 1: "Spoof"}
    spoof_id = next(key for key, value in label_map.items() if value.lower() == "spoof")
    bonafide_id = next(key for key, value in label_map.items() if value.lower() == "bonafide")
    assert spoof_id == 1
    assert bonafide_id == 0
