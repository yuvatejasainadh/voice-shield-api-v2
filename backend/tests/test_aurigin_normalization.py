from __future__ import annotations

import pytest
from app.services.aurigin_service import AuriginResultNormalizer


def test_normalize_bonafide():
    payload = {
        "prediction_id": "pred_1",
        "global": {"confidence": 0.9946, "result": "bonafide", "score": 0.0027, "reason": None},
        "segments": [{"start": 0.0, "end": 10.0, "confidence": 0.9946, "result": "bonafide"}],
    }
    result = AuriginResultNormalizer.normalize(payload, processing_ms=100)
    assert result.result == "REAL"
    assert result.score == 0.0027
    assert result.confidence == 0.9946
    assert result.risk_score == 0
    assert len(result.segments) == 1
    assert result.raw_result == "bonafide"


def test_normalize_spoofed(): 
    payload = {
        "prediction_id": "pred_2",
        "global": {"confidence": 0.98, "result": "spoofed", "score": 0.94, "reason": None},
        "segments": [{"start": 0.0, "end": 10.0, "confidence": 0.98, "result": "spoofed"}],
    }
    result = AuriginResultNormalizer.normalize(payload, processing_ms=120)
    assert result.result == "SPOOFED"
    assert result.score == 0.94
    assert result.confidence == 0.98
    assert result.risk_score == 94
    assert len(result.segments) == 1
    assert result.raw_result == "spoofed"


def test_normalize_partially_spoofed():
    payload = {
        "prediction_id": "pred_3",
        "global": {"confidence": 0.9916, "result": "partially_spoofed", "score": 0.3324, "reason": None},
        "segments": [
            {"start": 0.0, "end": 5.0, "confidence": 0.98, "result": "bonafide"},
            {"start": 5.0, "end": 10.0, "confidence": 0.97, "result": "spoofed"},
        ],
    }
    result = AuriginResultNormalizer.normalize(payload, processing_ms=150)
    assert result.result == "SPOOFED"
    assert result.score == 0.3324
    assert result.confidence == 0.9916
    assert result.risk_score == 33
    assert len(result.segments) == 2
    assert result.raw_result == "partially_spoofed"


def test_normalize_unknown_poor_audio():
    payload = {
        "prediction_id": "pred_4",
        "global": {"confidence": 0.0, "result": None, "reason": "Unable to analyze - audio quality too poor", "score": 0.0},
        "segments": [],
        "warnings": ["Unable to analyze - audio quality too poor"],
    }
    result = AuriginResultNormalizer.normalize(payload, processing_ms=80)
    assert result.result == "UNKNOWN"
    assert result.score == 0.0
    assert result.confidence == 0.0
    assert result.reason == "Unable to analyze - audio quality too poor"
    assert result.raw_result is None


def test_normalize_missing_global():
    payload = {"prediction_id": "pred_5"}
    result = AuriginResultNormalizer.normalize(payload, processing_ms=50)
    assert result.result == "UNKNOWN"
    assert result.score == 0.45
    assert result.confidence is None


def test_normalize_missing_confidence(): 
    payload = {
        "prediction_id": "pred_6",
        "global": {"result": "bonafide", "score": 0.05},
    }
    result = AuriginResultNormalizer.normalize(payload)
    assert result.result == "REAL"
    assert result.score == 0.05
    assert result.confidence is None


def test_normalize_valid_confidence_without_score(): 
    payload = {
        "prediction_id": "pred_7",
        "global": {"result": "bonafide", "confidence": 0.98},
    }
    result = AuriginResultNormalizer.normalize(payload)
    assert result.result == "REAL"
    assert result.score == 0.02
    assert result.confidence == 0.98


def test_normalize_result_value_mappings():
    """Verify centralized normalize_result_value mapping contract."""
    # Test 1: bonafide -> REAL
    assert AuriginResultNormalizer.normalize_result_value("bonafide") == "REAL"
    assert AuriginResultNormalizer.normalize_result_value("authentic") == "REAL"
    assert AuriginResultNormalizer.normalize_result_value("real") == "REAL"

    # Test 2: spoofed -> SPOOFED
    assert AuriginResultNormalizer.normalize_result_value("spoofed") == "SPOOFED"
    assert AuriginResultNormalizer.normalize_result_value("spoof") == "SPOOFED"

    # Test 3: partially_spoofed -> SPOOFED
    assert AuriginResultNormalizer.normalize_result_value("partially_spoofed") == "SPOOFED"

    # Test 4: unknown -> UNKNOWN
    assert AuriginResultNormalizer.normalize_result_value("unknown") == "UNKNOWN"
    assert AuriginResultNormalizer.normalize_result_value(None) == "UNKNOWN"
    assert AuriginResultNormalizer.normalize_result_value("other_unsupported_string") == "UNKNOWN"


def test_partially_spoofed_alone_is_spoof_evidence():
    """Test 6: A call containing only a partially_spoofed result is treated as spoof evidence."""
    from app.services.detection_decision_engine import DetectionDecisionEngine
    engine = DetectionDecisionEngine(
        spoof_threshold=0.70,
        early_spoof_threshold=0.85,
    )
    out = engine.add_observation(
        sequence_number=1,
        window_start_ms=0,
        window_end_ms=10000,
        raw_result="partially_spoofed",
        normalized_result=AuriginResultNormalizer.normalize_result_value("partially_spoofed"),
        score=0.75,
        confidence=0.90,
    )
    assert out.decision_state in ("suspicious", "confirmed")
    assert out.call_score >= 0.70
    assert engine.get_risk_level(out.call_score, out.decision_state) in ("MEDIUM", "HIGH")


def test_mixed_call_real_plus_partially_spoofed_vs_real_plus_spoofed():
    """Test 7: A mixed call containing REAL + PARTIALLY_SPOOFED behaves the same as REAL + SPOOFED."""
    from app.services.detection_decision_engine import DetectionDecisionEngine
    engine_partial = DetectionDecisionEngine(
        spoof_threshold=0.70,
        early_spoof_threshold=0.90,
        min_windows_for_confirmation=2,
    )
    engine_spoof = DetectionDecisionEngine(
        spoof_threshold=0.70,
        early_spoof_threshold=0.90,
        min_windows_for_confirmation=2,
    )

    # Window 1: REAL
    engine_partial.add_observation(
        sequence_number=1, window_start_ms=0, window_end_ms=10000,
        raw_result="bonafide", normalized_result="REAL", score=0.05, confidence=0.95,
    )
    engine_spoof.add_observation(
        sequence_number=1, window_start_ms=0, window_end_ms=10000,
        raw_result="bonafide", normalized_result="REAL", score=0.05, confidence=0.95,
    )

    # Window 2: PARTIALLY_SPOOFED vs SPOOFED with same score/confidence
    out_partial = engine_partial.add_observation(
        sequence_number=2, window_start_ms=0, window_end_ms=20000,
        raw_result="partially_spoofed", normalized_result="SPOOFED", score=0.75, confidence=0.90,
    )
    out_spoof = engine_spoof.add_observation(
        sequence_number=2, window_start_ms=0, window_end_ms=20000,
        raw_result="spoofed", normalized_result="SPOOFED", score=0.75, confidence=0.90,
    )

    assert out_partial.decision_state == out_spoof.decision_state
    assert out_partial.call_score == out_spoof.call_score
    assert out_partial.call_confidence == out_spoof.call_confidence


def test_partially_spoofed_plus_spoofed():
    """Test: PARTIALLY_SPOOFED + SPOOFED confirms spoofing."""
    from app.services.detection_decision_engine import DetectionDecisionEngine
    engine = DetectionDecisionEngine(
        spoof_threshold=0.70,
        min_windows_for_confirmation=2,
    )
    # Window 1: PARTIALLY_SPOOFED
    engine.add_observation(
        sequence_number=1, window_start_ms=0, window_end_ms=10000,
        raw_result="partially_spoofed", normalized_result="SPOOFED", score=0.75, confidence=0.90,
    )
    # Window 2: SPOOFED
    out = engine.add_observation(
        sequence_number=2, window_start_ms=0, window_end_ms=20000,
        raw_result="spoofed", normalized_result="SPOOFED", score=0.85, confidence=0.95,
    )
    assert out.decision_state == "confirmed"
    assert out.user_label == "VOICE CLONING DETECTED"


