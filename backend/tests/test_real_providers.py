"""Real-provider integration tests for Aurigin and Reality Defender APIs.

Run with:
    $env:RUN_REAL_PROVIDER_TESTS="1"; pytest tests/test_real_providers.py -v
"""

from __future__ import annotations

import os
from pathlib import Path
import pytest

from app.core.config import get_settings
from app.services.aurigin_service import AuriginService
from app.services.reality_defender_service import RealityDefenderService


@pytest.fixture(autouse=True)
def skip_unless_real_providers():
    if os.getenv("RUN_REAL_PROVIDER_TESTS", "0").lower() not in {"1", "true"}:
        pytest.skip("Set RUN_REAL_PROVIDER_TESTS=1 to run real provider integration tests")


@pytest.mark.anyio
async def test_real_aurigin_analysis():
    """Test real Aurigin /v1/predict API with sample audio if AURIGIN_API_KEY is configured."""
    settings = get_settings()
    if not settings.aurigin_api_key:
        pytest.skip("AURIGIN_API_KEY is not set")

    sample_path = Path(__file__).resolve().parents[1] / "testvoices" / "genuine" / "LJ001-0004.wav"
    if not sample_path.exists():
        pytest.skip(f"Sample audio not found at {sample_path}")

    service = AuriginService()
    result = await service.analyze_audio(
        sample_path.read_bytes(),
        filename="test_sample.wav",
        content_type="audio/wav",
    )

    assert result["provider"] == "aurigin"
    assert result["result"] in ("REAL", "SPOOF", "UNKNOWN")
    assert result["score"] is not None
    assert 0.0 <= result["score"] <= 1.0


@pytest.mark.anyio
async def test_real_reality_defender_analysis():
    """Test real Reality Defender RealAPI analysis with audio sample."""
    settings = get_settings()
    if not settings.reality_defender_api_key:
        pytest.skip("REALITY_DEFENDER_API_KEY is not set")

    sample_path = Path(__file__).resolve().parents[1] / "testvoices" / "genuine" / "LJ001-0004.wav"
    if not sample_path.exists():
        pytest.skip(f"Sample audio not found at {sample_path}")

    service = RealityDefenderService()
    result = await service.analyze_file(sample_path)

    assert result["status"] == "completed"
    assert result["classification"] in ("LIKELY_GENUINE", "SUSPICIOUS", "LIKELY_AI_GENERATED")
    assert 0 <= result["risk_score"] <= 100
    assert result["detector_version"] == "reality-defender"

