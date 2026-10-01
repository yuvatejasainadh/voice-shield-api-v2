"""Application configuration loaded from environment variables."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, ValidationError


class Settings(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    app_name: str = Field(default="Voice Clone Detection API")
    app_version: str = Field(default="0.1.0")
    database_url: str = Field(default="sqlite:///./voice_clone_detection.db")
    max_upload_size_mb: int = Field(default=25, ge=1, le=500)
    storage_path: str = Field(default="./storage")
    log_level: str = Field(default="INFO")
    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    risk_genuine_max: int = Field(default=29, ge=0, le=100)
    risk_suspicious_max: int = Field(default=69, ge=0, le=100)
    max_audio_duration_seconds: int = Field(default=600, ge=5, le=7200)

    # Audio preprocessor settings
    model_sample_rate: int = Field(default=16000, ge=8000, le=48000)
    analysis_segment_seconds: float = Field(default=6.0, ge=0.5, le=30.0)
    max_segments_per_analysis: int = Field(default=10, ge=1, le=100)
    # Reality Defender RealAPI Settings
    reality_defender_api_key: str | None = Field(default=None)
    reality_defender_api_base_url: str = Field(default="https://api.prd.realitydefender.xyz")
    reality_defender_timeout_seconds: int = Field(default=120, ge=1, le=600)
    # Aurigin realtime detection settings
    aurigin_api_key: str | None = Field(default=None)
    aurigin_api_base_url: str = Field(default="https://api.aurigin.ai")
    aurigin_timeout_seconds: int = Field(default=30, ge=1, le=300)
    aurigin_enabled: bool = Field(default=False)
    aurigin_provider_version: str = Field(default="v1")
    # Realtime risk engine settings (experimental defaults)
    realtime_medium_threshold: float = Field(default=0.55, ge=0.0, le=1.0)
    realtime_high_threshold: float = Field(default=0.80, ge=0.0, le=1.0)
    realtime_window_size: int = Field(default=5, ge=1, le=20)
    realtime_hysteresis_min_observations: int = Field(default=3, ge=1, le=10)
    realtime_cooldown_seconds: int = Field(default=30, ge=0, le=300)
    realtime_chunk_min_duration_seconds: float = Field(default=0.5, ge=0.1, le=30.0)
    realtime_chunk_max_duration_seconds: float = Field(default=30.0, ge=1.0, le=120.0)
    realtime_chunk_max_duration_ms: int = Field(default=30000, ge=1000, le=120000)
    # Cumulative window evaluation strategy (configurable)
    realtime_initial_window_seconds: float = Field(default=2.5, ge=0.5, le=60.0)
    realtime_second_window_seconds: float = Field(default=5.0, ge=0.5, le=120.0)
    realtime_primary_decision_window_seconds: int = Field(default=30, ge=1, le=300)
    realtime_continuation_step_seconds: float = Field(default=2.5, ge=0.5, le=60.0)
    # TCED (Temporal Cascaded Evaluation/Detection) settings
    tced_new_audio_interval_seconds: float = Field(default=2.5, ge=0.5, le=60.0)
    tced_window_stride_seconds: float = Field(default=2.5, ge=0.5, le=120.0)
    tced_max_window_length_seconds: float = Field(default=5.0, ge=0.5, le=300.0)
    # Decision engine thresholds and policy
    decision_spoof_threshold: float = Field(default=0.70, ge=0.0, le=1.0)
    decision_early_spoof_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
    decision_genuine_threshold: float = Field(default=0.25, ge=0.0, le=1.0)
    decision_min_confidence: float = Field(default=0.60, ge=0.0, le=1.0)
    decision_min_windows_for_confirmation: int = Field(default=2, ge=1, le=10)
    decision_allow_early_detection: bool = Field(default=True)

    @property
    def max_upload_size_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache
def get_settings() -> Settings:
    """Load settings once and cache the result for application lifetime."""
    load_dotenv()
    raw_db_url = os.getenv("DATABASE_URL", "sqlite:///./voice_clone_detection.db")
    if raw_db_url.startswith("postgres://"):
        raw_db_url = "postgresql+psycopg://" + raw_db_url[len("postgres://"):]
    elif raw_db_url.startswith("postgresql://"):
        raw_db_url = "postgresql+psycopg://" + raw_db_url[len("postgresql://"):]

    raw_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000")
    try:
        settings = Settings(
            app_name=os.getenv("APP_NAME", "Voice Clone Detection API"),
            app_version=os.getenv("APP_VERSION", "0.1.0"),
            database_url=raw_db_url,
            max_upload_size_mb=int(os.getenv("MAX_UPLOAD_SIZE_MB", "25")),
            storage_path=os.getenv("STORAGE_PATH", "./storage"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            cors_origins=[item.strip() for item in raw_origins.split(",") if item.strip()],
            risk_genuine_max=int(os.getenv("RISK_GENUINE_MAX", "29")),
            risk_suspicious_max=int(os.getenv("RISK_SUSPICIOUS_MAX", "69")),
            max_audio_duration_seconds=int(os.getenv("MAX_AUDIO_DURATION_SECONDS", "600")),
            reality_defender_api_key=os.getenv("REALITY_DEFENDER_API_KEY") or None,
            reality_defender_api_base_url=os.getenv("REALITY_DEFENDER_API_BASE_URL", "https://api.prd.realitydefender.xyz"),
            reality_defender_timeout_seconds=int(os.getenv("REALITY_DEFENDER_TIMEOUT_SECONDS", "120")),
            aurigin_api_key=os.getenv("AURIGIN_API_KEY") or None,
            aurigin_api_base_url=os.getenv("AURIGIN_API_BASE_URL", "https://api.aurigin.ai"),
            aurigin_timeout_seconds=int(os.getenv("AURIGIN_TIMEOUT_SECONDS", "30")),
            aurigin_enabled=os.getenv("AURIGIN_ENABLED", "false").lower() == "true",
            aurigin_provider_version=os.getenv("AURIGIN_PROVIDER_VERSION", "v1"),
            realtime_medium_threshold=float(os.getenv("REALTIME_MEDIUM_THRESHOLD", "0.55")),
            realtime_high_threshold=float(os.getenv("REALTIME_HIGH_THRESHOLD", "0.80")),
            realtime_window_size=int(os.getenv("REALTIME_WINDOW_SIZE", "5")),
            realtime_hysteresis_min_observations=int(os.getenv("REALTIME_HYSTERESIS_MIN_OBSERVATIONS", "3")),
            realtime_cooldown_seconds=int(os.getenv("REALTIME_COOLDOWN_SECONDS", "30")),
            realtime_chunk_min_duration_seconds=float(os.getenv("REALTIME_CHUNK_MIN_DURATION_SECONDS", "0.5")),
            realtime_chunk_max_duration_seconds=float(os.getenv("REALTIME_CHUNK_MAX_DURATION_SECONDS", "30.0")),
            realtime_chunk_max_duration_ms=int(os.getenv("REALTIME_CHUNK_MAX_DURATION_MS", "30000")),
            realtime_initial_window_seconds=float(os.getenv("REALTIME_INITIAL_WINDOW_SECONDS", "2.5")),
            realtime_second_window_seconds=float(os.getenv("REALTIME_SECOND_WINDOW_SECONDS", "5.0")),
            realtime_primary_decision_window_seconds=int(os.getenv("REALTIME_PRIMARY_DECISION_WINDOW_SECONDS", "30")),
            realtime_continuation_step_seconds=float(os.getenv("REALTIME_CONTINUATION_STEP_SECONDS", "2.5")),
            tced_new_audio_interval_seconds=float(os.getenv("TCED_NEW_AUDIO_INTERVAL_SECONDS", "2.5")),
            tced_window_stride_seconds=float(os.getenv("TCED_WINDOW_STRIDE_SECONDS", "2.5")),
            tced_max_window_length_seconds=float(os.getenv("TCED_MAX_WINDOW_LENGTH_SECONDS", "5.0")),
            decision_spoof_threshold=float(os.getenv("DECISION_SPOOF_THRESHOLD", "0.70")),
            decision_early_spoof_threshold=float(os.getenv("DECISION_EARLY_SPOOF_THRESHOLD", "0.85")),
            decision_genuine_threshold=float(os.getenv("DECISION_GENUINE_THRESHOLD", "0.25")),
            decision_min_confidence=float(os.getenv("DECISION_MIN_CONFIDENCE", "0.60")),
            decision_min_windows_for_confirmation=int(os.getenv("DECISION_MIN_WINDOWS_FOR_CONFIRMATION", "2")),
            decision_allow_early_detection=os.getenv("DECISION_ALLOW_EARLY_DETECTION", "true").lower() == "true",
        )
    except (ValidationError, ValueError) as exc:
        raise RuntimeError(f"Invalid environment configuration: {exc}") from exc

    if not settings.cors_origins:
        raise RuntimeError("CORS_ORIGINS must include at least one allowed origin")

    if settings.risk_suspicious_max <= settings.risk_genuine_max:
        raise RuntimeError("RISK_SUSPICIOUS_MAX must be greater than RISK_GENUINE_MAX")

    Path(settings.storage_path).mkdir(parents=True, exist_ok=True)
    return settings
