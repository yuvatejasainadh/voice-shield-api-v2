"""Sanitized Provider Debug Logger and Artifact Store."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
from typing import Any

from app.core.config import get_settings

logger = logging.getLogger("voice-clone-detection")

REDACT_KEYS = {
    "api_key",
    "api-key",
    "authorization",
    "token",
    "secret",
    "signature",
    "sig",
    "upload_url",
    "file_url",
    "presigned_url",
    "download_urls",
    "upload_urls",
}

PII_PATTERNS = [
    (re.compile(r"\b\d{10,12}\b"), "[REDACTED_PHONE_OR_ID]"),
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[REDACTED_CARD]"),
]


def redact_sensitive_data(obj: Any) -> Any:
    """Recursively redact API keys, tokens, SAS URLs, and sensitive headers."""
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            k_lower = str(k).lower()
            if any(rk in k_lower for rk in REDACT_KEYS):
                cleaned[k] = "[REDACTED]"
            else:
                cleaned[k] = redact_sensitive_data(v)
        return cleaned
    elif isinstance(obj, list):
        return [redact_sensitive_data(item) for item in obj]
    elif isinstance(obj, str):
        # Redact signatures/SAS parameters in URLs: preserve scheme, host, path and redact query
        if ("sig=" in obj or "sv=" in obj or "se=" in obj) and "?" in obj:
            base_part = obj.split("?")[0]
            return f"{base_part}?[REDACTED]"
        if "sig=" in obj or "sv=" in obj or "se=" in obj:
            return "[REDACTED_SAS_URL]"
        # Redact raw Bearer tokens
        if obj.startswith("Bearer ") or obj.startswith("sk_") or obj.startswith("gsk_") or obj.startswith("rd_"):
            return "[REDACTED_SECRET]"
        return obj
    return obj


def log_sanitized_provider_debug(provider: str, event_type: str, data: dict[str, Any]) -> None:
    """Log structured sanitized debug information if ENABLE_PROVIDER_DEBUG_LOGGING is active."""
    settings = get_settings()
    if not getattr(settings, "enable_provider_debug_logging", False):
        return

    sanitized = redact_sensitive_data(data)
    preview = json.dumps(sanitized, ensure_ascii=False)
    if len(preview) > 500:
        preview = preview[:500] + "... [TRUNCATED]"

    logger.debug("PROVIDER DEBUG [%s] %s: %s", provider, event_type, preview)


def save_provider_debug_artifact(
    request_id: str,
    artifact_name: str,
    raw_payload: Any,
    subfolder: str | None = None,
) -> None:
    """Store sanitized JSON provider response in storage/provider_debug/<request_id>/... if enabled."""
    settings = get_settings()
    if not getattr(settings, "enable_provider_debug_artifacts", False):
        return

    base_debug_dir = Path(getattr(settings, "provider_debug_dir", "storage/provider_debug")) / request_id
    if subfolder:
        base_debug_dir = base_debug_dir / subfolder

    try:
        base_debug_dir.mkdir(parents=True, exist_ok=True)
        filename = artifact_name if artifact_name.endswith(".json") else f"{artifact_name}.json"
        artifact_path = base_debug_dir / filename
        sanitized = redact_sensitive_data(raw_payload)
        artifact_path.write_text(json.dumps(sanitized, indent=2, ensure_ascii=False), encoding="utf-8")
        logger.debug("Saved provider debug artifact: %s", artifact_path)
    except Exception as exc:
        logger.warning("Failed to save provider debug artifact %s: %s", artifact_name, exc)
