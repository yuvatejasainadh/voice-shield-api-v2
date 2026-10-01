"""Audio utility helpers for secure upload handling."""

from __future__ import annotations

import os
import uuid
from pathlib import Path

ALLOWED_EXTENSIONS = {"wav", "mp3", "m4a", "aac", "ogg", "flac", "webm"}


def get_extension(filename: str) -> str:
    """Return a normalized extension without the leading dot."""
    _, ext = os.path.splitext(filename.lower())
    return ext.lstrip(".")


def is_allowed_extension(filename: str) -> bool:
    """Validate user-provided filename extension against allow-list."""
    return get_extension(filename) in ALLOWED_EXTENSIONS


def generate_safe_filename(extension: str) -> str:
    """Create a server-side filename that never trusts user-supplied names."""
    return f"audio_{uuid.uuid4().hex}.{extension}"


def ensure_storage_path(storage_path: str) -> Path:
    """Ensure storage path exists and return it as Path."""
    path = Path(storage_path)
    path.mkdir(parents=True, exist_ok=True)
    return path
