"""Service for file validation and persistence."""

from __future__ import annotations

import logging
from pathlib import Path

from fastapi import HTTPException, UploadFile, status

from app.audio.metadata import AudioMetadata
from app.audio.validator import AudioValidator
from app.core.config import get_settings
from app.utils.audio_utils import ensure_storage_path, generate_safe_filename


logger = logging.getLogger("voice-clone-detection")


class AudioService:
    """Handles secure upload validation and file storage."""

    def __init__(self) -> None:
        self.settings = get_settings()
        self.storage_path = ensure_storage_path(self.settings.storage_path)
        self.validator = AudioValidator()

    async def save_upload(self, upload: UploadFile) -> tuple[str, str, int, AudioMetadata]:
        """Validate and store uploaded audio; return stored path, safe filename, size, metadata."""
        if not upload.filename:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Missing filename")

        extension = self.validator.validate_extension(upload.filename)
        safe_filename = generate_safe_filename(extension)
        destination = self.storage_path / safe_filename

        content = await upload.read()
        size = len(content)
        if size <= 0:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail={"code": "EMPTY_AUDIO", "message": "Audio file is empty"},
            )

        if size > self.settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail={"code": "FILE_TOO_LARGE", "message": "File too large"},
            )

        destination.write_bytes(content)
        try:
            metadata = self.validator.validate_saved_file(destination, extension)
        except Exception:
            destination.unlink(missing_ok=True)
            raise
        logger.info(
            "Audio stored filename=%s size_bytes=%s sample_rate=%s channels=%s duration_seconds=%.3f",
            safe_filename,
            size,
            metadata.sample_rate,
            metadata.channels,
            metadata.duration_seconds,
        )
        return str(destination), safe_filename, size, metadata

    def resolve_for_processing(self, stored_path: str) -> Path:
        """Resolve stored file path for internal processing only."""
        return Path(stored_path)

    @staticmethod
    def discard_upload(stored_path: str) -> None:
        """Remove an upload when analysis cannot produce a result."""
        Path(stored_path).unlink(missing_ok=True)
