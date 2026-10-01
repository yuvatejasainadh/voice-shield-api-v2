from __future__ import annotations

from typing import Any, Protocol


class VoiceDeepfakeProvider(Protocol):
    """Provider abstraction for realtime and final deepfake analysis."""

    async def analyze_audio(
        self,
        audio_path_or_bytes: str | bytes | Any,
        filename: str,
        content_type: str | None = None,
    ) -> dict[str, Any]:
        ...
