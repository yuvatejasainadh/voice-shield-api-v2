from __future__ import annotations

from typing import Any


class RealtimeEventPublisher:
    """V1 publisher abstraction. This implementation is intentionally a no-op while HTTP responses remain the delivery mechanism."""

    def publish_state_change(self, **payload: Any) -> None:
        """Future WebSocket/SSE integration point. Implement when push infrastructure is available."""
        return None
