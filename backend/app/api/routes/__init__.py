"""Route package exports."""

from app.api.routes import analysis, device, health, history, ready, realtime, realtime_ws, reports

__all__ = [
    "analysis",
    "device",
    "health",
    "history",
    "ready",
    "realtime",
    "realtime_ws",
    "reports",
]
