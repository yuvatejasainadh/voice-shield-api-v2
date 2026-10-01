"""Structured logging helpers."""

import logging
import sys


def configure_logging(level: str = "INFO") -> None:
    """Configure root logger with a production-friendly structured format."""
    log_format = "%(asctime)s.%(msecs)03d %(levelname)s %(message)s"
    date_format = "%H:%M:%S"
    
    # Configure root logger
    logging.basicConfig(
        level=level.upper(),
        format=log_format,
        datefmt=date_format,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Silence low-level HTTP transport noise at INFO level
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

