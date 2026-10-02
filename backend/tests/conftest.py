"""Pytest global configuration and environment setup for VoiceShield V2 tests."""

import os
import pytest

# Ensure tests default to isolated test environment
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")
os.environ.setdefault("STORAGE_PATH", "./storage")
os.environ.setdefault("LOG_LEVEL", "WARNING")

from app.core.config import get_settings
from app.db.database import Base, engine, SessionLocal
from app.db.seed_compatibility import seed_device_compatibilities


@pytest.fixture(scope="session", autouse=True)
def setup_test_database_schema():
    """Create all schema tables on the test database engine and seed reference data."""
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as session:
        seed_device_compatibilities(session)
    yield
