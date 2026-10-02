"""Pytest global configuration and fixtures for VoiceShield V2 test suite."""

from __future__ import annotations

import os
from collections.abc import Generator
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

# Force test environment defaults before application imports
os.environ["ENVIRONMENT"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"
os.environ["STORAGE_PATH"] = "./storage"
os.environ["LOG_LEVEL"] = "WARNING"

from app.core.config import get_settings
from app.db.database import Base, get_db, init_database
from app.db.seed_compatibility import seed_device_compatibilities
from app.main import app


@pytest.fixture(scope="session")
def test_engine():
    """Create a persistent in-memory SQLite engine for the test session."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    return engine


@pytest.fixture(scope="function")
def db_session(test_engine) -> Generator[Session, None, None]:
    """Provide an isolated, transactional database session per test function."""
    connection = test_engine.connect()
    transaction = connection.begin()
    SessionLocal = sessionmaker(bind=connection, autocommit=False, autoflush=False)
    session = SessionLocal()

    # Seed initial device compatibilities
    seed_device_compatibilities(session)

    try:
        yield session
    finally:
        session.close()
        transaction.rollback()
        connection.close()


@pytest.fixture(scope="function")
def client(db_session: Session) -> Generator[TestClient, None, None]:
    """Provide a FastAPI TestClient with the request-scoped database dependency overridden."""
    def _override_get_db() -> Generator[Session, None, None]:
        try:
            yield db_session
        finally:
            pass

    app.dependency_overrides[get_db] = _override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()
