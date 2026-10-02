"""Unit tests for VoiceShield V2 database configuration, environment validation, and production safeguards."""

from __future__ import annotations

import os
from unittest.mock import patch
import pytest

from app.core.config import Settings, get_settings


def test_production_without_database_url_fails():
    """Verify production fails fast when DATABASE_URL is missing."""
    get_settings.cache_clear()
    with patch.dict(os.environ, {"ENVIRONMENT": "production", "DATABASE_URL": ""}, clear=True):
        with pytest.raises(RuntimeError) as exc_info:
            get_settings()
        assert "Production environment requires an explicit PostgreSQL DATABASE_URL" in str(exc_info.value)
    get_settings.cache_clear()


def test_production_with_sqlite_url_fails():
    """Verify production fails fast when DATABASE_URL points to SQLite."""
    get_settings.cache_clear()
    with patch.dict(os.environ, {
        "ENVIRONMENT": "production",
        "DATABASE_URL": "sqlite:///./voice_clone_detection.db"
    }, clear=True):
        with pytest.raises(RuntimeError) as exc_info:
            get_settings()
        assert "SQLite fallback is strictly forbidden in production" in str(exc_info.value)
    get_settings.cache_clear()


def test_production_with_postgres_url_succeeds():
    """Verify production succeeds when valid PostgreSQL DATABASE_URL is provided."""
    get_settings.cache_clear()
    with patch.dict(os.environ, {
        "ENVIRONMENT": "production",
        "DATABASE_URL": "postgresql+psycopg://testuser:testpass@localhost:5432/testdb",
        "CORS_ORIGINS": "https://voiceshield.example.com",
    }, clear=True):
        settings = get_settings()
        assert settings.is_production is True
        assert settings.is_postgres is True
        assert settings.is_sqlite is False
        assert settings.database_url == "postgresql+psycopg://testuser:testpass@localhost:5432/testdb"
    get_settings.cache_clear()


def test_postgres_protocol_normalization():
    """Verify postgres:// and postgresql:// are normalized to postgresql+psycopg://."""
    get_settings.cache_clear()
    with patch.dict(os.environ, {
        "ENVIRONMENT": "production",
        "DATABASE_URL": "postgres://testuser:testpass@localhost:5432/testdb",
        "CORS_ORIGINS": "https://voiceshield.example.com",
    }, clear=True):
        settings = get_settings()
        assert settings.database_url == "postgresql+psycopg://testuser:testpass@localhost:5432/testdb"

    get_settings.cache_clear()
    with patch.dict(os.environ, {
        "ENVIRONMENT": "production",
        "DATABASE_URL": "postgresql://testuser:testpass@localhost:5432/testdb",
        "CORS_ORIGINS": "https://voiceshield.example.com",
    }, clear=True):
        settings = get_settings()
        assert settings.database_url == "postgresql+psycopg://testuser:testpass@localhost:5432/testdb"
    get_settings.cache_clear()


def test_development_defaults():
    """Verify development mode defaults and allows development settings."""
    get_settings.cache_clear()
    with patch.dict(os.environ, {
        "ENVIRONMENT": "development",
        "CORS_ORIGINS": "http://localhost:3000",
    }, clear=True):
        settings = get_settings()
        assert settings.is_production is False
        assert settings.is_sqlite is True
    get_settings.cache_clear()


def test_test_environment_in_memory_default():
    """Verify test environment defaults to in-memory SQLite if no URL is specified."""
    get_settings.cache_clear()
    with patch.dict(os.environ, {
        "ENVIRONMENT": "test",
        "CORS_ORIGINS": "http://localhost:3000",
    }, clear=True):
        settings = get_settings()
        assert settings.is_test is True
        assert settings.database_url == "sqlite:///:memory:"
    get_settings.cache_clear()


def test_production_init_database_does_not_call_create_all_or_sqlite_upgrade():
    """Verify production init_database() does not execute Base.metadata.create_all or upgrade_sqlite_schema."""
    from app.db import database

    with patch.object(database.settings, "environment", "production"):
        with patch.object(database.Base.metadata, "create_all") as mock_create_all:
            with patch.object(database, "upgrade_sqlite_schema") as mock_upgrade:
                with patch("app.db.seed_compatibility.seed_device_compatibilities") as mock_seed:
                    database.init_database()
                    mock_create_all.assert_not_called()
                    mock_upgrade.assert_not_called()
                    mock_seed.assert_called_once()


def test_upgrade_sqlite_schema_disabled_in_production():
    """Verify upgrade_sqlite_schema does nothing when executed in production or non-SQLite."""
    from app.db import database

    with patch.object(database.settings, "environment", "production"):
        with patch.object(database.engine, "begin") as mock_begin:
            database.upgrade_sqlite_schema()
            mock_begin.assert_not_called()

