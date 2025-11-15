"""
Tests for configuration management.

Tests loading settings from environment variables, .env files, and validation.
"""

import os
import pytest
from pydantic import ValidationError


def test_settings_defaults():
    """Test that Settings loads with sensible defaults."""
    # Create settings with no environment variables
    from src.config import Settings

    settings = Settings()

    assert settings.llm_provider == "claude"
    assert settings.database_url == "sqlite:///./codeproject.db"
    assert settings.host == "0.0.0.0"
    assert settings.port == 8000
    assert settings.log_level == "INFO"
    assert settings.ollama_base_url == "http://localhost:11434"


def test_settings_from_environment_variables(monkeypatch):
    """Test that Settings loads from environment variables."""
    # Set environment variables
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("CLAUDE_API_KEY", "sk-test-key-123")
    monkeypatch.setenv("WEBHOOK_SECRET", "webhook-secret-123")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token_123")
    monkeypatch.setenv("DATABASE_URL", "postgresql://localhost/code_review")
    monkeypatch.setenv("HOST", "127.0.0.1")
    monkeypatch.setenv("PORT", "9000")

    from src.config import Settings

    settings = Settings()

    assert settings.llm_provider == "ollama"
    assert settings.claude_api_key == "sk-test-key-123"
    assert settings.webhook_secret == "webhook-secret-123"
    assert settings.github_token == "ghp_test_token_123"
    assert settings.database_url == "postgresql://localhost/code_review"
    assert settings.host == "127.0.0.1"
    assert settings.port == 9000


def test_settings_llm_provider_validation(monkeypatch):
    """Test that LLM provider is validated to claude or ollama."""
    monkeypatch.setenv("LLM_PROVIDER", "invalid_provider")

    from src.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "llm_provider" in str(exc_info.value).lower()


def test_settings_valid_llm_providers(monkeypatch):
    """Test that both valid LLM providers are accepted."""
    from src.config import Settings

    # Test claude
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    settings = Settings()
    assert settings.llm_provider == "claude"

    # Test ollama
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    settings = Settings()
    assert settings.llm_provider == "ollama"


def test_settings_database_url_sqlite(monkeypatch):
    """Test that SQLite database URLs are accepted."""
    monkeypatch.setenv("DATABASE_URL", "sqlite:///./test.db")

    from src.config import Settings

    settings = Settings()
    assert settings.database_url == "sqlite:///./test.db"


def test_settings_database_url_postgresql(monkeypatch):
    """Test that PostgreSQL database URLs are accepted."""
    monkeypatch.setenv("DATABASE_URL", "postgresql://user:password@localhost/dbname")

    from src.config import Settings

    settings = Settings()
    assert settings.database_url == "postgresql://user:password@localhost/dbname"


def test_settings_database_url_postgres_shorthand(monkeypatch):
    """Test that postgres:// shorthand is accepted."""
    monkeypatch.setenv("DATABASE_URL", "postgres://localhost/dbname")

    from src.config import Settings

    settings = Settings()
    assert settings.database_url == "postgres://localhost/dbname"


def test_settings_database_url_validation(monkeypatch):
    """Test that invalid database URLs are rejected."""
    monkeypatch.setenv("DATABASE_URL", "mysql://localhost/dbname")

    from src.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "database_url" in str(exc_info.value).lower()


def test_settings_port_validation(monkeypatch):
    """Test that port is validated to be within valid range."""
    monkeypatch.setenv("PORT", "99999")

    from src.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "port" in str(exc_info.value).lower()


def test_settings_valid_port_range(monkeypatch):
    """Test that valid port numbers are accepted."""
    from src.config import Settings

    # Test minimum valid port
    monkeypatch.setenv("PORT", "1")
    settings = Settings()
    assert settings.port == 1

    # Test maximum valid port
    monkeypatch.setenv("PORT", "65535")
    settings = Settings()
    assert settings.port == 65535

    # Test common port
    monkeypatch.setenv("PORT", "8080")
    settings = Settings()
    assert settings.port == 8080


def test_settings_log_level_validation(monkeypatch):
    """Test that log level is validated."""
    monkeypatch.setenv("LOG_LEVEL", "INVALID_LEVEL")

    from src.config import Settings

    with pytest.raises(ValidationError) as exc_info:
        Settings()

    assert "log_level" in str(exc_info.value).lower()


def test_settings_valid_log_levels(monkeypatch):
    """Test that all valid log levels are accepted."""
    from src.config import Settings

    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    for level in valid_levels:
        monkeypatch.setenv("LOG_LEVEL", level)
        settings = Settings()
        assert settings.log_level == level


def test_settings_log_level_case_insensitive(monkeypatch):
    """Test that log level is case insensitive."""
    from src.config import Settings

    monkeypatch.setenv("LOG_LEVEL", "debug")
    settings = Settings()
    assert settings.log_level == "DEBUG"

    monkeypatch.setenv("LOG_LEVEL", "InFo")
    settings = Settings()
    assert settings.log_level == "INFO"


def test_settings_all_fields_present():
    """Test that all expected configuration fields are present."""
    from src.config import Settings

    settings = Settings()

    # LLM Configuration
    assert hasattr(settings, "llm_provider")
    assert hasattr(settings, "claude_api_key")
    assert hasattr(settings, "ollama_base_url")

    # Git & Webhook Integration
    assert hasattr(settings, "webhook_secret")
    assert hasattr(settings, "github_token")
    assert hasattr(settings, "gitlab_token")

    # Database Configuration
    assert hasattr(settings, "database_url")

    # Server Configuration
    assert hasattr(settings, "host")
    assert hasattr(settings, "port")

    # Logging Configuration
    assert hasattr(settings, "log_level")


def test_global_settings_instance():
    """Test that global settings instance is available."""
    from src.config import settings

    assert settings is not None
    assert settings.llm_provider in ["claude", "ollama"]
    assert settings.port > 0


def test_settings_empty_optional_fields():
    """Test that optional fields can be empty strings."""
    from src.config import Settings

    settings = Settings()

    # These fields can be empty initially
    assert settings.claude_api_key == ""
    assert settings.webhook_secret == ""
    assert settings.github_token == ""
    assert settings.gitlab_token == ""


def test_settings_immutability(monkeypatch):
    """Test that settings can be accessed multiple times with same values."""
    monkeypatch.setenv("LLM_PROVIDER", "claude")
    monkeypatch.setenv("PORT", "8001")

    from src.config import Settings

    settings1 = Settings()
    settings2 = Settings()

    assert settings1.llm_provider == settings2.llm_provider
    assert settings1.port == settings2.port


def test_settings_with_env_file(tmp_path, monkeypatch):
    """Test that Settings can load from .env file."""
    # Create a temporary .env file
    env_file = tmp_path / ".env"
    env_file.write_text(
        "LLM_PROVIDER=ollama\n"
        "CLAUDE_API_KEY=sk-from-file\n"
        "WEBHOOK_SECRET=secret-from-file\n"
        "DATABASE_URL=postgresql://localhost/testdb\n"
        "PORT=9999\n"
    )

    # Change to the directory with the .env file
    monkeypatch.chdir(tmp_path)

    from src.config import Settings

    settings = Settings()

    assert settings.llm_provider == "ollama"
    assert settings.claude_api_key == "sk-from-file"
    assert settings.webhook_secret == "secret-from-file"
    assert settings.database_url == "postgresql://localhost/testdb"
    assert settings.port == 9999
