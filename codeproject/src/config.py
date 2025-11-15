"""
Configuration management using Pydantic BaseSettings.

Loads settings from environment variables and .env files with type validation
and sensible defaults.
"""

from typing import Literal
from pydantic_settings import BaseSettings
from pydantic import Field, field_validator, ConfigDict


class Settings(BaseSettings):
    """
    Application configuration settings.

    Loads from environment variables and .env file. All settings are typed
    and validated with sensible defaults.
    """

    # ============================================================================
    # LLM Configuration
    # ============================================================================

    llm_provider: Literal["claude", "ollama"] = Field(
        default="claude",
        description="LLM provider to use for analysis (claude or ollama)"
    )

    claude_api_key: str = Field(
        default="",
        description="Anthropic Claude API key"
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama instance"
    )

    # ============================================================================
    # Git & Webhook Integration
    # ============================================================================

    webhook_secret: str = Field(
        default="",
        description="Secret for verifying GitHub webhook signatures"
    )

    github_token: str = Field(
        default="",
        description="GitHub personal access token for posting comments"
    )

    gitlab_token: str = Field(
        default="",
        description="GitLab personal access token (for future support)"
    )

    # ============================================================================
    # Database Configuration
    # ============================================================================

    database_url: str = Field(
        default="sqlite:///./codeproject.db",
        description="Database connection URL (sqlite:// or postgresql://)"
    )

    # ============================================================================
    # Server Configuration
    # ============================================================================

    host: str = Field(
        default="0.0.0.0",
        description="Server host to bind to"
    )

    port: int = Field(
        default=8000,
        description="Server port to listen on"
    )

    # ============================================================================
    # Logging Configuration
    # ============================================================================

    log_level: str = Field(
        default="INFO",
        description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )

    # ============================================================================
    # Validation Methods
    # ============================================================================

    @field_validator("llm_provider")
    @classmethod
    def validate_llm_provider(cls, v: str) -> str:
        """Validate that LLM provider is either 'claude' or 'ollama'."""
        if v not in ["claude", "ollama"]:
            raise ValueError("llm_provider must be 'claude' or 'ollama'")
        return v

    @field_validator("database_url")
    @classmethod
    def validate_database_url(cls, v: str) -> str:
        """Validate that database URL uses supported schemes."""
        if not (v.startswith("sqlite://") or v.startswith("postgresql://") or v.startswith("postgres://")):
            raise ValueError(
                "database_url must start with 'sqlite://', 'postgresql://', or 'postgres://'"
            )
        return v

    @field_validator("port")
    @classmethod
    def validate_port(cls, v: int) -> int:
        """Validate that port is within valid range."""
        if not (1 <= v <= 65535):
            raise ValueError("port must be between 1 and 65535")
        return v

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Validate that log level is one of the standard levels."""
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    # ============================================================================
    # Pydantic Settings Configuration
    # ============================================================================

    model_config = ConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # Ignore unknown environment variables
    )


# Global settings instance
settings = Settings()
