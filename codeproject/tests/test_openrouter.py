"""
Tests for OpenRouter LLM Provider.

Tests OpenRouter API integration, error handling, and response parsing.
"""

import pytest
import json
from unittest.mock import patch, MagicMock

from src.llm.openrouter import OpenRouterProvider


class TestOpenRouterInitialization:
    """Test OpenRouter provider initialization."""

    def test_initialize_with_valid_api_key(self):
        """Test initialization with valid API key."""
        provider = OpenRouterProvider(api_key="sk-or-v1-test-key")
        assert provider.api_key == "sk-or-v1-test-key"
        assert provider.model == "anthropic/claude-3.5-sonnet"
        assert provider.base_url == "https://openrouter.ai/api/v1"

    def test_initialize_with_empty_api_key(self):
        """Test initialization with empty API key raises error."""
        with pytest.raises(ValueError, match="API key cannot be empty"):
            OpenRouterProvider(api_key="")

    def test_initialize_with_custom_model_full_path(self):
        """Test initialization with custom model using full path."""
        provider = OpenRouterProvider(
            api_key="sk-or-v1-test-key",
            model="openai/gpt-4",
        )
        assert provider.model == "openai/gpt-4"

    def test_initialize_with_custom_model_shorthand(self):
        """Test initialization with custom model using shorthand."""
        provider = OpenRouterProvider(
            api_key="sk-or-v1-test-key",
            model="gpt-4",
        )
        assert provider.model == "openai/gpt-4"

    def test_get_available_models(self):
        """Test getting available model shortcuts."""
        models = OpenRouterProvider.get_available_models()
        assert "claude-3.5-sonnet" in models
        assert "gpt-4" in models
        assert "mixtral" in models


class TestSecurityAnalysis:
    """Test security analysis functionality."""

    @patch("httpx.Client.post")
    def test_analyze_security_success(self, mock_post):
        """Test successful security analysis."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "findings": [
                                    {
                                        "severity": "high",
                                        "title": "SQL Injection",
                                        "description": "Vulnerable SQL query",
                                        "file_path": "app.py",
                                        "line_number": 42,
                                        "suggested_fix": "Use parameterized queries",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        provider = OpenRouterProvider(api_key="sk-or-v1-test-key")
        result = provider.analyze_security("vulnerable code")

        parsed = json.loads(result)
        assert "findings" in parsed
        assert len(parsed["findings"]) > 0


class TestPerformanceAnalysis:
    """Test performance analysis functionality."""

    @patch("httpx.Client.post")
    def test_analyze_performance_success(self, mock_post):
        """Test successful performance analysis."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {
                                "findings": [
                                    {
                                        "severity": "high",
                                        "title": "N+1 Query Problem",
                                        "description": "Loop with database queries",
                                        "file_path": "query.py",
                                        "line_number": 15,
                                        "suggested_fix": "Use JOIN instead",
                                    }
                                ]
                            }
                        )
                    }
                }
            ]
        }
        mock_post.return_value = mock_response

        provider = OpenRouterProvider(api_key="sk-or-v1-test-key")
        result = provider.analyze_performance("inefficient code")

        parsed = json.loads(result)
        assert "findings" in parsed


class TestErrorHandling:
    """Test error handling."""

    @patch("httpx.Client.post")
    def test_timeout_error(self, mock_post):
        """Test timeout error handling."""
        import httpx

        mock_post.side_effect = httpx.TimeoutException("Request timeout")

        provider = OpenRouterProvider(api_key="sk-or-v1-test-key")

        with pytest.raises(TimeoutError, match="timed out"):
            provider.analyze_security("code")

    @patch("httpx.Client.post")
    def test_api_error(self, mock_post):
        """Test API error handling."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {
            "error": {"message": "Invalid API key"}
        }
        mock_post.return_value = mock_response

        provider = OpenRouterProvider(api_key="sk-or-v1-invalid")

        with pytest.raises(RuntimeError, match="Invalid API key"):
            provider.analyze_security("code")


class TestConfigIntegration:
    """Test integration with configuration system."""

    def test_provider_factory_with_openrouter(self):
        """Test that provider factory creates OpenRouter provider."""
        from src.llm.provider import get_llm_provider
        from src.config import settings

        original_provider = settings.llm_provider
        original_key = settings.openrouter_api_key

        try:
            settings.llm_provider = "openrouter"
            settings.openrouter_api_key = "sk-or-v1-test-key"

            provider = get_llm_provider()
            assert isinstance(provider, OpenRouterProvider)

        finally:
            settings.llm_provider = original_provider
            settings.openrouter_api_key = original_key

    def test_provider_factory_missing_api_key(self):
        """Test that factory raises error when API key is missing."""
        from src.llm.provider import get_llm_provider
        from src.config import settings

        original_provider = settings.llm_provider
        original_key = settings.openrouter_api_key

        try:
            settings.llm_provider = "openrouter"
            settings.openrouter_api_key = ""

            with pytest.raises(ValueError, match="OPENROUTER_API_KEY not configured"):
                get_llm_provider()

        finally:
            settings.llm_provider = original_provider
            settings.openrouter_api_key = original_key
