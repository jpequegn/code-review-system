"""
Tests for LLM Provider abstraction.

Tests the abstract interface, provider factory, and individual implementations
(Claude and Ollama) with mocking to avoid external dependencies.
"""

import json
import pytest
from unittest.mock import Mock, patch, MagicMock
from anthropic import APITimeoutError, APIError

from src.llm.provider import LLMProvider, get_llm_provider
from src.llm.claude import ClaudeProvider
from src.llm.ollama import OllamaProvider


# ============================================================================
# Test Data & Fixtures
# ============================================================================

SAMPLE_CODE_DIFF = """
--- a/app.py
+++ b/app.py
@@ -1,5 +1,10 @@
 import sqlite3

+def get_user(username):
+    conn = sqlite3.connect(':memory:')
+    cursor = conn.cursor()
+    cursor.execute(f"SELECT * FROM users WHERE name = '{username}'")
+    return cursor.fetchone()
"""

VALID_SECURITY_RESPONSE = json.dumps({
    "findings": [
        {
            "severity": "critical",
            "title": "SQL Injection Vulnerability",
            "description": "User input directly interpolated into SQL query",
            "file_path": "app.py",
            "line_number": 6,
            "suggested_fix": "Use parameterized queries with cursor.execute(sql, (username,))"
        }
    ]
})

VALID_PERFORMANCE_RESPONSE = json.dumps({
    "findings": [
        {
            "severity": "high",
            "title": "In-Memory Database",
            "description": "Using ':memory:' creates new database on each call",
            "file_path": "app.py",
            "line_number": 3,
            "suggested_fix": "Use persistent database connection or connection pool"
        }
    ]
})

EMPTY_RESPONSE = json.dumps({"findings": []})


@pytest.fixture
def sample_code():
    """Provide sample code for testing."""
    return SAMPLE_CODE_DIFF


# ============================================================================
# Test LLMProvider Abstract Class
# ============================================================================

class TestLLMProviderAbstractClass:
    """Tests for the abstract LLMProvider base class."""

    def test_llm_provider_is_abstract(self):
        """Test that LLMProvider cannot be instantiated directly."""
        with pytest.raises(TypeError):
            LLMProvider()

    def test_llm_provider_requires_analyze_security(self):
        """Test that subclasses must implement analyze_security."""
        class IncompleteProvider(LLMProvider):
            def analyze_performance(self, code_diff: str) -> str:
                return "{}"

        with pytest.raises(TypeError):
            IncompleteProvider()

    def test_llm_provider_requires_analyze_performance(self):
        """Test that subclasses must implement analyze_performance."""
        class IncompleteProvider(LLMProvider):
            def analyze_security(self, code_diff: str) -> str:
                return "{}"

        with pytest.raises(TypeError):
            IncompleteProvider()


# ============================================================================
# Test Response Validation
# ============================================================================

class TestResponseValidation:
    """Tests for LLMProvider.validate_response()."""

    def test_validate_valid_response(self):
        """Test validation of valid response."""
        result = LLMProvider.validate_response(VALID_SECURITY_RESPONSE)
        assert "findings" in result
        assert len(result["findings"]) == 1
        assert result["findings"][0]["severity"] == "critical"

    def test_validate_empty_findings(self):
        """Test validation of response with empty findings."""
        result = LLMProvider.validate_response(EMPTY_RESPONSE)
        assert "findings" in result
        assert result["findings"] == []

    def test_validate_invalid_json(self):
        """Test validation rejects invalid JSON."""
        with pytest.raises(ValueError, match="Invalid JSON"):
            LLMProvider.validate_response("not valid json")

    def test_validate_missing_findings_key(self):
        """Test validation requires 'findings' key."""
        invalid = json.dumps({"results": []})
        with pytest.raises(ValueError, match="missing 'findings' key"):
            LLMProvider.validate_response(invalid)

    def test_validate_findings_not_list(self):
        """Test validation requires 'findings' to be a list."""
        invalid = json.dumps({"findings": "not a list"})
        with pytest.raises(ValueError, match="'findings' must be a list"):
            LLMProvider.validate_response(invalid)

    def test_validate_missing_required_fields(self):
        """Test validation requires all required fields."""
        invalid = json.dumps({
            "findings": [
                {
                    "severity": "high",
                    "title": "Issue"
                    # Missing description and file_path
                }
            ]
        })
        with pytest.raises(ValueError, match="missing required fields"):
            LLMProvider.validate_response(invalid)

    def test_validate_invalid_severity(self):
        """Test validation rejects invalid severity."""
        invalid = json.dumps({
            "findings": [
                {
                    "severity": "extreme",  # Invalid
                    "title": "Issue",
                    "description": "Test",
                    "file_path": "test.py"
                }
            ]
        })
        with pytest.raises(ValueError, match="invalid severity"):
            LLMProvider.validate_response(invalid)

    def test_validate_all_valid_severities(self):
        """Test validation accepts all valid severities."""
        for severity in ["critical", "high", "medium", "low"]:
            response = json.dumps({
                "findings": [
                    {
                        "severity": severity,
                        "title": "Issue",
                        "description": "Test",
                        "file_path": "test.py"
                    }
                ]
            })
            result = LLMProvider.validate_response(response)
            assert result["findings"][0]["severity"] == severity

    def test_validate_optional_fields(self):
        """Test that line_number and suggested_fix are optional."""
        response = json.dumps({
            "findings": [
                {
                    "severity": "high",
                    "title": "Issue",
                    "description": "Test",
                    "file_path": "test.py"
                    # No line_number or suggested_fix
                }
            ]
        })
        result = LLMProvider.validate_response(response)
        assert len(result["findings"]) == 1


# ============================================================================
# Test Provider Factory
# ============================================================================

class TestGetLLMProvider:
    """Tests for get_llm_provider() factory function."""

    @patch("src.llm.provider.settings")
    def test_factory_returns_claude_provider(self, mock_settings):
        """Test factory returns ClaudeProvider when configured."""
        mock_settings.llm_provider = "claude"
        mock_settings.claude_api_key = "test-key"

        with patch("src.llm.claude.Anthropic"):
            provider = get_llm_provider()
            assert isinstance(provider, ClaudeProvider)

    @patch("src.llm.provider.settings")
    def test_factory_returns_ollama_provider(self, mock_settings):
        """Test factory returns OllamaProvider when configured."""
        mock_settings.llm_provider = "ollama"
        mock_settings.ollama_base_url = "http://localhost:11434"

        with patch("src.llm.ollama.requests.get"):
            provider = get_llm_provider()
            assert isinstance(provider, OllamaProvider)

    @patch("src.llm.provider.settings")
    def test_factory_raises_for_unknown_provider(self, mock_settings):
        """Test factory raises error for unknown provider."""
        mock_settings.llm_provider = "unknown"

        with pytest.raises(ValueError, match="Unknown LLM provider"):
            get_llm_provider()

    @patch("src.llm.provider.settings")
    def test_factory_raises_if_claude_key_missing(self, mock_settings):
        """Test factory raises error if Claude selected but API key missing."""
        mock_settings.llm_provider = "claude"
        mock_settings.claude_api_key = ""

        with pytest.raises(ValueError, match="CLAUDE_API_KEY not configured"):
            get_llm_provider()

    @patch("src.llm.provider.settings")
    def test_factory_case_insensitive(self, mock_settings):
        """Test factory is case-insensitive for provider name."""
        mock_settings.claude_api_key = "test-key"

        with patch("src.llm.claude.Anthropic"):
            # Test uppercase
            mock_settings.llm_provider = "CLAUDE"
            provider = get_llm_provider()
            assert isinstance(provider, ClaudeProvider)

            # Test mixed case
            mock_settings.llm_provider = "Claude"
            provider = get_llm_provider()
            assert isinstance(provider, ClaudeProvider)


# ============================================================================
# Test Claude Provider
# ============================================================================

class TestClaudeProvider:
    """Tests for ClaudeProvider implementation."""

    @patch("src.llm.claude.Anthropic")
    def test_claude_provider_initialization(self, mock_anthropic):
        """Test ClaudeProvider initialization."""
        provider = ClaudeProvider(api_key="test-key")
        assert provider.api_key == "test-key"
        assert provider.model == "claude-3-5-sonnet-20241022"
        assert provider.timeout == 60

    @patch("src.llm.claude.Anthropic")
    def test_claude_provider_custom_config(self, mock_anthropic):
        """Test ClaudeProvider with custom configuration."""
        provider = ClaudeProvider(
            api_key="test-key",
            model="claude-opus",
            timeout=120
        )
        assert provider.model == "claude-opus"
        assert provider.timeout == 120

    @patch("src.llm.claude.Anthropic")
    def test_claude_analyze_security(self, mock_anthropic, sample_code):
        """Test Claude security analysis."""
        # Mock the API response
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=VALID_SECURITY_RESPONSE)]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        result = provider.analyze_security(sample_code)

        assert isinstance(result, str)
        response = LLMProvider.validate_response(result)
        assert len(response["findings"]) > 0

    @patch("src.llm.claude.Anthropic")
    def test_claude_analyze_performance(self, mock_anthropic, sample_code):
        """Test Claude performance analysis."""
        # Mock the API response
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = [MagicMock(text=VALID_PERFORMANCE_RESPONSE)]
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        result = provider.analyze_performance(sample_code)

        assert isinstance(result, str)
        response = LLMProvider.validate_response(result)
        assert len(response["findings"]) > 0

    @patch("src.llm.claude.Anthropic")
    def test_claude_timeout_error(self, mock_anthropic):
        """Test Claude provider handles timeout errors."""
        # Mock timeout error using side_effect with an instance
        mock_client = MagicMock()
        timeout_error = APITimeoutError.__new__(APITimeoutError)
        mock_client.messages.create.side_effect = timeout_error
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        with pytest.raises(TimeoutError):
            provider.analyze_security("code")

    @patch("src.llm.claude.Anthropic")
    def test_claude_api_error(self, mock_anthropic):
        """Test Claude provider handles API errors."""
        # Mock API error using a generic exception
        mock_client = MagicMock()
        api_error = APIError.__new__(APIError)
        mock_client.messages.create.side_effect = api_error
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        with pytest.raises(RuntimeError, match="Claude API error"):
            provider.analyze_security("code")

    @patch("src.llm.claude.Anthropic")
    def test_claude_empty_response(self, mock_anthropic):
        """Test Claude provider handles empty response."""
        # Mock empty response
        mock_client = MagicMock()
        mock_message = MagicMock()
        mock_message.content = []
        mock_client.messages.create.return_value = mock_message
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        with pytest.raises(RuntimeError, match="empty response"):
            provider.analyze_security("code")


# ============================================================================
# Test Ollama Provider
# ============================================================================

class TestOllamaProvider:
    """Tests for OllamaProvider implementation."""

    @patch("src.llm.ollama.requests.get")
    def test_ollama_provider_initialization(self, mock_get):
        """Test OllamaProvider initialization and connection verification."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        provider = OllamaProvider(base_url="http://localhost:11434")
        assert provider.base_url == "http://localhost:11434"
        assert provider.model == "llama2"
        assert provider.timeout == 120

    @patch("src.llm.ollama.requests.get")
    def test_ollama_provider_custom_config(self, mock_get):
        """Test OllamaProvider with custom configuration."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"models": []}
        mock_get.return_value = mock_response

        provider = OllamaProvider(
            base_url="http://custom:8000",
            model="mistral",
            timeout=180
        )
        assert provider.base_url == "http://custom:8000"
        assert provider.model == "mistral"
        assert provider.timeout == 180

    @patch("src.llm.ollama.requests.get")
    def test_ollama_connection_failure(self, mock_get):
        """Test OllamaProvider raises error when Ollama is not running."""
        from requests.exceptions import ConnectionError
        mock_get.side_effect = ConnectionError("Connection refused")

        with pytest.raises(RuntimeError, match="Cannot connect to Ollama"):
            OllamaProvider(base_url="http://localhost:11434")

    @patch("src.llm.ollama.requests.post")
    @patch("src.llm.ollama.requests.get")
    def test_ollama_analyze_security(self, mock_get, mock_post, sample_code):
        """Test Ollama security analysis."""
        # Mock connection check
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # Mock analysis response
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"response": VALID_SECURITY_RESPONSE}
        mock_post.return_value = mock_post_response

        provider = OllamaProvider()
        result = provider.analyze_security(sample_code)

        assert isinstance(result, str)
        response = LLMProvider.validate_response(result)
        assert len(response["findings"]) > 0

    @patch("src.llm.ollama.requests.post")
    @patch("src.llm.ollama.requests.get")
    def test_ollama_analyze_performance(self, mock_get, mock_post, sample_code):
        """Test Ollama performance analysis."""
        # Mock connection check
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # Mock analysis response
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"response": VALID_PERFORMANCE_RESPONSE}
        mock_post.return_value = mock_post_response

        provider = OllamaProvider()
        result = provider.analyze_performance(sample_code)

        assert isinstance(result, str)
        response = LLMProvider.validate_response(result)
        assert len(response["findings"]) > 0

    @patch("src.llm.ollama.requests.post")
    @patch("src.llm.ollama.requests.get")
    def test_ollama_timeout_error(self, mock_get, mock_post):
        """Test Ollama provider handles timeout errors."""
        from requests.exceptions import Timeout

        # Mock connection check
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # Mock timeout
        mock_post.side_effect = Timeout("Request timeout")

        provider = OllamaProvider()
        with pytest.raises(TimeoutError):
            provider.analyze_security("code")

    @patch("src.llm.ollama.requests.post")
    @patch("src.llm.ollama.requests.get")
    def test_ollama_connection_error(self, mock_get, mock_post):
        """Test Ollama provider handles connection errors."""
        from requests.exceptions import ConnectionError

        # Mock connection check
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # Mock connection error during analysis
        mock_post.side_effect = ConnectionError("Connection lost")

        provider = OllamaProvider()
        with pytest.raises(RuntimeError, match="Connection error"):
            provider.analyze_security("code")

    @patch("src.llm.ollama.requests.post")
    @patch("src.llm.ollama.requests.get")
    def test_ollama_http_error(self, mock_get, mock_post):
        """Test Ollama provider handles HTTP errors."""
        import requests

        # Mock connection check
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # Mock HTTP error
        mock_post_response = MagicMock()
        mock_post_response.raise_for_status.side_effect = requests.exceptions.HTTPError("500")
        mock_post.return_value = mock_post_response

        provider = OllamaProvider()
        with pytest.raises(RuntimeError, match="HTTP error"):
            provider.analyze_security("code")

    @patch("src.llm.ollama.requests.post")
    @patch("src.llm.ollama.requests.get")
    def test_ollama_unexpected_response_format(self, mock_get, mock_post):
        """Test Ollama provider handles unexpected response format."""
        # Mock connection check
        mock_get_response = MagicMock()
        mock_get_response.json.return_value = {"models": []}
        mock_get.return_value = mock_get_response

        # Mock unexpected response format
        mock_post_response = MagicMock()
        mock_post_response.json.return_value = {"unexpected": "format"}
        mock_post.return_value = mock_post_response

        provider = OllamaProvider()
        with pytest.raises(RuntimeError, match="unexpected response format"):
            provider.analyze_security("code")
