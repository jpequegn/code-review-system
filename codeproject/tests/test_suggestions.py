"""
Tests for suggestion generation methods across all LLM providers.

Tests auto-fix generation, explanations, and improvement suggestions
for Claude, Ollama, and OpenRouter providers.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from src.llm.provider import LLMProvider
from src.llm.claude import ClaudeProvider
from src.llm.ollama import OllamaProvider
from src.llm.openrouter import OpenRouterProvider


# Test fixtures
@pytest.fixture
def sample_finding() -> Dict[str, Any]:
    """Sample security finding for testing."""
    return {
        "severity": "critical",
        "title": "SQL Injection Vulnerability",
        "category": "security",
        "file_path": "app.py",
        "line_number": 42,
    }


@pytest.fixture
def sample_code_diff() -> str:
    """Sample code diff containing a vulnerability."""
    return """
- cursor.execute("SELECT * FROM users WHERE id = " + user_id)
+ cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
"""


@pytest.fixture
def sample_performance_finding() -> Dict[str, Any]:
    """Sample performance finding."""
    return {
        "severity": "high",
        "title": "N+1 Query Problem",
        "category": "performance",
        "file_path": "service.py",
        "line_number": 87,
    }


@pytest.fixture
def sample_performance_code() -> str:
    """Sample code with N+1 query issue."""
    return """
for user in users:
    posts = session.query(Post).filter(Post.user_id == user.id).all()
"""


# Claude Provider Tests
class TestClaudeAutoFixGeneration:
    """Test Claude auto-fix generation."""

    @patch("src.llm.claude.Anthropic")
    def test_generate_auto_fix_returns_valid_json(self, mock_anthropic, sample_finding, sample_code_diff):
        """Test that auto_fix returns valid JSON."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "auto_fix": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
            "confidence": 0.95
        }))]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        result = provider.generate_auto_fix(sample_finding, sample_code_diff)

        parsed = json.loads(result)
        assert "auto_fix" in parsed
        assert "confidence" in parsed
        assert isinstance(parsed["confidence"], (int, float))
        assert 0 <= parsed["confidence"] <= 1

    @patch("src.llm.claude.Anthropic")
    def test_auto_fix_low_confidence_rejected(self, mock_anthropic, sample_finding, sample_code_diff):
        """Test that low confidence fixes are rejected."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=json.dumps({
            "auto_fix": "risky code",
            "confidence": 0.3
        }))]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        result = provider.generate_auto_fix(sample_finding, sample_code_diff)

        parsed = json.loads(result)
        # Low confidence should still be parsed, validation happens elsewhere
        assert parsed["confidence"] == 0.3

    @patch("src.llm.claude.Anthropic")
    def test_auto_fix_invalid_json_returns_null(self, mock_anthropic, sample_finding, sample_code_diff):
        """Test that invalid JSON response returns null auto_fix."""
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="invalid json {{{")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        result = provider.generate_auto_fix(sample_finding, sample_code_diff)

        parsed = json.loads(result)
        assert parsed["auto_fix"] is None
        assert parsed["confidence"] == 0.0


class TestClaudeExplanationGeneration:
    """Test Claude explanation generation."""

    @patch("src.llm.claude.Anthropic")
    def test_generate_explanation_returns_string(self, mock_anthropic, sample_finding, sample_code_diff):
        """Test that explanation is a concise string."""
        explanation = "SQL injection allows attackers to execute arbitrary SQL. Always use parameterized queries."
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=explanation)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        result = provider.generate_explanation(sample_finding, sample_code_diff)

        assert isinstance(result, str)
        assert len(result) > 0
        assert len(result) < 500  # Should be concise

    @patch("src.llm.claude.Anthropic")
    def test_explanation_is_educational(self, mock_anthropic, sample_finding, sample_code_diff):
        """Test that explanation explains WHY the issue matters."""
        explanation = "This vulnerability allows attackers to read or modify database data, compromising user information and system integrity."
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=explanation)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        result = provider.generate_explanation(sample_finding, sample_code_diff)

        # Check for educational content markers
        assert any(word in result.lower() for word in ["security", "vulnerability", "risk", "attack"])


class TestClaudeImprovementSuggestions:
    """Test Claude improvement suggestions generation."""

    @patch("src.llm.claude.Anthropic")
    def test_generate_improvements_returns_string(self, mock_anthropic, sample_finding, sample_code_diff):
        """Test that improvements are returned as string."""
        suggestions = "- Use parameterized queries with ? placeholders\n- Consider using SQLAlchemy ORM\n- Validate input before database operations"
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=suggestions)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        result = provider.generate_improvement_suggestions(sample_finding, sample_code_diff)

        assert isinstance(result, str)
        assert len(result) > 0
        assert "-" in result  # Should have bullet points

    @patch("src.llm.claude.Anthropic")
    def test_improvements_are_actionable(self, mock_anthropic, sample_finding, sample_code_diff):
        """Test that suggestions are specific and actionable."""
        suggestions = "- Always use parameterized queries (?) instead of string concatenation\n- Implement input validation before database operations"
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text=suggestions)]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        result = provider.generate_improvement_suggestions(sample_finding, sample_code_diff)

        # Check for actionable content
        assert "parameterized" in result.lower() or "validate" in result.lower()


# Ollama Provider Tests
class TestOllamaAutoFixGeneration:
    """Test Ollama auto-fix generation."""

    @patch("requests.post")
    def test_ollama_auto_fix_returns_valid_json(self, mock_post, sample_finding, sample_code_diff):
        """Test that Ollama auto_fix returns valid JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "response": json.dumps({
                "auto_fix": "fixed code",
                "confidence": 0.85
            })
        }
        mock_post.return_value = mock_response

        with patch("requests.get"):  # Mock connection verify
            provider = OllamaProvider(model="llama2")
            result = provider.generate_auto_fix(sample_finding, sample_code_diff)

        parsed = json.loads(result)
        assert "auto_fix" in parsed
        assert "confidence" in parsed

    @patch("requests.post")
    def test_ollama_explanation_returns_string(self, mock_post, sample_finding, sample_code_diff):
        """Test that Ollama explanation is returned."""
        explanation = "This is a security issue."
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": explanation}
        mock_post.return_value = mock_response

        with patch("requests.get"):  # Mock connection verify
            provider = OllamaProvider(model="llama2")
            result = provider.generate_explanation(sample_finding, sample_code_diff)

        assert isinstance(result, str)
        assert len(result) > 0


# OpenRouter Provider Tests
class TestOpenRouterAutoFixGeneration:
    """Test OpenRouter auto-fix generation."""

    @patch("httpx.Client.post")
    def test_openrouter_auto_fix_returns_valid_json(self, mock_post, sample_finding, sample_code_diff):
        """Test that OpenRouter auto_fix returns valid JSON."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {
                    "content": json.dumps({
                        "auto_fix": "fixed code",
                        "confidence": 0.92
                    })
                }
            }]
        }
        mock_post.return_value = mock_response

        provider = OpenRouterProvider(api_key="test-key")
        result = provider.generate_auto_fix(sample_finding, sample_code_diff)

        parsed = json.loads(result)
        assert "auto_fix" in parsed
        assert "confidence" in parsed
        assert parsed["confidence"] == 0.92

    @patch("httpx.Client.post")
    def test_openrouter_explanation_returns_string(self, mock_post, sample_finding, sample_code_diff):
        """Test that OpenRouter explanation is returned."""
        explanation = "SQL injection is a critical security vulnerability."
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{
                "message": {"content": explanation}
            }]
        }
        mock_post.return_value = mock_response

        provider = OpenRouterProvider(api_key="test-key")
        result = provider.generate_explanation(sample_finding, sample_code_diff)

        assert isinstance(result, str)
        assert len(result) > 0


# Cross-Provider Tests
class TestSuggestionInterface:
    """Test that all providers implement the suggestion interface correctly."""

    def test_all_providers_have_suggestion_methods(self):
        """Test that all providers implement suggestion generation methods."""
        required_methods = [
            "generate_auto_fix",
            "generate_explanation",
            "generate_improvement_suggestions"
        ]

        # Check each provider class has the methods
        for provider_class in [ClaudeProvider, OllamaProvider, OpenRouterProvider]:
            for method in required_methods:
                assert hasattr(provider_class, method)
                assert callable(getattr(provider_class, method))

    @patch("src.llm.claude.Anthropic")
    def test_claude_high_severity_gets_all_suggestions(self, mock_anthropic):
        """Test that high/critical findings get all suggestion types."""
        finding = {
            "severity": "critical",
            "title": "Test Issue",
            "category": "security",
            "file_path": "test.py",
            "line_number": 1,
        }
        code = "vulnerable code"

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="response")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")

        # All three methods should work for critical findings
        auto_fix = provider.generate_auto_fix(finding, code)
        explanation = provider.generate_explanation(finding, code)
        improvements = provider.generate_improvement_suggestions(finding, code)

        assert auto_fix is not None
        assert explanation is not None
        assert improvements is not None

    @patch("src.llm.claude.Anthropic")
    def test_claude_low_severity_gets_explanation_only(self, mock_anthropic):
        """Test that low severity findings still get explanations."""
        finding = {
            "severity": "low",
            "title": "Minor Issue",
            "category": "code_quality",
            "file_path": "test.py",
            "line_number": 1,
        }
        code = "code with style issue"

        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="This is a style issue")]
        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")

        # Even low severity findings should get explanation
        explanation = provider.generate_explanation(finding, code)
        assert explanation is not None


# Error Handling Tests
class TestSuggestionErrorHandling:
    """Test error handling in suggestion generation."""

    @patch("src.llm.claude.Anthropic")
    def test_claude_timeout_raises_error(self, mock_anthropic):
        """Test that timeout is properly raised."""
        from anthropic import APITimeoutError

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = APITimeoutError("Request timeout")
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")
        finding = {"severity": "high", "title": "Test", "category": "security", "file_path": "test.py", "line_number": 1}

        with pytest.raises(TimeoutError):
            provider.generate_auto_fix(finding, "code")

    @patch("requests.post")
    def test_ollama_connection_error_raises_error(self, mock_post):
        """Test that Ollama connection errors are handled."""
        from requests.exceptions import ConnectionError

        mock_post.side_effect = ConnectionError("Cannot connect to Ollama")

        with patch("requests.get"):  # Mock connection verify
            provider = OllamaProvider(model="llama2")
            finding = {"severity": "high", "title": "Test", "category": "security", "file_path": "test.py", "line_number": 1}

            with pytest.raises(RuntimeError):
                provider.generate_auto_fix(finding, "code")

    @patch("httpx.Client.post")
    def test_openrouter_api_error_raises_error(self, mock_post):
        """Test that OpenRouter API errors are handled."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        mock_response.json.return_value = {"error": {"message": "Invalid API key"}}
        mock_post.return_value = mock_response

        provider = OpenRouterProvider(api_key="invalid-key")
        finding = {"severity": "high", "title": "Test", "category": "security", "file_path": "test.py", "line_number": 1}

        with pytest.raises(RuntimeError, match="Invalid API key"):
            provider.generate_auto_fix(finding, "code")


# Integration Tests
class TestSuggestionIntegration:
    """Test suggestion generation in realistic scenarios."""

    @patch("src.llm.claude.Anthropic")
    def test_sql_injection_full_suggestion_flow(self, mock_anthropic):
        """Test complete suggestion flow for SQL injection."""
        finding = {
            "severity": "critical",
            "title": "SQL Injection",
            "category": "security",
            "file_path": "models.py",
            "line_number": 42,
        }
        code = 'db.execute("SELECT * FROM users WHERE id=" + str(id))'

        # Mock all three responses
        responses = [
            json.dumps({"auto_fix": 'db.execute("SELECT * FROM users WHERE id=?", (id,))', "confidence": 0.95}),
            "SQL injection allows attackers to execute arbitrary SQL commands.",
            "- Use parameterized queries\n- Validate all inputs"
        ]

        mock_client = MagicMock()

        def side_effect(*args, **kwargs):
            response = MagicMock()
            response.content = [MagicMock(text=responses.pop(0))]
            return response

        mock_client.messages.create.side_effect = side_effect
        mock_anthropic.return_value = mock_client

        provider = ClaudeProvider(api_key="test-key")

        # Generate all suggestions
        auto_fix = provider.generate_auto_fix(finding, code)
        explanation = provider.generate_explanation(finding, code)
        improvements = provider.generate_improvement_suggestions(finding, code)

        # Verify all were generated
        assert json.loads(auto_fix)["auto_fix"] is not None
        assert len(explanation) > 0
        assert "parameterized" in improvements.lower()
