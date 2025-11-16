"""
Abstract LLM Provider Interface

Defines the interface that all LLM providers must implement.
Provides factory for creating provider instances based on configuration.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
import json
import logging

from src.config import settings

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """
    Abstract base class for LLM providers.

    All LLM providers (Claude, Ollama, etc.) must implement this interface
    to be compatible with the code review system.
    """

    @abstractmethod
    def analyze_security(self, code_diff: str) -> str:
        """
        Analyze code for security vulnerabilities.

        Args:
            code_diff: Git diff or code snippet to analyze

        Returns:
            JSON string containing array of findings with structure:
            {
                "findings": [
                    {
                        "severity": "critical|high|medium|low",
                        "title": "Issue title",
                        "description": "Detailed explanation",
                        "file_path": "path/to/file.py",
                        "line_number": 42,
                        "suggested_fix": "How to fix it"
                    }
                ]
            }

        Raises:
            TimeoutError: If analysis takes too long
            RuntimeError: If provider is unavailable
        """
        pass

    @abstractmethod
    def analyze_performance(self, code_diff: str) -> str:
        """
        Analyze code for performance and scalability issues.

        Args:
            code_diff: Git diff or code snippet to analyze

        Returns:
            JSON string containing array of findings with structure:
            {
                "findings": [
                    {
                        "severity": "critical|high|medium|low",
                        "title": "Issue title",
                        "description": "Detailed explanation",
                        "file_path": "path/to/file.py",
                        "line_number": 42,
                        "suggested_fix": "How to optimize"
                    }
                ]
            }

        Raises:
            TimeoutError: If analysis takes too long
            RuntimeError: If provider is unavailable
        """
        pass

    def analyze_security_with_context(
        self, code_diff: str, context_prompt: str
    ) -> str:
        """
        Analyze code for security vulnerabilities with codebase context.

        Args:
            code_diff: Git diff or code snippet to analyze
            context_prompt: Additional context about the codebase

        Returns:
            JSON string with security findings

        Note:
            Default implementation calls analyze_security.
            Subclasses can override for context-aware implementations.
        """
        return self.analyze_security(code_diff)

    def analyze_performance_with_context(
        self, code_diff: str, context_prompt: str
    ) -> str:
        """
        Analyze code for performance issues with codebase context.

        Args:
            code_diff: Git diff or code snippet to analyze
            context_prompt: Additional context about the codebase

        Returns:
            JSON string with performance findings

        Note:
            Default implementation calls analyze_performance.
            Subclasses can override for context-aware implementations.
        """
        return self.analyze_performance(code_diff)

    @staticmethod
    def validate_response(response: str) -> Dict[str, Any]:
        """
        Validate and parse LLM response as JSON.

        Args:
            response: Raw response from LLM

        Returns:
            Parsed JSON response

        Raises:
            ValueError: If response is not valid JSON or missing required fields
        """
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM response as JSON: {e}")
            raise ValueError(f"Invalid JSON response from LLM: {str(e)}")

        # Validate structure
        if "findings" not in data:
            raise ValueError("Response missing 'findings' key")

        if not isinstance(data["findings"], list):
            raise ValueError("'findings' must be a list")

        # Validate each finding
        required_fields = {"severity", "title", "description", "file_path"}
        for idx, finding in enumerate(data["findings"]):
            missing = required_fields - set(finding.keys())
            if missing:
                raise ValueError(f"Finding {idx} missing required fields: {missing}")

            # Validate severity
            valid_severities = {"critical", "high", "medium", "low"}
            if finding["severity"] not in valid_severities:
                raise ValueError(
                    f"Finding {idx} has invalid severity: {finding['severity']}"
                )

        return data


def get_llm_provider() -> LLMProvider:
    """
    Factory function to create LLM provider based on configuration.

    Returns:
        Configured LLM provider instance (ClaudeProvider or OllamaProvider)

    Raises:
        ValueError: If configured provider is not supported
    """
    provider_name = settings.llm_provider.lower()

    if provider_name == "claude":
        from src.llm.claude import ClaudeProvider

        if not settings.claude_api_key:
            raise ValueError(
                "Claude provider selected but CLAUDE_API_KEY not configured"
            )

        return ClaudeProvider(api_key=settings.claude_api_key)

    elif provider_name == "ollama":
        from src.llm.ollama import OllamaProvider

        return OllamaProvider(base_url=settings.ollama_base_url)

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider_name}. " f"Must be 'claude' or 'ollama'"
        )
