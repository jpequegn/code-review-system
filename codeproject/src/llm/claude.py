"""
Claude LLM Provider Implementation

Uses Anthropic SDK to interact with Claude models for code analysis.
Supports streaming for long-running analyses.
"""

import json
import logging
from typing import Optional

from anthropic import Anthropic, APIError, APITimeoutError

from src.llm.provider import LLMProvider

logger = logging.getLogger(__name__)


# Security Analysis Prompt
SECURITY_ANALYSIS_PROMPT = """Analyze the following code changes for security vulnerabilities.

Look for common security issues such as:
- SQL injection vulnerabilities
- Cross-site scripting (XSS) vulnerabilities
- Authentication/authorization issues
- Sensitive data exposure
- Insecure cryptography
- Hardcoded secrets or credentials
- Buffer overflows or memory safety issues
- Command injection vulnerabilities
- Path traversal vulnerabilities
- Insecure deserialization

Code to analyze:
{code_diff}

Return a JSON object with this structure:
{{
    "findings": [
        {{
            "severity": "critical|high|medium|low",
            "title": "Brief vulnerability title",
            "description": "Detailed explanation of the vulnerability",
            "file_path": "path/to/file.ext",
            "line_number": 42,
            "suggested_fix": "How to fix this vulnerability"
        }}
    ]
}}

If no vulnerabilities found, return {{"findings": []}}
Only return valid JSON, no additional text."""


# Performance Analysis Prompt
PERFORMANCE_ANALYSIS_PROMPT = """Analyze the following code changes for performance and scalability issues.

Look for:
- N+1 query problems
- Inefficient algorithms (wrong complexity)
- Memory leaks or excessive memory usage
- Blocking I/O in async contexts
- Unnecessary database queries
- Missing indexes
- Inefficient data structure usage
- Unnecessary copying of large objects
- Unbounded loops or recursion
- Inefficient string/list operations
- Missing caching opportunities
- Inefficient HTTP requests

Code to analyze:
{code_diff}

Return a JSON object with this structure:
{{
    "findings": [
        {{
            "severity": "critical|high|medium|low",
            "title": "Brief performance issue title",
            "description": "Detailed explanation of the performance problem",
            "file_path": "path/to/file.ext",
            "line_number": 42,
            "suggested_fix": "How to optimize this code"
        }}
    ]
}}

If no performance issues found, return {{"findings": []}}
Only return valid JSON, no additional text."""


class ClaudeProvider(LLMProvider):
    """
    Claude LLM Provider using Anthropic API.

    Implements code analysis using Claude models via the Anthropic SDK.
    Supports different Claude models with configurable temperature.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-3-5-sonnet-20241022",
        timeout: int = 60,
    ):
        """
        Initialize Claude provider.

        Args:
            api_key: Anthropic API key
            model: Claude model ID (default: claude-3-5-sonnet-20241022)
            timeout: Request timeout in seconds (default: 60)
        """
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.client = Anthropic(api_key=api_key)

    def analyze_security(self, code_diff: str) -> str:
        """
        Analyze code for security vulnerabilities using Claude.

        Args:
            code_diff: Git diff or code snippet to analyze

        Returns:
            JSON string with security findings

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If API returns an error
        """
        prompt = SECURITY_ANALYSIS_PROMPT.format(code_diff=code_diff)
        return self._call_claude(prompt)

    def analyze_performance(self, code_diff: str) -> str:
        """
        Analyze code for performance issues using Claude.

        Args:
            code_diff: Git diff or code snippet to analyze

        Returns:
            JSON string with performance findings

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If API returns an error
        """
        prompt = PERFORMANCE_ANALYSIS_PROMPT.format(code_diff=code_diff)
        return self._call_claude(prompt)

    def _call_claude(self, prompt: str) -> str:
        """
        Make a request to Claude API.

        Args:
            prompt: The prompt to send to Claude

        Returns:
            Claude's response text

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If API returns an error
        """
        try:
            message = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                timeout=self.timeout,
            )

            # Extract text from response
            if message.content and len(message.content) > 0:
                response_text = message.content[0].text
                logger.debug(f"Claude response: {response_text[:200]}...")
                return response_text
            else:
                raise RuntimeError("Claude returned empty response")

        except APITimeoutError as e:
            logger.error(f"Claude API timeout after {self.timeout}s: {str(e)}")
            raise TimeoutError(f"Claude analysis timed out: {str(e)}")

        except APIError as e:
            logger.error(f"Claude API error: {str(e)}")
            raise RuntimeError(f"Claude API error: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error calling Claude: {str(e)}")
            raise RuntimeError(f"Unexpected error: {str(e)}")
