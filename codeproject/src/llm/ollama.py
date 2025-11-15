"""
Ollama LLM Provider Implementation

Connects to local Ollama instance for running open-source models locally.
Useful for development and testing without API costs.
"""

import json
import logging
from typing import Optional

import requests
from requests.exceptions import RequestException, Timeout, ConnectionError

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


class OllamaProvider(LLMProvider):
    """
    Ollama LLM Provider for local model inference.

    Connects to a local Ollama instance for code analysis.
    Supports any model installed in Ollama (llama2, mistral, etc.)
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama2",
        timeout: int = 120,
    ):
        """
        Initialize Ollama provider.

        Args:
            base_url: Base URL of Ollama instance (default: http://localhost:11434)
            model: Model name to use (default: llama2)
            timeout: Request timeout in seconds (default: 120, as local inference is slower)
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout

        # Verify connection to Ollama
        self._verify_connection()

    def _verify_connection(self) -> None:
        """
        Verify that Ollama is accessible.

        Raises:
            RuntimeError: If Ollama is not running or accessible
        """
        try:
            response = requests.get(
                f"{self.base_url}/api/tags",
                timeout=5,
            )
            response.raise_for_status()
            logger.info(f"Connected to Ollama at {self.base_url}")
        except ConnectionError as e:
            raise RuntimeError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Is Ollama running? Error: {str(e)}"
            )
        except RequestException as e:
            raise RuntimeError(
                f"Error connecting to Ollama: {str(e)}"
            )

    def analyze_security(self, code_diff: str) -> str:
        """
        Analyze code for security vulnerabilities using Ollama.

        Args:
            code_diff: Git diff or code snippet to analyze

        Returns:
            JSON string with security findings

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If Ollama returns an error
        """
        prompt = SECURITY_ANALYSIS_PROMPT.format(code_diff=code_diff)
        return self._call_ollama(prompt)

    def analyze_performance(self, code_diff: str) -> str:
        """
        Analyze code for performance issues using Ollama.

        Args:
            code_diff: Git diff or code snippet to analyze

        Returns:
            JSON string with performance findings

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If Ollama returns an error
        """
        prompt = PERFORMANCE_ANALYSIS_PROMPT.format(code_diff=code_diff)
        return self._call_ollama(prompt)

    def _call_ollama(self, prompt: str) -> str:
        """
        Make a request to Ollama API.

        Args:
            prompt: The prompt to send to Ollama

        Returns:
            Ollama's response text

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If Ollama returns an error
        """
        try:
            url = f"{self.base_url}/api/generate"

            # Stream response from Ollama
            response = requests.post(
                url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "temperature": 0.3,  # Lower temperature for more consistent output
                },
                timeout=self.timeout,
            )

            response.raise_for_status()

            # Parse response
            data = response.json()

            if "response" in data:
                response_text = data["response"]
                logger.debug(f"Ollama response: {response_text[:200]}...")
                return response_text
            else:
                raise RuntimeError("Ollama returned unexpected response format")

        except Timeout:
            logger.error(f"Ollama request timeout after {self.timeout}s")
            raise TimeoutError(
                f"Ollama analysis timed out after {self.timeout}s. "
                f"Try using a smaller code snippet or increasing timeout."
            )

        except ConnectionError as e:
            logger.error(f"Connection error to Ollama: {str(e)}")
            raise RuntimeError(
                f"Connection error to Ollama at {self.base_url}: {str(e)}"
            )

        except requests.exceptions.HTTPError as e:
            logger.error(f"Ollama HTTP error: {str(e)}")
            raise RuntimeError(f"Ollama HTTP error: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error calling Ollama: {str(e)}")
            raise RuntimeError(f"Unexpected error: {str(e)}")
