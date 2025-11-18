"""
OpenRouter LLM Provider Implementation

Uses OpenRouter API to interact with multiple LLM models (Claude, GPT-4, Mixtral, etc.)
for code analysis. OpenRouter provides a unified interface to many LLM providers.

Supports:
- Model selection across providers (Claude, GPT-4, Mixtral, etc.)
- Cost optimization (compare prices across models)
- Automatic failover between providers
- OpenAI-compatible API format
"""

import logging
import httpx
import json
from typing import Dict, Any

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


class OpenRouterProvider(LLMProvider):
    """
    OpenRouter LLM Provider.

    Uses OpenRouter API to support multiple LLM models (Claude, GPT-4, Mixtral, etc.)
    with a unified interface. OpenRouter provides:
    - Access to dozens of LLM models
    - Cost comparison and optimization
    - Automatic provider failover
    - OpenAI-compatible API format
    """

    # Default models (cost-optimized)
    DEFAULT_MODEL = "anthropic/claude-3.5-sonnet"

    # Recommended models by use case
    MODELS = {
        "claude-3.5-sonnet": "anthropic/claude-3.5-sonnet",  # Balanced - recommended
        "claude-3-opus": "anthropic/claude-3-opus",  # Most capable
        "gpt-4": "openai/gpt-4",  # Alternative
        "gpt-4-turbo": "openai/gpt-4-turbo",  # Faster, cheaper
        "mixtral": "mistralai/mixtral-8x7b",  # Cost effective
        "llama-2": "meta-llama/llama-2-70b-chat",  # Open source
    }

    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        timeout: int = 60,
    ):
        """
        Initialize OpenRouter provider.

        Args:
            api_key: OpenRouter API key (from https://openrouter.ai)
            model: Model identifier (default: anthropic/claude-3.5-sonnet)
                   Can be full path (e.g., "anthropic/claude-3.5-sonnet")
                   or shorthand (e.g., "claude-3.5-sonnet")
            timeout: Request timeout in seconds (default: 60)

        Raises:
            ValueError: If API key is empty
        """
        if not api_key or not api_key.strip():
            raise ValueError("OpenRouter API key cannot be empty")

        self.api_key = api_key
        self.timeout = timeout

        # Resolve model - support both full paths and shortcuts
        if model in self.MODELS:
            self.model = self.MODELS[model]
        else:
            self.model = model

        self.base_url = "https://openrouter.ai/api/v1"
        logger.info(f"Initialized OpenRouter provider with model: {self.model}")

    def analyze_security(self, code_diff: str) -> str:
        """
        Analyze code for security vulnerabilities using OpenRouter.

        Args:
            code_diff: Git diff or code snippet to analyze

        Returns:
            JSON string with security findings

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If API returns an error
        """
        prompt = SECURITY_ANALYSIS_PROMPT.format(code_diff=code_diff)
        return self._call_openrouter(prompt)

    def analyze_performance(self, code_diff: str) -> str:
        """
        Analyze code for performance issues using OpenRouter.

        Args:
            code_diff: Git diff or code snippet to analyze

        Returns:
            JSON string with performance findings

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If API returns an error
        """
        prompt = PERFORMANCE_ANALYSIS_PROMPT.format(code_diff=code_diff)
        return self._call_openrouter(prompt)

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
        """
        from src.llm.enhanced_prompts import enhance_security_prompt

        enhanced = enhance_security_prompt(code_diff, context_prompt=context_prompt)
        return self._call_openrouter(enhanced)

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
        """
        from src.llm.enhanced_prompts import enhance_performance_prompt

        enhanced = enhance_performance_prompt(code_diff, context_prompt=context_prompt)
        return self._call_openrouter(enhanced)

    def generate_auto_fix(self, finding: Dict[str, Any], code_diff: str) -> str:
        """
        Generate a safe, conservative auto-fix for a finding.

        Args:
            finding: Finding dictionary with keys: severity, title, category, file_path, line_number
            code_diff: Git diff or code snippet containing the issue

        Returns:
            JSON string with structure:
            {
                "auto_fix": "fixed code snippet",
                "confidence": 0.95
            }

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If API returns an error
        """
        prompt = f"""Generate a SAFE, CONSERVATIVE auto-fix for this code issue.

ISSUE DETAILS:
Title: {finding.get('title', 'Unknown')}
Severity: {finding.get('severity', 'unknown')}
Category: {finding.get('category', 'unknown')}
File: {finding.get('file_path', 'unknown')}
Line: {finding.get('line_number', '?')}

CODE CONTEXT:
{code_diff}

REQUIREMENTS FOR THE FIX:
1. Fix ONLY the specific issue - no additional refactoring
2. Maintain existing code style and conventions
3. No new external dependencies unless already imported
4. Return valid, runnable code
5. Be conservative - prefer minimal changes
6. Include a confidence score (0.0-1.0) where 1.0 is very confident

RESPONSE FORMAT:
Return ONLY valid JSON (no markdown, no extra text):
{{
    "auto_fix": "the fixed code",
    "confidence": 0.95
}}

If you cannot generate a safe fix, set auto_fix to null and confidence to a low value."""

        try:
            response = self._call_openrouter(prompt)
            # Parse and validate response
            data = json.loads(response)

            # Ensure confidence is a number between 0 and 1
            if "confidence" in data and not isinstance(data["confidence"], (int, float)):
                data["confidence"] = float(data.get("confidence", 0))

            return json.dumps(data)
        except json.JSONDecodeError:
            # Return low confidence if we can't parse response
            logger.warning(f"Failed to parse auto_fix response as JSON: {response[:100]}")
            return json.dumps({"auto_fix": None, "confidence": 0.0})

    def generate_explanation(self, finding: Dict[str, Any], code_diff: str) -> str:
        """
        Generate an educational explanation of why this finding matters.

        Args:
            finding: Finding dictionary with keys: severity, title, category, file_path, line_number
            code_diff: Git diff or code snippet containing the issue

        Returns:
            String with 2-3 sentence explanation

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If API returns an error
        """
        prompt = f"""Explain this code issue in 2-3 sentences for a developer.

ISSUE:
Title: {finding.get('title', 'Unknown')}
Category: {finding.get('category', 'unknown')}
Severity: {finding.get('severity', 'unknown')}

CODE:
{code_diff}

EXPLANATION REQUIREMENTS:
1. Keep it concise (2-3 sentences, <200 chars)
2. Explain WHY it matters
3. Be educational and actionable
4. Avoid overly technical jargon
5. Make it clear what the impact is

Return ONLY the explanation text, no JSON or markdown."""

        try:
            response = self._call_openrouter(prompt)
            # Clean up response
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating explanation: {e}")
            raise

    def generate_improvement_suggestions(self, finding: Dict[str, Any], code_diff: str) -> str:
        """
        Generate best practices and improvement suggestions for a finding.

        Args:
            finding: Finding dictionary with keys: severity, title, category, file_path, line_number
            code_diff: Git diff or code snippet containing the issue

        Returns:
            String with 2-3 bullet-point suggestions

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If API returns an error
        """
        prompt = f"""Generate 2-3 best practice suggestions for this code issue.

ISSUE:
Title: {finding.get('title', 'Unknown')}
Category: {finding.get('category', 'unknown')}
Severity: {finding.get('severity', 'unknown')}

CODE:
{code_diff}

SUGGESTION REQUIREMENTS:
1. Provide 2-3 actionable best practices
2. Reference specific patterns, libraries, or approaches
3. Be specific and non-generic
4. Focus on prevention and best practices
5. Format as bullet points with "-" prefix

Return ONLY the suggestions, one bullet point per line.
Example format:
- Use parameterized queries with database drivers
- Consider using an ORM like SQLAlchemy
- Always validate user input before database operations"""

        try:
            response = self._call_openrouter(prompt)
            # Clean up response
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            raise

    def _call_openrouter(self, prompt: str) -> str:
        """
        Make a request to OpenRouter API.

        Uses OpenAI-compatible chat completions format.

        Args:
            prompt: The prompt to send to the model

        Returns:
            Model's response text

        Raises:
            TimeoutError: If request exceeds timeout
            RuntimeError: If API returns an error
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "HTTP-Referer": "https://github.com/jpequegn/code-review-system",
            "X-Title": "Code Review System",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            "max_tokens": 2048,
            "temperature": 0.7,
        }

        try:
            with httpx.Client(timeout=self.timeout) as client:
                response = client.post(
                    f"{self.base_url}/chat/completions",
                    headers=headers,
                    json=payload,
                )

                # Check for HTTP errors
                if response.status_code != 200:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get("error", {}).get(
                        "message", f"HTTP {response.status_code}"
                    )
                    logger.error(f"OpenRouter API error: {error_msg}")
                    raise RuntimeError(f"OpenRouter API error: {error_msg}")

                # Parse response
                response_data = response.json()

                # Extract text from response
                if (
                    "choices" in response_data
                    and len(response_data["choices"]) > 0
                    and "message" in response_data["choices"][0]
                ):
                    response_text = response_data["choices"][0]["message"]["content"]
                    logger.debug(f"OpenRouter response: {response_text[:200]}...")
                    return response_text
                else:
                    raise RuntimeError("OpenRouter returned empty response")

        except httpx.TimeoutException as e:
            logger.error(f"OpenRouter API timeout after {self.timeout}s: {str(e)}")
            raise TimeoutError(f"OpenRouter analysis timed out: {str(e)}")

        except httpx.RequestError as e:
            logger.error(f"OpenRouter request error: {str(e)}")
            raise RuntimeError(f"OpenRouter request error: {str(e)}")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse OpenRouter response: {str(e)}")
            raise RuntimeError(f"Invalid JSON response from OpenRouter: {str(e)}")

        except Exception as e:
            logger.error(f"Unexpected error calling OpenRouter: {str(e)}")
            raise RuntimeError(f"Unexpected error: {str(e)}")

    @staticmethod
    def get_available_models() -> Dict[str, str]:
        """
        Get available model shortcuts and their full identifiers.

        Returns:
            Dictionary mapping model names to full OpenRouter model IDs
        """
        return OpenRouterProvider.MODELS.copy()
