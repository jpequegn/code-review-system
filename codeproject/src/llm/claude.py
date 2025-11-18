"""
Claude LLM Provider Implementation

Uses Anthropic SDK to interact with Claude models for code analysis.
Supports streaming for long-running analyses.
"""

import json
import logging
from typing import Any, Dict

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

        # Build enhanced prompt with context
        enhanced = enhance_security_prompt(code_diff, context_prompt=context_prompt)
        return self._call_claude(enhanced)

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

        # Build enhanced prompt with context
        enhanced = enhance_performance_prompt(code_diff, context_prompt=context_prompt)
        return self._call_claude(enhanced)

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

If you cannot generate a safe fix, set auto_fix to null and explain in confidence_reasoning."""

        try:
            response = self._call_claude(prompt)
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
            response = self._call_claude(prompt)
            # Clean up response (remove any markdown or extra formatting)
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
            response = self._call_claude(prompt)
            # Clean up response
            return response.strip()
        except Exception as e:
            logger.error(f"Error generating improvement suggestions: {e}")
            raise

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
