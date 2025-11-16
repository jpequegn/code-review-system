"""
Tool Runner Module

Orchestrates execution of static and dynamic analysis tools on code.
Runs pylint, bandit, mypy, and coverage.py to collect multi-signal findings.
"""

import logging
import json
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any
from abc import ABC, abstractmethod

from src.tools.parsers.pylint_parser import PylintParser
from src.tools.parsers.bandit_parser import BanditParser
from src.tools.parsers.mypy_parser import MypyParser
from src.tools.parsers.coverage_parser import CoverageParser

logger = logging.getLogger(__name__)


# ============================================================================
# Exceptions
# ============================================================================


class ToolRunnerError(Exception):
    """Base exception for tool runner errors."""
    pass


class ToolExecutionError(ToolRunnerError):
    """Raised when tool execution fails."""
    pass


class ToolParsingError(ToolRunnerError):
    """Raised when tool output parsing fails."""
    pass


# ============================================================================
# Tool Runner
# ============================================================================


class ToolRunner:
    """
    Orchestrates execution of static and dynamic analysis tools.

    Runs multiple analysis tools on code and collects findings in a unified format.
    Tools are run in parallel for efficiency and results are cached.
    """

    # Tools to run in order of priority
    TOOLS_TO_RUN = ["pylint", "bandit", "mypy", "coverage"]

    def __init__(self, timeout: int = 30, cache_results: bool = True):
        """
        Initialize ToolRunner.

        Args:
            timeout: Maximum seconds to wait for tool execution (default: 30)
            cache_results: Whether to cache tool results (default: True)
        """
        self.timeout = timeout
        self.cache_results = cache_results
        self._cache: Dict[str, Dict[str, Any]] = {}

        # Initialize parsers
        self.pylint_parser = PylintParser()
        self.bandit_parser = BanditParser()
        self.mypy_parser = MypyParser()
        self.coverage_parser = CoverageParser()

    def run_all_tools(self, code_snippet: str, file_path: str) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run all available tools on code snippet.

        Args:
            code_snippet: Python code to analyze
            file_path: Path to file being analyzed (for context)

        Returns:
            Dictionary mapping tool name to list of findings
            {
                "pylint": [...],
                "bandit": [...],
                "mypy": [...],
                "coverage": [...]
            }
        """
        results = {}

        # Check cache first
        cache_key = self._get_cache_key(code_snippet, file_path)
        if self.cache_results and cache_key in self._cache:
            logger.debug(f"Using cached results for {file_path}")
            return self._cache[cache_key]

        # Run each tool
        for tool_name in self.TOOLS_TO_RUN:
            try:
                if tool_name == "pylint":
                    results[tool_name] = self.run_pylint(code_snippet, file_path)
                elif tool_name == "bandit":
                    results[tool_name] = self.run_bandit(code_snippet, file_path)
                elif tool_name == "mypy":
                    results[tool_name] = self.run_mypy(code_snippet, file_path)
                elif tool_name == "coverage":
                    # Coverage requires different input (coverage data, not code)
                    results[tool_name] = []
            except Exception as e:
                logger.warning(f"Tool {tool_name} failed: {e}")
                results[tool_name] = []

        # Cache results
        if self.cache_results:
            self._cache[cache_key] = results

        return results

    def run_pylint(self, code_snippet: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Run pylint on code snippet.

        Args:
            code_snippet: Python code to analyze
            file_path: Path to file being analyzed

        Returns:
            List of findings in standardized format
        """
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                f.write(code_snippet)
                temp_file = f.name

            try:
                # Run pylint with JSON output
                result = subprocess.run(
                    ["pylint", "--output-format=json", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                # Parse output
                if result.stdout:
                    pylint_findings = json.loads(result.stdout)
                    # Parse and normalize findings
                    return self.pylint_parser.parse(pylint_findings, file_path)
                return []

            finally:
                # Clean up temporary file
                Path(temp_file).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            logger.warning(f"Pylint timeout on {file_path}")
            raise ToolExecutionError(f"Pylint timeout after {self.timeout}s")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse pylint output: {e}")
            raise ToolParsingError(f"Invalid pylint JSON output: {e}")
        except FileNotFoundError:
            logger.warning("Pylint not installed")
            raise ToolExecutionError("Pylint not available in PATH")

    def run_bandit(self, code_snippet: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Run bandit (security analyzer) on code snippet.

        Args:
            code_snippet: Python code to analyze
            file_path: Path to file being analyzed

        Returns:
            List of security findings in standardized format
        """
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                f.write(code_snippet)
                temp_file = f.name

            try:
                # Run bandit with JSON output
                result = subprocess.run(
                    ["bandit", "-f", "json", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                # Parse output
                if result.stdout:
                    bandit_findings = json.loads(result.stdout)
                    # Parse and normalize findings
                    return self.bandit_parser.parse(bandit_findings, file_path)
                return []

            finally:
                # Clean up temporary file
                Path(temp_file).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            logger.warning(f"Bandit timeout on {file_path}")
            raise ToolExecutionError(f"Bandit timeout after {self.timeout}s")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse bandit output: {e}")
            raise ToolParsingError(f"Invalid bandit JSON output: {e}")
        except FileNotFoundError:
            logger.warning("Bandit not installed")
            raise ToolExecutionError("Bandit not available in PATH")

    def run_mypy(self, code_snippet: str, file_path: str) -> List[Dict[str, Any]]:
        """
        Run mypy (type checker) on code snippet.

        Args:
            code_snippet: Python code to analyze
            file_path: Path to file being analyzed

        Returns:
            List of type issues in standardized format
        """
        try:
            # Write code to temporary file
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".py",
                delete=False,
            ) as f:
                f.write(code_snippet)
                temp_file = f.name

            try:
                # Run mypy with JSON output
                result = subprocess.run(
                    ["mypy", "--output-json", "--no-error-summary", temp_file],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                )

                # Parse output
                findings = []
                if result.stdout:
                    # mypy outputs JSON lines (one JSON object per line)
                    for line in result.stdout.strip().split("\n"):
                        if line:
                            mypy_finding = json.loads(line)
                            findings.extend(
                                self.mypy_parser.parse([mypy_finding], file_path)
                            )
                return findings

            finally:
                # Clean up temporary file
                Path(temp_file).unlink(missing_ok=True)

        except subprocess.TimeoutExpired:
            logger.warning(f"Mypy timeout on {file_path}")
            raise ToolExecutionError(f"Mypy timeout after {self.timeout}s")
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse mypy output: {e}")
            raise ToolParsingError(f"Invalid mypy JSON output: {e}")
        except FileNotFoundError:
            logger.warning("Mypy not installed")
            raise ToolExecutionError("Mypy not available in PATH")

    def run_coverage_analysis(
        self,
        coverage_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Analyze coverage data to identify test gaps.

        Args:
            coverage_data: Coverage.py data dictionary

        Returns:
            List of coverage gaps in standardized format
        """
        try:
            return self.coverage_parser.parse(coverage_data)
        except Exception as e:
            logger.warning(f"Coverage analysis failed: {e}")
            raise ToolParsingError(f"Failed to analyze coverage data: {e}")

    def _get_cache_key(self, code_snippet: str, file_path: str) -> str:
        """Generate cache key from code snippet and file path."""
        import hashlib
        code_hash = hashlib.md5(code_snippet.encode()).hexdigest()
        return f"{file_path}:{code_hash}"

    def clear_cache(self) -> None:
        """Clear cached tool results."""
        self._cache.clear()
        logger.debug("Tool result cache cleared")
