"""
Mypy output parser.

Converts mypy JSON output to standardized finding format.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class MypyParser:
    """Parses mypy JSON output into standardized findings."""

    def parse(
        self,
        mypy_output: List[Dict[str, Any]],
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse mypy JSON output to standardized format.

        Args:
            mypy_output: List of mypy error dictionaries
            file_path: File being analyzed

        Returns:
            List of findings in standardized format
        """
        findings = []

        for error in mypy_output:
            try:
                finding = self._parse_error(error, file_path)
                if finding:
                    findings.append(finding)
            except Exception as e:
                logger.warning(f"Failed to parse mypy error: {e}")
                continue

        return findings

    def _parse_error(
        self,
        error: Dict[str, Any],
        file_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single mypy error.

        Args:
            error: Single mypy error dictionary
            file_path: File being analyzed

        Returns:
            Standardized finding dict or None if invalid
        """
        # Extract required fields
        line = error.get("line")
        column = error.get("column")
        message = error.get("message", "")
        error_code = error.get("error_code", "misc")

        if not message:
            return None

        # Determine severity based on error type
        severity = self._get_severity_from_message(message, error_code)

        return {
            "tool": "mypy",
            "category": "quality",
            "severity": severity,
            "title": f"Type error: {message[:60]}",
            "description": message,
            "file_path": file_path,
            "line_number": line,
            "column_number": column,
            "error_code": error_code,
            "confidence": 0.95,  # Mypy is very reliable for type checking
            "suggested_fix": None,  # Mypy doesn't provide fixes
        }

    def _get_severity_from_message(self, message: str, error_code: str) -> str:
        """
        Determine severity based on error message and code.

        Args:
            message: Error message
            error_code: Error code from mypy

        Returns:
            Severity level: critical, high, medium, or low
        """
        # Some error codes are more serious
        if error_code in ["assignment", "arg-type", "return-value"]:
            return "high"
        elif error_code in ["name-defined", "import", "unused-ignore"]:
            return "medium"
        else:
            return "medium"
