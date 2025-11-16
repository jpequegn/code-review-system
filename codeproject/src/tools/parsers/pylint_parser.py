"""
Pylint output parser.

Converts pylint JSON output to standardized finding format.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class PylintParser:
    """Parses pylint JSON output into standardized findings."""

    # Map pylint message types to finding categories
    MESSAGE_TYPE_MAP = {
        "convention": "quality",
        "refactor": "quality",
        "warning": "quality",
        "error": "quality",
        "fatal": "quality",
    }

    # Map pylint message type to severity
    SEVERITY_MAP = {
        "convention": "low",
        "refactor": "low",
        "warning": "medium",
        "error": "high",
        "fatal": "critical",
    }

    def parse(
        self,
        pylint_output: List[Dict[str, Any]],
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse pylint JSON output to standardized format.

        Args:
            pylint_output: List of pylint message dictionaries
            file_path: File being analyzed

        Returns:
            List of findings in standardized format
        """
        findings = []

        for message in pylint_output:
            try:
                finding = self._parse_message(message, file_path)
                if finding:
                    findings.append(finding)
            except Exception as e:
                logger.warning(f"Failed to parse pylint message: {e}")
                continue

        return findings

    def _parse_message(
        self,
        message: Dict[str, Any],
        file_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single pylint message.

        Args:
            message: Single pylint message dictionary
            file_path: File being analyzed

        Returns:
            Standardized finding dict or None if invalid
        """
        # Extract required fields
        msg_type = message.get("type", "").lower()
        symbol = message.get("symbol", "unknown")
        line = message.get("line")
        column = message.get("column")
        message_text = message.get("message", "")

        if not msg_type or not message_text:
            return None

        # Map to standard finding format
        return {
            "tool": "pylint",
            "category": self.MESSAGE_TYPE_MAP.get(msg_type, "quality"),
            "severity": self.SEVERITY_MAP.get(msg_type, "medium"),
            "title": f"{symbol}: {message_text[:60]}",
            "description": message_text,
            "file_path": file_path,
            "line_number": line,
            "column_number": column,
            "message_id": symbol,
            "confidence": 0.85,  # Pylint is reliable
            "suggested_fix": None,  # Pylint doesn't provide fixes
        }
