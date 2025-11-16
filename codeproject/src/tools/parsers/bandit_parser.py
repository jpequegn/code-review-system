"""
Bandit output parser.

Converts bandit JSON output to standardized finding format.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class BanditParser:
    """Parses bandit JSON output into standardized findings."""

    # Map bandit severity to standard severity
    SEVERITY_MAP = {
        "LOW": "low",
        "MEDIUM": "medium",
        "HIGH": "high",
    }

    def parse(
        self,
        bandit_output: Dict[str, Any],
        file_path: str,
    ) -> List[Dict[str, Any]]:
        """
        Parse bandit JSON output to standardized format.

        Args:
            bandit_output: Bandit JSON output dictionary
            file_path: File being analyzed

        Returns:
            List of findings in standardized format
        """
        findings = []

        # Extract results array from bandit output
        results = bandit_output.get("results", [])

        for result in results:
            try:
                finding = self._parse_result(result, file_path)
                if finding:
                    findings.append(finding)
            except Exception as e:
                logger.warning(f"Failed to parse bandit result: {e}")
                continue

        return findings

    def _parse_result(
        self,
        result: Dict[str, Any],
        file_path: str,
    ) -> Optional[Dict[str, Any]]:
        """
        Parse a single bandit result.

        Args:
            result: Single bandit result dictionary
            file_path: File being analyzed

        Returns:
            Standardized finding dict or None if invalid
        """
        # Extract required fields
        test_id = result.get("test_id", "B000")
        test_name = result.get("test_name", "unknown")
        severity = result.get("severity", "MEDIUM").upper()
        confidence = result.get("confidence", "MEDIUM").upper()
        issue_text = result.get("issue_text", "")
        line_number = result.get("line_number")
        line_range = result.get("line_range", [])

        if not issue_text:
            return None

        # Map severity
        severity_mapped = self.SEVERITY_MAP.get(severity, "medium")

        # Calculate confidence score (0.0-1.0)
        confidence_score = 0.9 if confidence == "HIGH" else 0.75

        return {
            "tool": "bandit",
            "category": "security",
            "severity": severity_mapped,
            "title": f"{test_id} {test_name}: {issue_text[:60]}",
            "description": issue_text,
            "file_path": file_path,
            "line_number": line_number,
            "line_range": line_range,
            "test_id": test_id,
            "confidence": confidence_score,
            "suggested_fix": None,  # Bandit doesn't provide fixes
        }
