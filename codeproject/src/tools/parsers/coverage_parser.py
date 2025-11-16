"""
Coverage.py output parser.

Converts coverage data to standardized finding format for uncovered lines.
"""

import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class CoverageParser:
    """Parses coverage.py data into standardized findings."""

    def parse(
        self,
        coverage_data: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Parse coverage data to identify test gaps.

        Args:
            coverage_data: Coverage data dictionary from coverage.py

        Returns:
            List of coverage gap findings in standardized format
        """
        findings = []

        # Coverage data format from coverage.py:
        # {
        #   "meta": {...},
        #   "files": {
        #     "path/to/file.py": {
        #       "summary": {"covered_lines": N, "num_statements": N, ...},
        #       "excluded_lines": [...],
        #       "missing_lines": [...]
        #     }
        #   }
        # }

        files = coverage_data.get("files", {})

        for file_path, file_coverage in files.items():
            try:
                file_findings = self._parse_file_coverage(file_path, file_coverage)
                findings.extend(file_findings)
            except Exception as e:
                logger.warning(f"Failed to parse coverage for {file_path}: {e}")
                continue

        return findings

    def _parse_file_coverage(
        self,
        file_path: str,
        coverage: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Parse coverage data for a single file.

        Args:
            file_path: Path to file
            coverage: Coverage data for file

        Returns:
            List of findings for uncovered areas
        """
        findings = []

        # Get summary statistics
        summary = coverage.get("summary", {})
        covered_lines = summary.get("covered_lines", 0)
        num_statements = summary.get("num_statements", 0)

        # Calculate coverage percentage
        if num_statements > 0:
            coverage_percent = (covered_lines / num_statements) * 100
        else:
            coverage_percent = 100.0

        # Get missing lines
        missing_lines = coverage.get("missing_lines", [])

        # Create finding if coverage is low
        if coverage_percent < 80:  # Flag files with <80% coverage
            finding = {
                "tool": "coverage",
                "category": "testing",
                "severity": self._get_severity_from_coverage(coverage_percent),
                "title": f"Low test coverage: {coverage_percent:.1f}%",
                "description": (
                    f"File has {coverage_percent:.1f}% coverage "
                    f"({covered_lines}/{num_statements} lines). "
                    f"Missing {len(missing_lines)} lines."
                ),
                "file_path": file_path,
                "line_number": None,  # Coverage is file-level
                "missing_lines": missing_lines[:20],  # Limit to first 20
                "coverage_percent": coverage_percent,
                "confidence": 1.0,  # Coverage is definitive
                "suggested_fix": f"Add tests to cover the {len(missing_lines)} missing lines",
            }
            findings.append(finding)

        return findings

    def _get_severity_from_coverage(self, coverage_percent: float) -> str:
        """
        Determine severity based on coverage percentage.

        Args:
            coverage_percent: Coverage percentage (0-100)

        Returns:
            Severity level
        """
        if coverage_percent < 50:
            return "critical"
        elif coverage_percent < 70:
            return "high"
        elif coverage_percent < 80:
            return "medium"
        else:
            return "low"
