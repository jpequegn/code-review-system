"""
Code Analysis Orchestration

Coordinates security and performance analysis of code changes.
Integrates LLM analysis with multi-tool static analysis (pylint, bandit, mypy, coverage).
Parses responses, deduplicates findings, and assigns confidence scores.
"""

import logging
import json
from typing import List, Dict, Any, Optional, Tuple

from src.llm.provider import LLMProvider, get_llm_provider
from src.analysis.diff_parser import FileDiff
from src.database import FindingCategory, FindingSeverity
from src.tools.runner import ToolRunner, ToolExecutionError, ToolParsingError
from src.tools.unifier import FindingsUnifier, UnifiedFinding

logger = logging.getLogger(__name__)


# ============================================================================
# Finding Data Model
# ============================================================================


class AnalyzedFinding:
    """
    Represents a finding parsed from LLM response.

    Intermediate model between LLM response and database Finding object.
    Includes confidence score and deduplication metadata.
    """

    def __init__(
        self,
        category: FindingCategory,
        severity: FindingSeverity,
        title: str,
        description: str,
        file_path: str,
        line_number: Optional[int] = None,
        suggested_fix: Optional[str] = None,
        confidence: float = 0.95,
    ):
        self.category = category
        self.severity = severity
        self.title = title
        self.description = description
        self.file_path = file_path
        self.line_number = line_number
        self.suggested_fix = suggested_fix
        self.confidence = confidence  # 0.0-1.0

    def __repr__(self) -> str:
        return (
            f"AnalyzedFinding({self.file_path}:{self.line_number} "
            f"[{self.severity}] {self.title[:40]}...)"
        )

    def dedup_key(self) -> Tuple:
        """
        Generate deduplication key for finding comparison.

        Two findings are considered duplicates if they have the same:
        - Category (security vs performance)
        - File path
        - Line number (if specified)
        - Title

        Returns:
            Tuple of (category, file_path, line_number, title)
        """
        return (
            (
                self.category.value
                if isinstance(self.category, FindingCategory)
                else self.category
            ),
            self.file_path,
            self.line_number,
            self.title.lower().strip(),
        )


# ============================================================================
# Code Analyzer
# ============================================================================


class CodeAnalyzer:
    """
    Orchestrates security and performance analysis of code changes.

    Responsibilities:
    - Run static analysis tools (pylint, bandit, mypy, coverage)
    - Route code to LLM analyzers for semantic analysis
    - Deduplicate overlapping findings across tools
    - Unify findings with confidence scores
    - Assign severity based on multi-signal analysis
    """

    def __init__(
        self,
        llm_provider: Optional[LLMProvider] = None,
        tool_runner: Optional[ToolRunner] = None,
        use_tools: bool = True,
    ):
        """
        Initialize CodeAnalyzer.

        Args:
            llm_provider: LLM provider instance (uses default if not provided)
            tool_runner: Tool runner instance (creates new if not provided)
            use_tools: Whether to run static analysis tools (default: True)
        """
        self.llm_provider = llm_provider or get_llm_provider()
        self.tool_runner = tool_runner if tool_runner is not None else ToolRunner()
        self.unifier = FindingsUnifier()
        self.use_tools = use_tools

    def analyze_code_changes(
        self,
        file_diffs: List[FileDiff],
    ) -> List[AnalyzedFinding]:
        """
        Analyze code changes using tools and LLM.

        Pipeline:
        1. Run static analysis tools (pylint, bandit, mypy)
        2. Run LLM analysis (security, performance)
        3. Unify findings and deduplicate
        4. Sort by severity and confidence

        Args:
            file_diffs: List of FileDiff objects from DiffParser

        Returns:
            List of deduplicated AnalyzedFinding objects sorted by severity
        """
        if not file_diffs:
            return []

        # Convert file diffs to code snippet for analysis
        code_snippet = self._diffs_to_code_snippet(file_diffs)

        # Step 1: Run static analysis tools
        tool_findings: Dict[str, List[Dict[str, Any]]] = {}
        if self.use_tools:
            tool_findings = self._run_static_analysis_tools(code_snippet, file_diffs)

        # Step 2: Run LLM analysis
        security_findings = self._analyze_security(code_snippet)
        performance_findings = self._analyze_performance(code_snippet)
        llm_findings = security_findings + performance_findings

        # Step 2b: Deduplicate LLM findings (same issue from both analyzers)
        llm_findings = self._deduplicate_findings(llm_findings)

        # Step 3: Convert AnalyzedFinding objects to dicts for unifier
        llm_findings_dicts = [self._analyzed_to_dict(f) for f in llm_findings]

        # Step 4: Unify tool and LLM findings
        unified_findings = self.unifier.unify_findings(tool_findings, llm_findings_dicts)
        unified_findings = self.unifier.sort_by_severity_and_confidence(unified_findings)

        # Step 4: Convert to AnalyzedFinding format for consistency
        analyzed = [self._unified_to_analyzed(f) for f in unified_findings]

        return analyzed

    def _run_static_analysis_tools(
        self,
        code_snippet: str,
        file_diffs: List[FileDiff],
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Run static analysis tools on code snippet.

        Args:
            code_snippet: Code to analyze
            file_diffs: File diffs for context

        Returns:
            Dictionary mapping tool name to findings
        """
        try:
            # Get primary file path from diffs
            file_path = file_diffs[0].file_path if file_diffs else "unknown"

            # Run tools
            return self.tool_runner.run_all_tools(code_snippet, file_path)

        except (ToolExecutionError, ToolParsingError) as e:
            logger.warning(f"Tool execution failed: {e}")
            return {}
        except Exception as e:
            logger.error(f"Unexpected error running tools: {e}")
            return {}

    def _analyzed_to_dict(self, finding: AnalyzedFinding) -> Dict[str, Any]:
        """
        Convert AnalyzedFinding to dict for unifier.

        Args:
            finding: AnalyzedFinding instance

        Returns:
            Dictionary representation
        """
        return {
            "category": finding.category.value,
            "severity": finding.severity.value,
            "title": finding.title,
            "description": finding.description,
            "file_path": finding.file_path,
            "line_number": finding.line_number,
            "suggested_fix": finding.suggested_fix,
            "confidence": finding.confidence,
        }

    def _unified_to_analyzed(self, unified: UnifiedFinding) -> AnalyzedFinding:
        """
        Convert UnifiedFinding to AnalyzedFinding for backward compatibility.

        Args:
            unified: UnifiedFinding instance

        Returns:
            AnalyzedFinding instance
        """
        # Map category string to FindingCategory enum
        category_map = {
            "security": FindingCategory.SECURITY,
            "performance": FindingCategory.PERFORMANCE,
            "quality": FindingCategory.BEST_PRACTICE,
            "testing": FindingCategory.BEST_PRACTICE,
        }
        category = category_map.get(unified.category, FindingCategory.BEST_PRACTICE)

        # Map severity string to FindingSeverity enum
        severity_map = {
            "critical": FindingSeverity.CRITICAL,
            "high": FindingSeverity.HIGH,
            "medium": FindingSeverity.MEDIUM,
            "low": FindingSeverity.LOW,
        }
        severity = severity_map.get(unified.severity, FindingSeverity.MEDIUM)

        return AnalyzedFinding(
            category=category,
            severity=severity,
            title=unified.title,
            description=unified.description,
            file_path=unified.file_path,
            line_number=unified.line_number,
            suggested_fix=unified.suggested_fix,
            confidence=unified.combined_confidence,
        )

    def _diffs_to_code_snippet(self, file_diffs: List[FileDiff]) -> str:
        """
        Convert FileDiff objects to formatted code snippet for LLM analysis.

        Args:
            file_diffs: List of FileDiff objects

        Returns:
            Formatted code snippet string
        """
        lines = []

        for file_diff in file_diffs:
            # File header
            lines.append(f"\n# File: {file_diff.file_path}")
            lines.append(f"# Changes: +{file_diff.additions}/-{file_diff.deletions}")
            lines.append("")

            # Code changes with context
            for change in file_diff.changes:
                # Context before
                for ctx_line in change.context_before:
                    lines.append(f"  {ctx_line}")

                # The actual change
                if change.change_type == "add":
                    lines.append(f"+ {change.content}")
                elif change.change_type == "remove":
                    lines.append(f"- {change.content}")
                else:  # context
                    lines.append(f"  {change.content}")

                # Context after
                for ctx_line in change.context_after:
                    lines.append(f"  {ctx_line}")

                lines.append("")

        return "\n".join(lines)

    def _analyze_security(self, code_snippet: str) -> List[AnalyzedFinding]:
        """
        Analyze code for security vulnerabilities.

        Args:
            code_snippet: Formatted code snippet

        Returns:
            List of security findings
        """
        try:
            response = self.llm_provider.analyze_security(code_snippet)
            return self._parse_findings_response(response, FindingCategory.SECURITY)
        except Exception as e:
            logger.error(f"Security analysis failed: {str(e)}")
            return []

    def _analyze_performance(self, code_snippet: str) -> List[AnalyzedFinding]:
        """
        Analyze code for performance issues.

        Args:
            code_snippet: Formatted code snippet

        Returns:
            List of performance findings
        """
        try:
            response = self.llm_provider.analyze_performance(code_snippet)
            return self._parse_findings_response(response, FindingCategory.PERFORMANCE)
        except Exception as e:
            logger.error(f"Performance analysis failed: {str(e)}")
            return []

    def _parse_findings_response(
        self,
        response: str,
        category: FindingCategory,
    ) -> List[AnalyzedFinding]:
        """
        Parse LLM response JSON into AnalyzedFinding objects.

        Args:
            response: JSON string from LLM provider
            category: Finding category (security or performance)

        Returns:
            List of parsed AnalyzedFinding objects

        Note:
            Handles malformed JSON gracefully, logging errors and returning
            empty list rather than raising exceptions.
        """
        findings = []

        try:
            # Parse JSON response
            data = json.loads(response)

            # Extract findings array
            if not isinstance(data, dict) or "findings" not in data:
                logger.warning(
                    f"Malformed LLM response: missing 'findings' key. "
                    f"Response: {response[:200]}"
                )
                return []

            findings_list = data.get("findings", [])
            if not isinstance(findings_list, list):
                logger.warning(
                    f"Malformed LLM response: 'findings' is not a list. "
                    f"Type: {type(findings_list)}"
                )
                return []

            # Parse each finding
            for idx, finding_data in enumerate(findings_list):
                try:
                    analyzed_finding = self._parse_single_finding(
                        finding_data, category
                    )
                    if analyzed_finding:
                        findings.append(analyzed_finding)
                except Exception as e:
                    logger.warning(
                        f"Failed to parse finding {idx}: {str(e)}. "
                        f"Data: {finding_data}"
                    )
                    continue

            return findings

        except json.JSONDecodeError as e:
            logger.error(
                f"Failed to parse LLM response as JSON: {str(e)}. "
                f"Response: {response[:200]}"
            )
            return []
        except Exception as e:
            logger.error(f"Unexpected error parsing findings: {str(e)}")
            return []

    def _parse_single_finding(
        self,
        finding_data: Dict[str, Any],
        category: FindingCategory,
    ) -> Optional[AnalyzedFinding]:
        """
        Parse a single finding from LLM response.

        Args:
            finding_data: Dictionary containing finding fields
            category: Finding category

        Returns:
            AnalyzedFinding object or None if required fields are missing

        Raises:
            ValueError: If required fields are invalid
        """
        if not isinstance(finding_data, dict):
            raise ValueError(f"Finding data must be dict, got {type(finding_data)}")

        # Extract and validate required fields
        severity_str = finding_data.get("severity", "").lower().strip()
        if not severity_str:
            raise ValueError("Missing required field: severity")

        # Convert severity string to enum
        try:
            severity = FindingSeverity[severity_str.upper()]
        except KeyError:
            raise ValueError(
                f"Invalid severity: {severity_str}. "
                f"Must be one of: {[s.value for s in FindingSeverity]}"
            )

        # Extract other required fields
        title = finding_data.get("title", "").strip()
        if not title:
            raise ValueError("Missing required field: title")

        description = finding_data.get("description", "").strip()
        if not description:
            raise ValueError("Missing required field: description")

        file_path = finding_data.get("file_path", "").strip()
        if not file_path:
            raise ValueError("Missing required field: file_path")

        # Extract optional fields
        line_number = finding_data.get("line_number")
        if line_number is not None and not isinstance(line_number, int):
            try:
                line_number = int(line_number)
            except (ValueError, TypeError):
                logger.warning(f"Invalid line_number: {line_number}, setting to None")
                line_number = None

        suggested_fix = finding_data.get("suggested_fix", "").strip() or None

        # Extract confidence (optional, default 0.95)
        confidence = finding_data.get("confidence", 0.95)
        if not isinstance(confidence, (int, float)):
            try:
                confidence = float(confidence)
            except (ValueError, TypeError):
                logger.warning(f"Invalid confidence: {confidence}, using default 0.95")
                confidence = 0.95

        # Clamp confidence to [0.0, 1.0]
        confidence = max(0.0, min(1.0, confidence))

        return AnalyzedFinding(
            category=category,
            severity=severity,
            title=title,
            description=description,
            file_path=file_path,
            line_number=line_number,
            suggested_fix=suggested_fix,
            confidence=confidence,
        )

    def _deduplicate_findings(
        self,
        findings: List[AnalyzedFinding],
    ) -> List[AnalyzedFinding]:
        """
        Deduplicate identical findings.

        When duplicate findings are detected (same category, file, line, title),
        keeps the one with higher confidence score.

        Args:
            findings: List of AnalyzedFinding objects

        Returns:
            Deduplicated list of findings
        """
        if not findings:
            return []

        # Group by dedup key
        finding_groups: Dict[Tuple, List[AnalyzedFinding]] = {}
        for finding in findings:
            key = finding.dedup_key()
            if key not in finding_groups:
                finding_groups[key] = []
            finding_groups[key].append(finding)

        # Keep finding with highest confidence from each group
        deduplicated = []
        for group in finding_groups.values():
            # Sort by confidence descending, take first
            best_finding = max(group, key=lambda f: f.confidence)
            deduplicated.append(best_finding)

        return deduplicated

    def _sort_by_severity(
        self,
        findings: List[AnalyzedFinding],
    ) -> List[AnalyzedFinding]:
        """
        Sort findings by severity (critical to low).

        Args:
            findings: List of AnalyzedFinding objects

        Returns:
            Sorted list of findings
        """
        severity_order = {
            FindingSeverity.CRITICAL: 0,
            FindingSeverity.HIGH: 1,
            FindingSeverity.MEDIUM: 2,
            FindingSeverity.LOW: 3,
        }

        return sorted(findings, key=lambda f: severity_order.get(f.severity, 99))
