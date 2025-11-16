"""
Finding Unification Module

Combines findings from multiple tools and deduplicates them.
Merges tool signals with LLM findings for enriched analysis.
"""

import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


# ============================================================================
# Unified Finding Model
# ============================================================================


@dataclass
class UnifiedFinding:
    """
    Unified finding combining signals from multiple analysis sources.

    Attributes:
        file_path: File path where finding was detected
        line_number: Line number (if applicable)
        category: Finding category (security, performance, quality, testing)
        severity: Severity level (critical, high, medium, low)
        title: Short title of the finding
        description: Detailed description
        tools: Dictionary mapping tool name to tool-specific finding
        llm_signal: LLM analysis findings for this issue
        combined_confidence: Confidence score combining all signals (0.0-1.0)
        suggested_fix: Suggested remediation
    """

    file_path: str
    category: str
    severity: str
    title: str
    description: str
    line_number: Optional[int] = None
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    llm_signal: Optional[Dict[str, Any]] = None
    combined_confidence: float = 0.5
    suggested_fix: Optional[str] = None

    def dedup_key(self) -> Tuple:
        """Generate deduplication key for finding comparison."""
        return (
            self.category,
            self.file_path,
            self.line_number,
            self.title.lower().strip(),
        )

    def __repr__(self) -> str:
        tool_count = len(self.tools) if self.tools else 0
        return (
            f"UnifiedFinding({self.file_path}:{self.line_number} "
            f"[{self.severity}] {self.title[:40]}... "
            f"({tool_count} tools))"
        )


# ============================================================================
# Findings Unifier
# ============================================================================


class FindingsUnifier:
    """
    Unifies findings from multiple sources (tools + LLM).

    Deduplicates findings that represent the same issue from different sources
    and combines confidence scores.
    """

    def __init__(self):
        """Initialize unifier."""
        self.logger = logger

    def unify_findings(
        self,
        tool_findings: Dict[str, List[Dict[str, Any]]],
        llm_findings: Optional[List[Dict[str, Any]]] = None,
    ) -> List[UnifiedFinding]:
        """
        Unify findings from all sources.

        Args:
            tool_findings: Dictionary mapping tool name to list of findings
            llm_findings: Optional list of LLM-generated findings

        Returns:
            List of unified findings
        """
        # Step 1: Deduplicate tool findings
        deduplicated_tools = self._deduplicate_tool_findings(tool_findings)

        # Step 2: Convert to unified findings
        unified = []
        for finding_key, finding_info in deduplicated_tools.items():
            unified_finding = self._create_unified_finding(finding_info)
            unified.append(unified_finding)

        # Step 3: Merge with LLM findings (enrich existing ones)
        if llm_findings:
            unified = self._merge_llm_findings(unified, llm_findings)

        return unified

    def _deduplicate_tool_findings(
        self,
        tool_findings: Dict[str, List[Dict[str, Any]]],
    ) -> Dict[Tuple, Dict[str, Any]]:
        """
        Deduplicate findings across all tools.

        Finding from different tools are merged if they refer to same issue.

        Args:
            tool_findings: Dictionary mapping tool name to findings

        Returns:
            Dictionary mapping finding key to deduplicated finding info
        """
        deduplicated: Dict[Tuple, Dict[str, Any]] = {}

        for tool_name, findings in tool_findings.items():
            for finding in findings:
                # Generate dedup key
                key = self._get_finding_key(finding)

                if key not in deduplicated:
                    # First occurrence: create entry
                    deduplicated[key] = {
                        "finding": finding,
                        "tools": {tool_name: finding},
                        "tool_count": 1,
                        "confidence_scores": [finding.get("confidence", 0.5)],
                    }
                else:
                    # Duplicate detected: merge
                    deduplicated[key]["tools"][tool_name] = finding
                    deduplicated[key]["tool_count"] += 1
                    deduplicated[key]["confidence_scores"].append(
                        finding.get("confidence", 0.5)
                    )

        return deduplicated

    def _get_finding_key(self, finding: Dict[str, Any]) -> Tuple:
        """Generate deduplication key for a finding."""
        return (
            finding.get("category", "unknown"),
            finding.get("file_path", "unknown"),
            finding.get("line_number"),
            finding.get("title", "").lower().strip(),
        )

    def _create_unified_finding(
        self,
        finding_info: Dict[str, Any],
    ) -> UnifiedFinding:
        """
        Create a unified finding from deduplicated tool findings.

        Args:
            finding_info: Dictionary with dedup info

        Returns:
            UnifiedFinding instance
        """
        primary_finding = finding_info["finding"]
        tool_count = finding_info["tool_count"]
        confidence_scores = finding_info["confidence_scores"]

        # Calculate combined confidence
        # Higher when multiple tools agree
        base_confidence = sum(confidence_scores) / len(confidence_scores)
        tool_agreement_bonus = min(0.15, tool_count * 0.05)  # +5% per tool, max 15%
        combined_confidence = min(1.0, base_confidence + tool_agreement_bonus)

        # Upgrade severity if multiple tools agree on high/critical
        severity = primary_finding.get("severity", "medium")
        if tool_count >= 2:
            severity = self._elevate_severity_on_agreement(severity, primary_finding)

        return UnifiedFinding(
            file_path=primary_finding.get("file_path", "unknown"),
            line_number=primary_finding.get("line_number"),
            category=primary_finding.get("category", "unknown"),
            severity=severity,
            title=primary_finding.get("title", "Unknown finding"),
            description=primary_finding.get("description", ""),
            tools=finding_info["tools"],
            combined_confidence=combined_confidence,
            suggested_fix=primary_finding.get("suggested_fix"),
        )

    def _elevate_severity_on_agreement(
        self,
        severity: str,
        finding: Dict[str, Any],
    ) -> str:
        """
        Elevate severity if multiple tools agree.

        Args:
            severity: Current severity level
            finding: Finding dictionary

        Returns:
            Potentially elevated severity
        """
        # If finding is already high/critical, keep it
        if severity in ["critical", "high"]:
            return severity

        # If tool is security-related, elevate medium to high
        if finding.get("category") == "security" and severity == "medium":
            return "high"

        return severity

    def _merge_llm_findings(
        self,
        tool_findings: List[UnifiedFinding],
        llm_findings: List[Dict[str, Any]],
    ) -> List[UnifiedFinding]:
        """
        Merge LLM findings into tool findings.

        Args:
            tool_findings: List of tool-based unified findings
            llm_findings: List of LLM findings

        Returns:
            Merged list of findings
        """
        result = []

        # Create lookup for tool findings
        tool_finding_map = {f.dedup_key(): f for f in tool_findings}

        # Merge LLM findings
        for llm_finding in llm_findings:
            llm_key = self._get_finding_key(llm_finding)

            if llm_key in tool_finding_map:
                # Tool finding exists: enrich it with LLM data
                unified = tool_finding_map[llm_key]
                unified.llm_signal = llm_finding
                # Boost confidence when LLM agrees with tools
                unified.combined_confidence = min(
                    1.0,
                    unified.combined_confidence
                    + (llm_finding.get("confidence", 0.8) * 0.1),
                )
                result.append(unified)
            else:
                # New finding from LLM: create unified finding
                unified = UnifiedFinding(
                    file_path=llm_finding.get("file_path", "unknown"),
                    line_number=llm_finding.get("line_number"),
                    category=llm_finding.get("category", "unknown"),
                    severity=llm_finding.get("severity", "medium"),
                    title=llm_finding.get("title", "Unknown finding"),
                    description=llm_finding.get("description", ""),
                    llm_signal=llm_finding,
                    combined_confidence=llm_finding.get("confidence", 0.8),
                    suggested_fix=llm_finding.get("suggested_fix"),
                )
                result.append(unified)

        # Add tool findings that weren't merged with LLM findings
        for finding in tool_findings:
            if finding.dedup_key() not in {f.dedup_key() for f in result}:
                result.append(finding)

        return result

    def sort_by_severity_and_confidence(
        self,
        findings: List[UnifiedFinding],
    ) -> List[UnifiedFinding]:
        """
        Sort findings by severity and confidence.

        Args:
            findings: List of findings to sort

        Returns:
            Sorted list (critical/high first, higher confidence first)
        """
        severity_order = {
            "critical": 0,
            "high": 1,
            "medium": 2,
            "low": 3,
        }

        return sorted(
            findings,
            key=lambda f: (
                severity_order.get(f.severity, 99),
                -f.combined_confidence,  # Higher confidence first (negative for desc)
            ),
        )
