"""
Adaptive Severity Adjustment

Adjusts finding severity based on historical accuracy and personal preferences.
Learns which types of findings are most accurate for the user's codebase.
"""

from typing import Dict, Tuple

from sqlalchemy.orm import Session

from src.database import FindingCategory, FindingSeverity, LearningMetrics, Finding
from src.learning.learner import LearningEngine


class AdaptiveSeverityAdjuster:
    """
    Adjusts severity of findings based on historical accuracy.

    Uses personal learning metrics to increase or decrease severity
    of findings based on how accurate they've been historically.
    """

    def __init__(self, db: Session, repo_url: str):
        """
        Initialize severity adjuster.

        Args:
            db: SQLAlchemy database session
            repo_url: Repository URL for personal thresholds
        """
        self.db = db
        self.repo_url = repo_url
        self.learning_engine = LearningEngine(db)

    def adjust_severity(
        self,
        category: FindingCategory,
        base_severity: FindingSeverity,
        confidence: float = 1.0,
    ) -> Tuple[FindingSeverity, float]:
        """
        Adjust severity of a finding based on historical accuracy.

        Args:
            category: Finding category
            base_severity: Base severity from tool/LLM
            confidence: System confidence in the finding (0.0-1.0)

        Returns:
            Tuple of (adjusted_severity, confidence_adjustment_factor)
        """
        # Get metrics for this category/severity
        metrics = self.db.query(LearningMetrics).filter(
            LearningMetrics.repo_url == self.repo_url,
            LearningMetrics.category == category,
            LearningMetrics.severity == base_severity,
        ).first()

        if not metrics or metrics.total_findings < 5:
            # Not enough data, return base severity
            return base_severity, 1.0

        # Calculate adjustment based on false positive rate and accuracy
        false_positive_rate = metrics.false_positive_rate / 100.0  # Convert from percentage
        accuracy = metrics.accuracy / 100.0

        # If this type has high false positive rate, demote severity
        if false_positive_rate > 0.4:  # More than 40% false positives
            adjusted_severity = self._demote_severity(base_severity)
            adjustment_factor = 0.8
        elif accuracy > 0.9 and false_positive_rate < 0.1:
            # High accuracy and low false positives, keep or promote
            adjusted_severity = base_severity
            adjustment_factor = 1.2
        else:
            # Moderate accuracy, keep as-is
            adjusted_severity = base_severity
            adjustment_factor = 1.0

        return adjusted_severity, adjustment_factor

    def adjust_confidence(
        self,
        category: FindingCategory,
        base_severity: FindingSeverity,
        confidence: float,
    ) -> float:
        """
        Adjust confidence level of a finding.

        Args:
            category: Finding category
            base_severity: Base severity
            confidence: Base confidence (0.0-1.0)

        Returns:
            Adjusted confidence level
        """
        # Get metrics
        metrics = self.db.query(LearningMetrics).filter(
            LearningMetrics.repo_url == self.repo_url,
            LearningMetrics.category == category,
            LearningMetrics.severity == base_severity,
        ).first()

        if not metrics or metrics.total_findings < 5:
            return confidence

        # Adjust based on accuracy
        accuracy = metrics.accuracy / 100.0
        false_positive_rate = metrics.false_positive_rate / 100.0

        # Formula: base_confidence * accuracy_factor * fp_penalty
        accuracy_factor = 0.5 + (accuracy * 0.5)  # Range 0.5-1.0
        fp_penalty = 1.0 - (false_positive_rate * 0.5)  # Penalize high FP rates

        adjusted = confidence * accuracy_factor * fp_penalty
        return max(0.0, min(1.0, adjusted))  # Clamp to 0-1

    def should_include_finding(
        self,
        category: FindingCategory,
        base_severity: FindingSeverity,
        confidence: float,
    ) -> bool:
        """
        Decide whether to include a finding based on learned thresholds.

        Args:
            category: Finding category
            base_severity: Base severity
            confidence: System confidence

        Returns:
            True if finding should be included
        """
        threshold = self.learning_engine.thresholds.calculate_confidence_threshold(
            category, base_severity
        )
        adjusted_confidence = self.adjust_confidence(category, base_severity, confidence)
        return adjusted_confidence >= threshold

    def _demote_severity(self, severity: FindingSeverity) -> FindingSeverity:
        """
        Demote severity by one level.

        Args:
            severity: Current severity

        Returns:
            Demoted severity
        """
        severity_order = [
            FindingSeverity.CRITICAL,
            FindingSeverity.HIGH,
            FindingSeverity.MEDIUM,
            FindingSeverity.LOW,
        ]

        try:
            idx = severity_order.index(severity)
            if idx < len(severity_order) - 1:
                return severity_order[idx + 1]
            return severity
        except (ValueError, IndexError):
            return severity

    def _promote_severity(self, severity: FindingSeverity) -> FindingSeverity:
        """
        Promote severity by one level.

        Args:
            severity: Current severity

        Returns:
            Promoted severity
        """
        severity_order = [
            FindingSeverity.CRITICAL,
            FindingSeverity.HIGH,
            FindingSeverity.MEDIUM,
            FindingSeverity.LOW,
        ]

        try:
            idx = severity_order.index(severity)
            if idx > 0:
                return severity_order[idx - 1]
            return severity
        except (ValueError, IndexError):
            return severity

    def get_severity_adjustments(self) -> Dict[Tuple[FindingCategory, FindingSeverity], Tuple[FindingSeverity, float]]:
        """
        Get all severity adjustments for this repo.

        Returns:
            Dict mapping (category, severity) to (adjusted_severity, factor)
        """
        adjustments = {}
        for category in FindingCategory:
            for severity in FindingSeverity:
                adjusted, factor = self.adjust_severity(category, severity)
                adjustments[(category, severity)] = (adjusted, factor)
        return adjustments
