"""
Learning Engine

Tracks system accuracy, learns patterns, and identifies personal thresholds.
Enables adaptive severity adjustments based on historical data.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, Optional, Tuple

from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database import (
    FindingFeedback, FeedbackType, IssueValidation,
    ProductionIssue, LearningMetrics, FindingCategory, FindingSeverity,
    FindingFeedback as FB, Finding
)


class HistoricalAccuracy:
    """Tracks accuracy of findings over time."""

    def __init__(self, db: Session):
        """
        Initialize historical accuracy tracker.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def calculate_accuracy(
        self,
        category: Optional[FindingCategory] = None,
        severity: Optional[FindingSeverity] = None,
        days: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Calculate accuracy metrics for findings.

        Args:
            category: Filter by category (optional)
            severity: Filter by severity (optional)
            days: Look back N days (optional)

        Returns:
            Dict with accuracy, precision, recall, false_positive_rate
        """
        query = self.db.query(FB)

        if days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            query = query.filter(FB.created_at >= cutoff)

        # Filter by category if provided (need to join with Finding table)
        if category or severity:
            query = query.join(Finding)
            if category:
                query = query.filter(Finding.category == category)
            if severity:
                query = query.filter(Finding.severity == severity)

        feedback_list = query.all()

        if not feedback_list:
            return {
                "accuracy": 0.0,
                "precision": 0.0,
                "recall": 0.0,
                "false_positive_rate": 0.0,
                "total_findings": 0,
                "confirmed": 0,
                "false_positives": 0,
            }

        # Count outcomes
        total = len(feedback_list)
        confirmed = sum(1 for f in feedback_list if f.validation == IssueValidation.CONFIRMED)
        false_positives = sum(1 for f in feedback_list if f.feedback_type == FeedbackType.FALSE_POSITIVE)
        helpful = sum(1 for f in feedback_list if f.helpful is True)

        # Calculate metrics
        accuracy = (confirmed / total * 100) if total > 0 else 0.0
        precision = (confirmed / (confirmed + false_positives) * 100) if (confirmed + false_positives) > 0 else 0.0
        recall = (confirmed / total * 100) if total > 0 else 0.0
        false_positive_rate = (false_positives / total * 100) if total > 0 else 0.0

        return {
            "accuracy": round(accuracy, 2),
            "precision": round(precision, 2),
            "recall": round(recall, 2),
            "false_positive_rate": round(false_positive_rate, 2),
            "total_findings": total,
            "confirmed": confirmed,
            "false_positives": false_positives,
            "helpful": helpful,
        }

    def get_per_category_accuracy(self) -> Dict[FindingCategory, Dict[str, float]]:
        """
        Get accuracy metrics broken down by category.

        Returns:
            Dict mapping category to accuracy metrics
        """
        results = {}
        for category in FindingCategory:
            results[category] = self.calculate_accuracy(category=category)
        return results

    def get_per_severity_accuracy(self) -> Dict[FindingSeverity, Dict[str, float]]:
        """
        Get accuracy metrics broken down by severity.

        Returns:
            Dict mapping severity to accuracy metrics
        """
        results = {}
        for severity in FindingSeverity:
            results[severity] = self.calculate_accuracy(severity=severity)
        return results


class PatternLearner:
    """Identifies patterns in what findings are acted on."""

    def __init__(self, db: Session):
        """
        Initialize pattern learner.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def get_acted_on_patterns(self) -> Dict[FindingCategory, Dict[str, int]]:
        """
        Identify which finding types are most often acted on.

        Returns:
            Dict with patterns and action counts
        """
        feedback_list = self.db.query(FB).all()

        patterns = {}
        for category in FindingCategory:
            cat_feedback = [f for f in feedback_list if f.finding_id in self._get_finding_ids_by_category(category)]
            acted = sum(1 for f in cat_feedback if f.helpful is True or f.validation == IssueValidation.CONFIRMED)
            dismissed = sum(1 for f in cat_feedback if f.feedback_type == FeedbackType.FALSE_POSITIVE)
            patterns[category] = {
                "total": len(cat_feedback),
                "acted_on": acted,
                "dismissed": dismissed,
                "action_rate": (acted / len(cat_feedback) * 100) if cat_feedback else 0,
            }

        return patterns

    def _get_finding_ids_by_category(self, category: FindingCategory) -> list:
        """Get all finding IDs in a category."""
        results = self.db.query(Finding.id).filter(Finding.category == category).all()
        return [r[0] for r in results]

    def get_most_acted_categories(self, limit: int = 5) -> list:
        """
        Get the categories with highest action rate.

        Args:
            limit: Number of top categories to return

        Returns:
            List of (category, action_rate) tuples
        """
        patterns = self.get_acted_on_patterns()
        sorted_patterns = sorted(
            patterns.items(),
            key=lambda x: x[1]["action_rate"],
            reverse=True
        )
        return sorted_patterns[:limit]


class PersonalThresholdCalculator:
    """Learns personal thresholds for different types of findings."""

    def __init__(self, db: Session):
        """
        Initialize threshold calculator.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def calculate_confidence_threshold(
        self,
        category: FindingCategory,
        severity: FindingSeverity,
    ) -> float:
        """
        Calculate personal confidence threshold for a category/severity combo.

        Based on historical accuracy of that combination.

        Args:
            category: Finding category
            severity: Finding severity

        Returns:
            Confidence threshold (0.0-1.0)
        """
        metrics = self.db.query(LearningMetrics).filter(
            LearningMetrics.category == category,
            LearningMetrics.severity == severity,
        ).first()

        if metrics:
            # Use accuracy as basis for confidence threshold
            # High accuracy = higher threshold
            return metrics.accuracy / 100.0

        # Default threshold if no data yet
        return 0.5

    def get_personal_thresholds(self) -> Dict[Tuple[FindingCategory, FindingSeverity], float]:
        """
        Get all personal thresholds.

        Returns:
            Dict mapping (category, severity) to threshold
        """
        thresholds = {}
        for category in FindingCategory:
            for severity in FindingSeverity:
                key = (category, severity)
                thresholds[key] = self.calculate_confidence_threshold(category, severity)
        return thresholds

    def should_include_finding(
        self,
        category: FindingCategory,
        severity: FindingSeverity,
        confidence: float,
    ) -> bool:
        """
        Decide if a finding should be included based on personal threshold.

        Args:
            category: Finding category
            severity: Finding severity
            confidence: System confidence in the finding (0.0-1.0)

        Returns:
            True if finding should be included
        """
        threshold = self.calculate_confidence_threshold(category, severity)
        return confidence >= threshold


class LearningEngine:
    """
    Complete learning engine combining all learning capabilities.

    Orchestrates accuracy tracking, pattern learning, and threshold calculation.
    """

    def __init__(self, db: Session):
        """
        Initialize learning engine.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self.accuracy = HistoricalAccuracy(db)
        self.patterns = PatternLearner(db)
        self.thresholds = PersonalThresholdCalculator(db)

    def update_learning_metrics(
        self,
        repo_url: str,
        category: FindingCategory,
        severity: FindingSeverity,
    ) -> None:
        """
        Update learning metrics for a repo/category/severity combination.

        Recalculates accuracy and updates confidence thresholds.

        Args:
            repo_url: Repository URL
            category: Finding category
            severity: Finding severity
        """
        # Get or create metrics record
        metrics = self.db.query(LearningMetrics).filter(
            LearningMetrics.repo_url == repo_url,
            LearningMetrics.category == category,
            LearningMetrics.severity == severity,
        ).first()

        if not metrics:
            metrics = LearningMetrics(
                repo_url=repo_url,
                category=category,
                severity=severity,
            )
            self.db.add(metrics)

        # Recalculate accuracy for this combo
        accuracy_data = self.accuracy.calculate_accuracy(
            category=category,
            severity=severity,
        )

        # Update metrics
        metrics.total_findings = accuracy_data["total_findings"]
        metrics.confirmed_findings = accuracy_data["confirmed"]
        metrics.false_positives = accuracy_data["false_positives"]
        metrics.accuracy = accuracy_data["accuracy"]
        metrics.precision = accuracy_data["precision"]
        metrics.recall = accuracy_data["recall"]
        metrics.false_positive_rate = accuracy_data["false_positive_rate"]
        metrics.confidence_threshold = self.thresholds.calculate_confidence_threshold(
            category, severity
        )
        metrics.updated_at = datetime.now(timezone.utc)

        self.db.commit()

    def get_learning_report(self) -> Dict:
        """
        Generate a comprehensive learning report.

        Returns:
            Dict with overall metrics, trends, and recommendations
        """
        overall_accuracy = self.accuracy.calculate_accuracy()
        category_accuracy = self.accuracy.get_per_category_accuracy()
        patterns = self.patterns.get_acted_on_patterns()
        thresholds = self.thresholds.get_personal_thresholds()

        return {
            "overall": overall_accuracy,
            "by_category": category_accuracy,
            "patterns": patterns,
            "thresholds": thresholds,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
