"""
Learning Report Generator

Generates weekly/monthly insights about system accuracy and improvements.
Shows trending accuracy, most common issues, and learning metrics.
"""

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from sqlalchemy.orm import Session

from src.database import (
    FindingFeedback, FeedbackType, IssueValidation,
    ProductionIssue, LearningMetrics, FindingCategory, FindingSeverity
)
from src.learning.learner import LearningEngine, HistoricalAccuracy


class LearningReportGenerator:
    """Generates learning reports for insights and improvement tracking."""

    def __init__(self, db: Session):
        """
        Initialize report generator.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self.learning_engine = LearningEngine(db)
        self.accuracy = HistoricalAccuracy(db)

    def generate_weekly_report(self) -> Dict:
        """
        Generate a weekly learning report.

        Returns:
            Dict with weekly metrics, trends, and insights
        """
        return self._generate_period_report(days=7, period_name="Weekly")

    def generate_monthly_report(self) -> Dict:
        """
        Generate a monthly learning report.

        Returns:
            Dict with monthly metrics, trends, and insights
        """
        return self._generate_period_report(days=30, period_name="Monthly")

    def generate_overall_report(self) -> Dict:
        """
        Generate an overall learning report across all time.

        Returns:
            Dict with overall metrics and insights
        """
        return self._generate_period_report(days=None, period_name="Overall")

    def _generate_period_report(self, days: Optional[int], period_name: str) -> Dict:
        """
        Generate a report for a specific period.

        Args:
            days: Number of days to include (None for all time)
            period_name: Name of the period for reporting

        Returns:
            Dict with report data
        """
        # Get feedback data for period
        query = self.db.query(FindingFeedback)
        if days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            query = query.filter(FindingFeedback.created_at >= cutoff)

        feedback_list = query.all()

        # Get accuracy metrics
        accuracy_metrics = self.accuracy.calculate_accuracy(days=days)

        # Count feedback types
        feedback_counts = self._count_feedback_types(feedback_list)

        # Get most common findings
        top_findings = self._get_top_findings(feedback_list)

        # Get production issues
        production_issues = self._get_production_issues(days)

        # Get improvement trends
        improvement_trend = self._calculate_improvement_trend(days)

        # Generate recommendations
        recommendations = self._generate_recommendations(feedback_list, accuracy_metrics)

        return {
            "period": period_name,
            "date_range": {
                "start": (datetime.now(timezone.utc) - timedelta(days=days)).isoformat() if days else "All time",
                "end": datetime.now(timezone.utc).isoformat(),
            },
            "accuracy_metrics": accuracy_metrics,
            "feedback_counts": feedback_counts,
            "top_findings": top_findings,
            "production_issues": production_issues,
            "improvement_trend": improvement_trend,
            "recommendations": recommendations,
            "summary": self._generate_summary(accuracy_metrics, feedback_counts),
        }

    def _count_feedback_types(self, feedback_list: List[FindingFeedback]) -> Dict[str, int]:
        """Count feedback by type."""
        counts = {
            "helpful": sum(1 for f in feedback_list if f.feedback_type == FeedbackType.HELPFUL),
            "false_positive": sum(1 for f in feedback_list if f.feedback_type == FeedbackType.FALSE_POSITIVE),
            "missed": sum(1 for f in feedback_list if f.feedback_type == FeedbackType.MISSED),
            "resolved": sum(1 for f in feedback_list if f.feedback_type == FeedbackType.RESOLVED),
            "recurring": sum(1 for f in feedback_list if f.feedback_type == FeedbackType.RECURRING),
        }
        return counts

    def _get_top_findings(self, feedback_list: List[FindingFeedback], limit: int = 5) -> List[Dict]:
        """Get most common finding types in feedback."""
        finding_types = {}
        for feedback in feedback_list:
            key = f"{feedback.feedback_type.value}"
            finding_types[key] = finding_types.get(key, 0) + 1

        top = sorted(finding_types.items(), key=lambda x: x[1], reverse=True)[:limit]
        return [{"type": k, "count": v} for k, v in top]

    def _get_production_issues(self, days: Optional[int]) -> List[Dict]:
        """Get production issues for the period."""
        query = self.db.query(ProductionIssue)
        if days:
            cutoff = datetime.now(timezone.utc) - timedelta(days=days)
            query = query.filter(ProductionIssue.date_discovered >= cutoff)

        issues = query.all()
        return [
            {
                "description": issue.description,
                "severity": issue.severity.value,
                "date": issue.date_discovered.isoformat(),
                "time_to_fix_minutes": issue.time_to_fix_minutes,
            }
            for issue in issues
        ]

    def _calculate_improvement_trend(self, days: Optional[int]) -> Dict:
        """Calculate if accuracy is improving over time."""
        if not days or days < 14:
            # Need at least 2 weeks of data for trend
            return {"status": "insufficient_data"}

        # Split period in half
        half_days = days // 2
        cutoff_recent = datetime.now(timezone.utc) - timedelta(days=half_days)
        cutoff_old = datetime.now(timezone.utc) - timedelta(days=days)

        # Get accuracy for first half
        query_old = self.db.query(FindingFeedback).filter(
            FindingFeedback.created_at < cutoff_recent,
            FindingFeedback.created_at >= cutoff_old
        )
        old_accuracy = self.accuracy.calculate_accuracy()

        # Get accuracy for second half
        query_recent = self.db.query(FindingFeedback).filter(
            FindingFeedback.created_at >= cutoff_recent
        )
        recent_accuracy = self.accuracy.calculate_accuracy()

        # Calculate trend
        if old_accuracy["total_findings"] > 0 and recent_accuracy["total_findings"] > 0:
            old_rate = old_accuracy["accuracy"]
            recent_rate = recent_accuracy["accuracy"]
            trend = recent_rate - old_rate

            return {
                "status": "improving" if trend > 2 else "stable" if trend > -2 else "declining",
                "trend": round(trend, 2),
                "previous_accuracy": round(old_rate, 2),
                "current_accuracy": round(recent_rate, 2),
            }

        return {"status": "insufficient_data"}

    def _generate_recommendations(
        self,
        feedback_list: List[FindingFeedback],
        accuracy_metrics: Dict,
    ) -> List[str]:
        """Generate recommendations based on metrics."""
        recommendations = []

        # Check false positive rate
        if accuracy_metrics["false_positive_rate"] > 30:
            recommendations.append(
                f"High false positive rate ({accuracy_metrics['false_positive_rate']:.1f}%). "
                "Consider adjusting severity thresholds for frequently dismissed findings."
            )

        # Check recall
        if accuracy_metrics["recall"] < 50:
            recommendations.append(
                "Low recall rate - many issues are being missed. "
                "Review recent production issues to identify patterns."
            )

        # Check total feedback
        total_feedback = len(feedback_list)
        if total_feedback < 10:
            recommendations.append(
                "Limited feedback data. Provide more feedback to improve learning accuracy."
            )

        # Check for recurring issues
        recurring_count = sum(1 for f in feedback_list if f.feedback_type == FeedbackType.RECURRING)
        if recurring_count > 2:
            recommendations.append(
                f"Found {recurring_count} recurring issues. "
                "Consider addressing root causes to reduce repetition."
            )

        if not recommendations:
            recommendations.append(
                "System performing well! Continue providing feedback to maintain accuracy."
            )

        return recommendations

    def _generate_summary(self, accuracy_metrics: Dict, feedback_counts: Dict) -> str:
        """Generate a human-readable summary."""
        total = accuracy_metrics["total_findings"]
        accuracy = accuracy_metrics["accuracy"]

        if total == 0:
            return "No feedback data available for this period."

        summary = (
            f"System analyzed {total} findings with {accuracy:.1f}% accuracy. "
            f"Received {feedback_counts['helpful']} helpful votes and "
            f"{feedback_counts['false_positive']} false positive reports."
        )
        return summary
