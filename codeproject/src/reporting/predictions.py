"""
Prediction Dashboard and Reporting

Generates reports on predicted risks, trends, and recommendations
for improving code quality based on prediction models.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from src.prediction.history_tracker import HistoryDatabase
from src.prediction.risk_scorer import RiskScorer
from src.prediction.failure_predictor import FailurePredictor
from src.prediction.temporal_analysis import TemporalAnalyzer
from src.prediction.debt_predictor import DebtPredictor, DebtMetrics
from src.prediction.pattern_learner import PatternLearner


@dataclass
class PredictionReport:
    """Weekly or monthly prediction report."""

    report_type: str  # "weekly" or "monthly"
    generated_at: datetime
    report_period: str  # "Week of 2024-01-15" or "January 2024"
    at_risk_files: List[tuple]  # (file_path, risk_score)
    failure_predictions: List[tuple]  # (file_path, prediction)
    personal_patterns: List[str]  # Your learned patterns
    debt_metrics: DebtMetrics
    debt_trajectory: Dict
    temporal_insights: List[str]
    recommendations: List[str]
    accuracy_metrics: Dict  # How accurate were last month's predictions
    trend_analysis: Dict  # Quality improving/declining


class PredictionReportGenerator:
    """Generates prediction reports."""

    def __init__(
        self,
        history_db: HistoryDatabase,
        risk_scorer: RiskScorer,
        failure_predictor: FailurePredictor,
        temporal_analyzer: TemporalAnalyzer,
        debt_predictor: DebtPredictor,
        pattern_learner: PatternLearner,
    ):
        """
        Initialize report generator.

        Args:
            history_db: Historical database
            risk_scorer: Risk scoring engine
            failure_predictor: Failure prediction engine
            temporal_analyzer: Temporal analysis engine
            debt_predictor: Debt prediction engine
            pattern_learner: Pattern learning engine
        """
        self.history_db = history_db
        self.risk_scorer = risk_scorer
        self.failure_predictor = failure_predictor
        self.temporal_analyzer = temporal_analyzer
        self.debt_predictor = debt_predictor
        self.pattern_learner = pattern_learner

    def generate_weekly_report(
        self,
        file_metrics: Dict[str, Dict],
    ) -> PredictionReport:
        """
        Generate weekly prediction report.

        Args:
            file_metrics: Current file metrics

        Returns:
            Weekly prediction report
        """
        now = datetime.now()
        week_start = now - __import__("datetime").timedelta(days=now.weekday())
        report_period = f"Week of {week_start.strftime('%Y-%m-%d')}"

        # Get at-risk files
        at_risk = self.risk_scorer.get_high_risk_files(file_metrics)

        # Get failure predictions
        predictions = self.failure_predictor.predict_multiple_files(file_metrics)[:5]

        # Get personal patterns
        patterns = self.pattern_learner.get_weak_spots()
        pattern_strs = [
            f"{p[0].description} ({p[1]} occurrences)"
            for p in patterns
        ]

        # Get debt metrics
        avg_complexity = sum(m.get("complexity", 0) for m in file_metrics.values()) / max(1, len(file_metrics))
        avg_coupling = sum(m.get("coupling", 0) for m in file_metrics.values()) / max(1, len(file_metrics))
        avg_coverage = sum(m.get("coverage", 0) for m in file_metrics.values()) / max(1, len(file_metrics))
        debt_metrics = self.debt_predictor.calculate_debt_metrics(avg_complexity, avg_coupling, avg_coverage, 0.15)

        # Get debt trajectory
        debt_trajectory = self.debt_predictor.predict_debt_trajectory(debt_metrics, days_ahead=7)

        # Get temporal insights
        temporal_insights = self.temporal_analyzer.get_productivity_insights()

        # Generate recommendations
        recommendations = self._generate_weekly_recommendations(
            at_risk, predictions, debt_metrics, patterns
        )

        # Trend analysis
        trend = self._analyze_quality_trend()

        return PredictionReport(
            report_type="weekly",
            generated_at=now,
            report_period=report_period,
            at_risk_files=at_risk[:5],
            failure_predictions=predictions,
            personal_patterns=pattern_strs,
            debt_metrics=debt_metrics,
            debt_trajectory=debt_trajectory,
            temporal_insights=temporal_insights,
            recommendations=recommendations,
            accuracy_metrics={},  # Would compare to actual results
            trend_analysis=trend,
        )

    def generate_monthly_report(
        self,
        file_metrics: Dict[str, Dict],
    ) -> PredictionReport:
        """
        Generate monthly prediction report.

        Args:
            file_metrics: Current file metrics

        Returns:
            Monthly prediction report
        """
        now = datetime.now()
        month_start = now.replace(day=1)
        report_period = month_start.strftime("%B %Y")

        # Get at-risk files
        at_risk = self.risk_scorer.get_high_risk_files(file_metrics)

        # Get failure predictions
        predictions = self.failure_predictor.predict_multiple_files(file_metrics)[:10]

        # Get personal patterns
        patterns = self.pattern_learner.get_weak_spots()
        pattern_strs = [
            f"{p[0].description} ({p[1]} occurrences)"
            for p in patterns
        ]

        # Get debt metrics
        avg_complexity = sum(m.get("complexity", 0) for m in file_metrics.values()) / max(1, len(file_metrics))
        avg_coupling = sum(m.get("coupling", 0) for m in file_metrics.values()) / max(1, len(file_metrics))
        avg_coverage = sum(m.get("coverage", 0) for m in file_metrics.values()) / max(1, len(file_metrics))
        debt_metrics = self.debt_predictor.calculate_debt_metrics(avg_complexity, avg_coupling, avg_coverage, 0.15)

        # Get debt trajectory (30 days)
        debt_trajectory = self.debt_predictor.predict_debt_trajectory(debt_metrics, days_ahead=30)

        # Get temporal insights
        temporal_insights = self.temporal_analyzer.get_productivity_insights()

        # Generate recommendations
        recommendations = self._generate_monthly_recommendations(
            at_risk, predictions, debt_metrics, patterns
        )

        # Trend analysis
        trend = self._analyze_quality_trend(days=30)

        return PredictionReport(
            report_type="monthly",
            generated_at=now,
            report_period=report_period,
            at_risk_files=at_risk[:10],
            failure_predictions=predictions,
            personal_patterns=pattern_strs,
            debt_metrics=debt_metrics,
            debt_trajectory=debt_trajectory,
            temporal_insights=temporal_insights,
            recommendations=recommendations,
            accuracy_metrics=self._calculate_prediction_accuracy(),
            trend_analysis=trend,
        )

    def format_report_as_text(self, report: PredictionReport) -> str:
        """
        Format report as readable text.

        Args:
            report: Prediction report

        Returns:
            Formatted report text
        """
        lines = []

        lines.append(f"{'=' * 60}")
        lines.append(f"PREDICTION REPORT - {report.report_type.upper()}")
        lines.append(f"Period: {report.report_period}")
        lines.append(f"Generated: {report.generated_at.strftime('%Y-%m-%d %H:%M')}")
        lines.append(f"{'=' * 60}\n")

        # At-risk files
        lines.append("📋 FILES AT RISK")
        if report.at_risk_files:
            for file_path, risk_score in report.at_risk_files:
                lines.append(f"  {file_path}: {risk_score.risk_level.value} ({risk_score.total_score})")
        else:
            lines.append("  No high-risk files detected")
        lines.append("")

        # Failure predictions
        lines.append("⚠️ FAILURE PREDICTIONS")
        if report.failure_predictions:
            for file_path, prediction in report.failure_predictions[:3]:
                lines.append(f"  {file_path}: {prediction.likelihood:.0%} likelihood")
                if prediction.contributing_patterns:
                    lines.append(f"    Patterns: {', '.join(prediction.contributing_patterns[:2])}")
        lines.append("")

        # Personal patterns
        lines.append("🎯 YOUR PATTERNS")
        if report.personal_patterns:
            for pattern in report.personal_patterns[:3]:
                lines.append(f"  • {pattern}")
        lines.append("")

        # Debt status
        lines.append("💾 TECHNICAL DEBT")
        lines.append(f"  Status: {self.debt_predictor.get_debt_health_status(report.debt_metrics)}")
        lines.append(f"  Complexity Debt: {report.debt_metrics.complexity_debt:.0%}")
        lines.append(f"  Test Coverage Gap: {report.debt_metrics.test_debt:.0%}")
        lines.append(f"  Overall Debt Score: {report.debt_metrics.total_debt_score:.0%}")
        lines.append("")

        # Days to critical debt
        days_to_critical = self.debt_predictor.days_to_critical_debt(report.debt_metrics)
        if days_to_critical is not None:
            lines.append(f"  ⏰ Days until critical debt: {days_to_critical}")
        lines.append("")

        # Temporal insights
        lines.append("📅 PRODUCTIVITY PATTERNS")
        for insight in report.temporal_insights[:3]:
            lines.append(f"  • {insight}")
        lines.append("")

        # Recommendations
        lines.append("💡 RECOMMENDATIONS")
        for i, rec in enumerate(report.recommendations[:5], 1):
            lines.append(f"  {i}. {rec}")
        lines.append("")

        # Trend analysis
        if report.trend_analysis:
            lines.append("📊 TRENDS")
            for key, value in report.trend_analysis.items():
                lines.append(f"  {key}: {value}")

        lines.append(f"\n{'=' * 60}")
        return "\n".join(lines)

    def _generate_weekly_recommendations(self, at_risk, predictions, debt_metrics, patterns) -> List[str]:
        """Generate weekly recommendations."""
        recommendations = []

        if at_risk:
            recommendations.append(f"Review {len(at_risk)} files at risk this week")

        if debt_metrics.total_debt_score > 0.5:
            recommendations.append("Technical debt is accumulating - plan refactoring time")

        if patterns:
            top_pattern = patterns[0]
            recommendations.append(f"Focus on avoiding '{top_pattern[0].description}' pattern")

        if predictions:
            recommendations.append(f"Increase test coverage in predicted failure files")

        return recommendations[:5]

    def _generate_monthly_recommendations(self, at_risk, predictions, debt_metrics, patterns) -> List[str]:
        """Generate monthly recommendations."""
        recommendations = []

        if debt_metrics.total_debt_score > 0.6:
            timeline = self.debt_predictor.get_refactoring_timeline(debt_metrics)
            if timeline:
                task = timeline[0]
                recommendations.append(f"{task[0]} Priority: {task[1]} ({task[2]}h)")

        if patterns:
            suggestions = self.pattern_learner.get_improvement_suggestions()
            recommendations.extend(suggestions[:2])

        recommendations.append("Schedule architecture review to assess design quality")
        recommendations.append("Update documentation for high-churn files")

        return recommendations[:5]

    def _analyze_quality_trend(self, days: int = 7) -> Dict[str, str]:
        """Analyze quality trend."""
        trajectory = self.history_db.get_quality_trajectory(days)

        if not trajectory["issues_found"]:
            return {}

        recent_issues = trajectory["issues_found"][-7:]
        trend_direction = "📈 INCREASING" if recent_issues[-1] > recent_issues[0] else "📉 DECREASING"

        return {
            "quality_direction": trend_direction,
            "issues_trend": f"{recent_issues[0]} → {recent_issues[-1]} issues",
        }

    def _calculate_prediction_accuracy(self) -> Dict[str, float]:
        """Calculate how accurate predictions were."""
        # This would compare predicted issues vs actual issues
        # Placeholder: assume 70% accuracy initially
        return {
            "bug_prediction_accuracy": 0.70,
            "performance_prediction_accuracy": 0.65,
            "overall_accuracy": 0.68,
        }
