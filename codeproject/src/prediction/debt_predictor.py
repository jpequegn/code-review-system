"""
Technical Debt Trajectory Prediction

Tracks technical debt accumulation and predicts when
critical debt levels will be reached.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
from datetime import datetime, timedelta

from src.prediction.history_tracker import HistoryDatabase


@dataclass
class DebtMetrics:
    """Metrics for different types of technical debt."""

    complexity_debt: float  # How much over target complexity
    coupling_debt: float  # How much decoupling is needed
    test_debt: float  # Coverage gaps as percentage
    doc_debt: float  # Undocumented code percentage
    total_debt_score: float  # Overall debt level (0-1)


class DebtPredictor:
    """Predicts technical debt trajectory and thresholds."""

    def __init__(self, history_db: HistoryDatabase):
        """
        Initialize debt predictor.

        Args:
            history_db: Historical database
        """
        self.history_db = history_db
        self.target_complexity = 10.0  # Target cyclomatic complexity
        self.target_coupling = 3.0  # Target afferent coupling
        self.target_coverage = 0.80  # Target test coverage (80%)

    def calculate_debt_metrics(
        self,
        avg_complexity: float,
        avg_coupling: float,
        avg_coverage: float,
        undocumented_ratio: float,
    ) -> DebtMetrics:
        """
        Calculate current technical debt metrics.

        Args:
            avg_complexity: Average cyclomatic complexity
            avg_coupling: Average coupling score
            avg_coverage: Average test coverage (0-1)
            undocumented_ratio: Ratio of undocumented code (0-1)

        Returns:
            DebtMetrics with breakdown
        """
        # Complexity debt = how much above target
        complexity_debt = max(0.0, (avg_complexity - self.target_complexity) / self.target_complexity)

        # Coupling debt = how much above target
        coupling_debt = max(0.0, (avg_coupling - self.target_coupling) / self.target_coupling)

        # Test debt = coverage gap
        test_debt = max(0.0, self.target_coverage - avg_coverage)

        # Doc debt = undocumented percentage
        doc_debt = undocumented_ratio

        # Total debt score
        total_debt_score = (complexity_debt * 0.35 + coupling_debt * 0.25 + test_debt * 0.25 + doc_debt * 0.15)
        total_debt_score = min(1.0, total_debt_score)

        return DebtMetrics(
            complexity_debt=round(complexity_debt, 2),
            coupling_debt=round(coupling_debt, 2),
            test_debt=round(test_debt, 2),
            doc_debt=round(doc_debt, 2),
            total_debt_score=round(total_debt_score, 2),
        )

    def predict_debt_trajectory(
        self,
        current_metrics: DebtMetrics,
        days_ahead: int = 90,
    ) -> Dict[str, list]:
        """
        Predict how debt will accumulate over time.

        Args:
            current_metrics: Current debt metrics
            days_ahead: How many days to predict

        Returns:
            Dict with projected trajectory
        """
        trajectory = {
            "dates": [],
            "complexity_debt": [],
            "coupling_debt": [],
            "test_debt": [],
            "total_debt": [],
        }

        # Get historical trend
        snapshots = self.history_db.get_snapshots_since(90)

        if len(snapshots) < 2:
            # Not enough data, assume linear growth
            daily_growth = current_metrics.total_debt_score / 100.0  # Slow growth
        else:
            # Calculate growth rate
            old_debt = self._calculate_debt_from_snapshot(snapshots[0])
            new_debt = self._calculate_debt_from_snapshot(snapshots[-1])
            days_elapsed = (snapshots[-1].timestamp - snapshots[0].timestamp).days
            daily_growth = (new_debt.total_debt_score - old_debt.total_debt_score) / max(1, days_elapsed)

        # Project forward
        for day in range(0, days_ahead + 1, 7):  # Weekly snapshots
            date = datetime.now() + timedelta(days=day)
            trajectory["dates"].append(date.strftime("%Y-%m-%d"))

            # Linear projection
            weeks = day / 7
            projected_total = current_metrics.total_debt_score + (daily_growth * day)
            projected_total = min(1.0, max(0.0, projected_total))

            # Proportional growth for each component
            complexity = current_metrics.complexity_debt + (daily_growth * 0.35 * day)
            coupling = current_metrics.coupling_debt + (daily_growth * 0.25 * day)
            test = current_metrics.test_debt + (daily_growth * 0.25 * day)

            trajectory["complexity_debt"].append(round(min(1.0, complexity), 2))
            trajectory["coupling_debt"].append(round(min(1.0, coupling), 2))
            trajectory["test_debt"].append(round(min(1.0, test), 2))
            trajectory["total_debt"].append(round(projected_total, 2))

        return trajectory

    def days_to_critical_debt(self, current_metrics: DebtMetrics) -> Optional[int]:
        """
        Predict how many days until debt reaches critical level.

        Args:
            current_metrics: Current debt metrics

        Returns:
            Days until critical (None if already critical or declining)
        """
        snapshots = self.history_db.get_snapshots_since(90)

        if len(snapshots) < 2:
            return None

        # Calculate growth rate
        old_debt = self._calculate_debt_from_snapshot(snapshots[0])
        new_debt = self._calculate_debt_from_snapshot(snapshots[-1])
        days_elapsed = (snapshots[-1].timestamp - snapshots[0].timestamp).days

        daily_growth = (new_debt.total_debt_score - old_debt.total_debt_score) / max(1, days_elapsed)

        if daily_growth <= 0:
            return None  # Debt is declining

        if current_metrics.total_debt_score >= 0.8:
            return 0  # Already critical

        # Days to reach 0.8 (critical)
        critical_threshold = 0.8
        days_to_critical = (critical_threshold - current_metrics.total_debt_score) / daily_growth

        return max(0, int(days_to_critical))

    def get_debt_breakdown(self, metrics: DebtMetrics) -> Dict[str, str]:
        """
        Get human-readable debt breakdown.

        Args:
            metrics: Debt metrics

        Returns:
            Dict with explanations
        """
        breakdown = {}

        if metrics.complexity_debt > 0.1:
            excess = metrics.complexity_debt * 100
            breakdown["complexity"] = (
                f"Code complexity is {excess:.0f}% above target. "
                f"Consider breaking down large functions."
            )

        if metrics.coupling_debt > 0.1:
            excess = metrics.coupling_debt * 100
            breakdown["coupling"] = (
                f"Module coupling is {excess:.0f}% above target. "
                f"Consider extracting shared code."
            )

        if metrics.test_debt > 0.1:
            gap = metrics.test_debt * 100
            breakdown["testing"] = (
                f"Test coverage gap of {gap:.0f}%. "
                f"Need {gap:.0f}% more coverage to reach target."
            )

        if metrics.doc_debt > 0.2:
            undoc = metrics.doc_debt * 100
            breakdown["documentation"] = (
                f"{undoc:.0f}% of code is undocumented. "
                f"Add docstrings and comments to critical sections."
            )

        return breakdown

    def get_refactoring_timeline(self, metrics: DebtMetrics) -> List[tuple]:
        """
        Get suggested refactoring timeline.

        Args:
            metrics: Debt metrics

        Returns:
            List of (priority, task, estimated_hours)
        """
        timeline = []

        # Prioritize by impact
        if metrics.complexity_debt > 0.2:
            timeline.append(("HIGH", "Reduce complexity in high-impact modules", 40))

        if metrics.test_debt > 0.15:
            timeline.append(("HIGH", "Increase test coverage to 80%", 30))

        if metrics.coupling_debt > 0.15:
            timeline.append(("MEDIUM", "Decouple tightly coupled modules", 20))

        if metrics.doc_debt > 0.3:
            timeline.append(("MEDIUM", "Add documentation to critical code", 15))

        if metrics.complexity_debt > 0.1:
            timeline.append(("LOW", "Refactor remaining complex functions", 25))

        return timeline

    def get_debt_health_status(self, metrics: DebtMetrics) -> str:
        """
        Get overall debt health status.

        Args:
            metrics: Debt metrics

        Returns:
            Status string
        """
        score = metrics.total_debt_score

        if score < 0.2:
            return "HEALTHY - Low technical debt"
        elif score < 0.4:
            return "GOOD - Manageable debt level"
        elif score < 0.6:
            return "WARNING - Debt accumulating"
        elif score < 0.8:
            return "CRITICAL - High debt level"
        else:
            return "CRITICAL - Unmaintainable debt level"

    def compare_to_baseline(
        self,
        current_metrics: DebtMetrics,
        baseline_metrics: DebtMetrics,
    ) -> Dict[str, str]:
        """
        Compare current debt to baseline.

        Args:
            current_metrics: Current debt
            baseline_metrics: Baseline/target debt

        Returns:
            Dict with comparisons
        """
        comparisons = {}

        if current_metrics.complexity_debt > baseline_metrics.complexity_debt:
            diff = (current_metrics.complexity_debt - baseline_metrics.complexity_debt) * 100
            comparisons["complexity"] = f"↑ {diff:.0f}% worse than baseline"
        else:
            diff = (baseline_metrics.complexity_debt - current_metrics.complexity_debt) * 100
            comparisons["complexity"] = f"↓ {diff:.0f}% better than baseline"

        if current_metrics.test_debt > baseline_metrics.test_debt:
            diff = (current_metrics.test_debt - baseline_metrics.test_debt) * 100
            comparisons["testing"] = f"↑ {diff:.0f}% worse than baseline"
        else:
            diff = (baseline_metrics.test_debt - current_metrics.test_debt) * 100
            comparisons["testing"] = f"↓ {diff:.0f}% better than baseline"

        return comparisons

    def _calculate_debt_from_snapshot(self, snapshot) -> DebtMetrics:
        """Calculate debt from a historical snapshot."""
        avg_complexity = sum(snapshot.complexity_trend.values()) / max(1, len(snapshot.complexity_trend))
        avg_coverage = sum(snapshot.test_coverage_trend.values()) / max(1, len(snapshot.test_coverage_trend))
        avg_coupling = sum(snapshot.coupling_trend.values()) / max(1, len(snapshot.coupling_trend))

        return self.calculate_debt_metrics(avg_complexity, avg_coupling, avg_coverage, 0.2)
