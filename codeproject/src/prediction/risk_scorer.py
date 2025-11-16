"""
Risk Scoring System

Calculates risk scores for files based on multiple factors including
churn rate, complexity trends, test coverage, and bug history.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

from src.prediction.history_tracker import HistoryDatabase


class RiskLevel(str, Enum):
    """Risk level classification."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskScore:
    """Represents risk assessment for a file."""

    file_path: str
    total_score: float  # 0-1, higher = more risk
    risk_level: RiskLevel
    churn_score: float  # 0-1
    complexity_score: float  # 0-1
    coverage_score: float  # 0-1
    bug_history_score: float  # 0-1
    coupling_score: float  # 0-1
    age_score: float  # 0-1
    contributing_factors: Dict[str, str]  # Factor -> explanation


class RiskScorer:
    """Calculates risk scores for files based on historical patterns."""

    def __init__(self, history_db: HistoryDatabase):
        """
        Initialize risk scorer.

        Args:
            history_db: Historical database for trend analysis
        """
        self.history_db = history_db
        self.weights = {
            "churn": 0.25,
            "complexity": 0.20,
            "coverage": 0.20,
            "bug_history": 0.20,
            "coupling": 0.10,
            "age": 0.05,
        }

    def score_file(
        self,
        file_path: str,
        current_complexity: float,
        current_coverage: float,
        coupling_score: float = 0.0,
    ) -> RiskScore:
        """
        Calculate risk score for a file.

        Args:
            file_path: Path to file
            current_complexity: Current cyclomatic complexity
            current_coverage: Current test coverage (0-1)
            coupling_score: Current coupling score (0-1)

        Returns:
            RiskScore with detailed breakdown
        """
        # Calculate component scores
        churn = self._score_churn(file_path)
        complexity = self._score_complexity(file_path, current_complexity)
        coverage = self._score_coverage(current_coverage)
        bug_history = self._score_bug_history(file_path)
        coupling = self._score_coupling(coupling_score)
        age = self._score_age(file_path)

        # Weighted combination
        total_score = (
            churn * self.weights["churn"]
            + complexity * self.weights["complexity"]
            + coverage * self.weights["coverage"]
            + bug_history * self.weights["bug_history"]
            + coupling * self.weights["coupling"]
            + age * self.weights["age"]
        )

        # Determine risk level
        risk_level = self._score_to_level(total_score)

        # Identify contributing factors
        factors = {}
        if churn > 0.7:
            factors["high_churn"] = f"File changed {churn * 100:.0f}% of the time"
        if complexity > 0.7:
            factors["high_complexity"] = f"Complexity increasing at {complexity * 100:.0f}%"
        if coverage < 0.3:
            factors["low_coverage"] = f"Test coverage only {coverage * 100:.0f}%"
        if bug_history > 0.7:
            factors["high_bug_history"] = f"History of bugs: {bug_history * 100:.0f}% risk"
        if coupling > 0.7:
            factors["high_coupling"] = f"Highly coupled module ({coupling * 100:.0f}%)"
        if age > 0.6:
            factors["old_code"] = "Hasn't been maintained recently"

        return RiskScore(
            file_path=file_path,
            total_score=round(total_score, 2),
            risk_level=risk_level,
            churn_score=round(churn, 2),
            complexity_score=round(complexity, 2),
            coverage_score=round(coverage, 2),
            bug_history_score=round(bug_history, 2),
            coupling_score=round(coupling, 2),
            age_score=round(age, 2),
            contributing_factors=factors,
        )

    def _score_churn(self, file_path: str) -> float:
        """Score based on how often file changes (churn rate)."""
        churn_rate = self.history_db.get_file_churn_rate(file_path, days=30)
        # Normalize: 10+ changes in a month = high risk
        return min(1.0, churn_rate / 10.0)

    def _score_complexity(self, file_path: str, current_complexity: float) -> float:
        """Score based on complexity and its trend."""
        trend = self.history_db.get_complexity_trend(file_path, days=90)

        if not trend:
            # No history, use current complexity
            return min(1.0, current_complexity / 20.0)

        # Check if complexity is increasing
        if len(trend) > 1:
            avg_past = sum(trend[:-1]) / len(trend[:-1])
            current = trend[-1]
            increasing = current > avg_past
        else:
            increasing = False

        base_score = min(1.0, current_complexity / 20.0)
        penalty = 0.3 if increasing else 0.0

        return min(1.0, base_score + penalty)

    def _score_coverage(self, current_coverage: float) -> float:
        """Score based on test coverage (inverted: low coverage = high risk)."""
        # 0% coverage = 1.0 risk, 100% coverage = 0.0 risk
        return 1.0 - current_coverage

    def _score_bug_history(self, file_path: str) -> float:
        """Score based on past bugs in the file."""
        bug_count = self.history_db.get_bug_history(file_path, days=180)
        # Normalize: 5+ bugs in 6 months = high risk
        return min(1.0, bug_count / 5.0)

    def _score_coupling(self, coupling_score: float) -> float:
        """Score based on module coupling."""
        # Coupling score already 0-1
        return coupling_score

    def _score_age(self, file_path: str) -> float:
        """Score based on how long since file was modified (age)."""
        history = self.history_db.get_file_history(file_path)

        if not history:
            return 0.0  # New file, low age risk

        from datetime import datetime

        last_modified = history[-1]["timestamp"]
        days_since = (datetime.now() - last_modified).days

        # Score: 180+ days without change = high risk
        return min(1.0, days_since / 180.0)

    def _score_to_level(self, score: float) -> RiskLevel:
        """Convert numerical score to risk level."""
        if score >= 0.75:
            return RiskLevel.CRITICAL
        elif score >= 0.5:
            return RiskLevel.HIGH
        elif score >= 0.25:
            return RiskLevel.MEDIUM
        else:
            return RiskLevel.LOW

    def score_multiple_files(
        self,
        file_metrics: Dict[str, Dict],
    ) -> Dict[str, RiskScore]:
        """
        Score multiple files at once.

        Args:
            file_metrics: Dict of file_path -> {complexity, coverage, coupling}

        Returns:
            Dict of file_path -> RiskScore
        """
        scores = {}
        for file_path, metrics in file_metrics.items():
            scores[file_path] = self.score_file(
                file_path,
                metrics.get("complexity", 0.0),
                metrics.get("coverage", 0.0),
                metrics.get("coupling", 0.0),
            )
        return scores

    def get_critical_files(
        self,
        file_metrics: Dict[str, Dict],
    ) -> list:
        """
        Get files with CRITICAL risk level.

        Args:
            file_metrics: Dict of file_path -> metrics

        Returns:
            List of (file_path, risk_score) tuples sorted by risk
        """
        scores = self.score_multiple_files(file_metrics)
        critical = [
            (path, score)
            for path, score in scores.items()
            if score.risk_level == RiskLevel.CRITICAL
        ]
        return sorted(critical, key=lambda x: x[1].total_score, reverse=True)

    def get_high_risk_files(
        self,
        file_metrics: Dict[str, Dict],
    ) -> list:
        """
        Get files with HIGH or CRITICAL risk.

        Args:
            file_metrics: Dict of file_path -> metrics

        Returns:
            List of (file_path, risk_score) tuples sorted by risk
        """
        scores = self.score_multiple_files(file_metrics)
        high_risk = [
            (path, score)
            for path, score in scores.items()
            if score.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]
        ]
        return sorted(high_risk, key=lambda x: x[1].total_score, reverse=True)
