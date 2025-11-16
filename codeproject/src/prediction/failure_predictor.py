"""
Failure Prediction System

Predicts likelihood of failures in code based on historical patterns,
similar failures, and personal mistake patterns.
"""

from dataclasses import dataclass
from typing import List, Dict, Optional
from enum import Enum

from src.prediction.history_tracker import HistoryDatabase
from src.prediction.pattern_learner import PatternLearner, CodeContext, IssueType
from src.prediction.risk_scorer import RiskScorer


class FailureType(str, Enum):
    """Types of potential failures."""

    BUG = "bug"
    PERFORMANCE_ISSUE = "performance_issue"
    SECURITY_VULNERABILITY = "security_vulnerability"
    RELIABILITY_ISSUE = "reliability_issue"
    DATA_INTEGRITY = "data_integrity"


@dataclass
class PastFailure:
    """Record of a past failure."""

    file_path: str
    failure_type: FailureType
    commit_sha: str
    description: str


@dataclass
class FailurePrediction:
    """Prediction of potential failure."""

    file_path: str
    likelihood: float  # 0-1, higher = more likely to fail
    confidence: float  # 0-1, how sure we are
    predicted_failure_types: Dict[FailureType, float]  # Type -> likelihood
    contributing_patterns: List[str]  # What patterns contribute to this prediction
    similar_failures: List[PastFailure]  # Similar past failures
    recommendations: List[str]  # How to prevent this failure


class FailurePredictor:
    """Predicts failure likelihood based on multiple signals."""

    def __init__(
        self,
        history_db: HistoryDatabase,
        pattern_learner: PatternLearner,
        risk_scorer: RiskScorer,
    ):
        """
        Initialize failure predictor.

        Args:
            history_db: Historical database
            pattern_learner: Personal pattern learner
            risk_scorer: Risk scoring engine
        """
        self.history_db = history_db
        self.pattern_learner = pattern_learner
        self.risk_scorer = risk_scorer
        self.past_failures: List[PastFailure] = []

    def predict_failure(
        self,
        file_path: str,
        current_complexity: float,
        current_coverage: float,
        coupling_score: float = 0.0,
        code_context: Optional[CodeContext] = None,
    ) -> FailurePrediction:
        """
        Predict failure likelihood for a file.

        Args:
            file_path: Path to file
            current_complexity: Current complexity metric
            current_coverage: Current test coverage
            coupling_score: Current coupling score
            code_context: Optional code context hint

        Returns:
            FailurePrediction with details
        """
        # Get risk score
        risk_score = self.risk_scorer.score_file(
            file_path, current_complexity, current_coverage, coupling_score
        )

        # Get similar failures
        similar = self._find_similar_failures(file_path)

        # Get patterns that apply to this file
        patterns = self._get_applicable_patterns(file_path, code_context)

        # Calculate failure type likelihoods
        failure_types = self._predict_failure_types(
            file_path, code_context, current_complexity, current_coverage
        )

        # Overall likelihood = weighted combination of signals
        overall_likelihood = self._calculate_likelihood(
            risk_score.total_score,
            len(similar),
            patterns,
        )

        # Confidence = how much data we have
        confidence = self._calculate_confidence(file_path, len(similar), len(patterns))

        # Get recommendations
        recommendations = self._get_recommendations(
            file_path, risk_score, patterns, failure_types
        )

        return FailurePrediction(
            file_path=file_path,
            likelihood=round(overall_likelihood, 2),
            confidence=round(confidence, 2),
            predicted_failure_types=failure_types,
            contributing_patterns=[p.description for p in patterns],
            similar_failures=similar,
            recommendations=recommendations,
        )

    def predict_multiple_files(
        self,
        file_metrics: Dict[str, Dict],
    ) -> List[tuple]:
        """
        Predict failures for multiple files.

        Args:
            file_metrics: Dict of file_path -> {complexity, coverage, coupling}

        Returns:
            List of (file_path, prediction) sorted by likelihood
        """
        predictions = []

        for file_path, metrics in file_metrics.items():
            pred = self.predict_failure(
                file_path,
                metrics.get("complexity", 0.0),
                metrics.get("coverage", 0.0),
                metrics.get("coupling", 0.0),
            )
            predictions.append((file_path, pred))

        return sorted(predictions, key=lambda x: x[1].likelihood, reverse=True)

    def record_failure(
        self,
        file_path: str,
        failure_type: FailureType,
        commit_sha: str,
        description: str,
    ) -> None:
        """
        Record a failure to improve future predictions.

        Args:
            file_path: File where failure occurred
            failure_type: Type of failure
            commit_sha: Commit where failure was introduced
            description: Description of failure
        """
        failure = PastFailure(
            file_path=file_path,
            failure_type=failure_type,
            commit_sha=commit_sha,
            description=description,
        )
        self.past_failures.append(failure)

    def get_most_risky_files(
        self,
        file_metrics: Dict[str, Dict],
        limit: int = 10,
    ) -> List[tuple]:
        """
        Get files most likely to fail.

        Args:
            file_metrics: File metrics dictionary
            limit: Number of files to return

        Returns:
            List of (file_path, prediction) sorted by likelihood
        """
        predictions = self.predict_multiple_files(file_metrics)
        return [(path, pred) for path, pred in predictions if pred.likelihood > 0.5][
            :limit
        ]

    def get_high_confidence_predictions(
        self,
        file_metrics: Dict[str, Dict],
        min_confidence: float = 0.7,
    ) -> List[tuple]:
        """
        Get predictions with high confidence.

        Args:
            file_metrics: File metrics dictionary
            min_confidence: Minimum confidence threshold

        Returns:
            List of (file_path, prediction)
        """
        predictions = self.predict_multiple_files(file_metrics)
        return [
            (path, pred)
            for path, pred in predictions
            if pred.confidence >= min_confidence
        ]

    def _find_similar_failures(
        self,
        file_path: str,
        limit: int = 5,
    ) -> List[PastFailure]:
        """Find past failures in similar files."""
        # Simple similarity: same directory or similar complexity
        similar = []

        file_dir = file_path.rsplit("/", 1)[0] if "/" in file_path else ""

        for failure in self.past_failures:
            # Match if in same directory or same file
            if failure.file_path == file_path or failure.file_path.startswith(file_dir):
                similar.append(failure)

        return similar[:limit]

    def _get_applicable_patterns(
        self,
        file_path: str,
        code_context: Optional[CodeContext],
    ) -> list:
        """Get personal patterns applicable to this file."""
        patterns = []

        # If we know the code context, get patterns for it
        if code_context:
            patterns.extend(self.pattern_learner.get_patterns_by_context(code_context))

        # Also get high confidence patterns
        patterns.extend(self.pattern_learner.get_high_confidence_patterns())

        return patterns[:5]  # Top 5

    def _predict_failure_types(
        self,
        file_path: str,
        code_context: Optional[CodeContext],
        complexity: float,
        coverage: float,
    ) -> Dict[FailureType, float]:
        """Predict likelihood of different failure types."""
        failure_likelihoods: Dict[FailureType, float] = {t: 0.0 for t in FailureType}

        # If low coverage, more likely to have bugs
        if coverage < 0.3:
            failure_likelihoods[FailureType.BUG] += 0.4
            failure_likelihoods[FailureType.RELIABILITY_ISSUE] += 0.3

        # If high complexity, more likely to have logic issues
        if complexity > 15:
            failure_likelihoods[FailureType.BUG] += 0.3
            failure_likelihoods[FailureType.PERFORMANCE_ISSUE] += 0.2

        # Check personal patterns
        if code_context:
            issue_likelihoods = self.pattern_learner.predict_issue_likelihood(
                code_context
            )
            for issue_type, likelihood in issue_likelihoods.items():
                # Map issue types to failure types
                if issue_type == IssueType.SECURITY:
                    failure_likelihoods[FailureType.SECURITY_VULNERABILITY] += likelihood
                elif issue_type == IssueType.PERFORMANCE:
                    failure_likelihoods[FailureType.PERFORMANCE_ISSUE] += likelihood
                elif issue_type == IssueType.RELIABILITY:
                    failure_likelihoods[FailureType.RELIABILITY_ISSUE] += likelihood
                elif issue_type == IssueType.LOGIC:
                    failure_likelihoods[FailureType.BUG] += likelihood

        # Check past failures
        similar = self._find_similar_failures(file_path)
        for failure in similar:
            failure_likelihoods[failure.failure_type] += 0.2

        # Normalize to 0-1
        total = sum(failure_likelihoods.values())
        if total > 0:
            failure_likelihoods = {k: min(1.0, v / total) for k, v in failure_likelihoods.items()}

        return {k: round(v, 2) for k, v in failure_likelihoods.items()}

    def _calculate_likelihood(
        self,
        risk_score: float,
        similar_failure_count: int,
        patterns: list,
    ) -> float:
        """Calculate overall failure likelihood."""
        # 70% from risk score, 20% from similar failures, 10% from patterns
        likelihood = risk_score * 0.7

        # Similar failures boost likelihood
        likelihood += min(0.2, similar_failure_count * 0.04)

        # Patterns boost likelihood
        likelihood += min(0.1, len(patterns) * 0.02)

        return min(1.0, likelihood)

    def _calculate_confidence(
        self,
        file_path: str,
        similar_count: int,
        pattern_count: int,
    ) -> float:
        """Calculate confidence in prediction."""
        history = self.history_db.get_file_history(file_path)

        # More historical data = higher confidence
        history_confidence = min(0.5, len(history) / 10.0)

        # Similar failures = higher confidence
        similar_confidence = min(0.3, similar_count / 3.0)

        # Matching patterns = higher confidence
        pattern_confidence = min(0.2, pattern_count / 5.0)

        return min(1.0, history_confidence + similar_confidence + pattern_confidence)

    def _get_recommendations(
        self,
        file_path: str,
        risk_score,
        patterns: list,
        failure_types: Dict[FailureType, float],
    ) -> List[str]:
        """Get actionable recommendations to prevent failure."""
        recommendations = []

        # Risk-based recommendations
        if risk_score.coverage_score > 0.7:
            recommendations.append(f"Increase test coverage (currently {(1-risk_score.coverage_score)*100:.0f}%)")

        if risk_score.churn_score > 0.6:
            recommendations.append("This file changes frequently - prioritize testing and review")

        if risk_score.complexity_score > 0.6:
            recommendations.append("High complexity - consider breaking into smaller functions")

        # Pattern-based recommendations
        for pattern in patterns:
            if pattern.preventions:
                recommendations.append(f"Pattern: {pattern.preventions[0]}")

        # Failure type recommendations
        likely_failure = max(failure_types.items(), key=lambda x: x[1])[0]
        if likely_failure == FailureType.BUG:
            recommendations.append("High bug risk - add more assertions and tests")
        elif likely_failure == FailureType.SECURITY_VULNERABILITY:
            recommendations.append("Security risk detected - review input validation")
        elif likely_failure == FailureType.PERFORMANCE_ISSUE:
            recommendations.append("Performance risk - profile and optimize hot paths")

        return recommendations[:5]  # Top 5 recommendations
