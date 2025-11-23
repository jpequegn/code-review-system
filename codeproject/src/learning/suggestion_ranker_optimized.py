"""
Optimized Suggestion Ranking Engine - Batch queries to eliminate N+1 patterns

Combines multiple scoring components to rank findings by:
- Confidence (calibrated from LLM)
- Acceptance rate (historical feedback)
- Impact score (severity × prevalence)
- Fix time (how quickly developers can fix)
- Team preferences (learned patterns)
- Diversity (avoid redundant suggestions)

Key optimization: Batch load all related data upfront instead of per-finding queries.
"""

from typing import Optional, List, Dict
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database import (
    Finding,
    SuggestionFeedback,
    LearningMetrics,
    PatternMetrics,
)
from src.learning.confidence_tuner import ConfidenceTuner
from src.learning.deduplication import DeduplicationService


class SuggestionRankerOptimized:
    """
    Optimized ranking engine with batch queries to eliminate N+1 patterns.

    Preloads all required data in bulk queries before processing findings,
    reducing database round-trips from 7 per finding to ~4 total.
    """

    def __init__(self, db: Session):
        """
        Initialize ranker with database session.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        # Default weights: balanced approach
        self.weights = {
            "confidence": 0.30,  # Tuning-adjusted confidence
            "acceptance_rate": 0.25,  # How often accepted
            "impact_score": 0.25,  # Severity × prevalence
            "fix_time": 0.10,  # How quick to fix
            "team_preference": 0.10,  # Team-learned patterns
        }

    # ============================================================================
    # Batch Data Loading
    # ============================================================================

    def _preload_feedback_by_finding(self, finding_ids: List[int]) -> Dict[int, List[SuggestionFeedback]]:
        """
        Batch load all feedback for findings.

        Single query instead of N queries (one per finding).

        Args:
            finding_ids: List of finding IDs

        Returns:
            Dict mapping finding_id → list of feedbacks
        """
        if not finding_ids:
            return {}

        feedbacks = (
            self.db.query(SuggestionFeedback)
            .filter(SuggestionFeedback.finding_id.in_(finding_ids))
            .all()
        )

        feedback_by_finding = {}
        for feedback in feedbacks:
            if feedback.finding_id not in feedback_by_finding:
                feedback_by_finding[feedback.finding_id] = []
            feedback_by_finding[feedback.finding_id].append(feedback)

        return feedback_by_finding

    def _preload_learning_metrics(self) -> Dict[tuple, LearningMetrics]:
        """
        Batch load all learning metrics.

        Single query instead of N queries (one per finding).

        Args:
            None

        Returns:
            Dict mapping (category, severity) → LearningMetrics
        """
        metrics = self.db.query(LearningMetrics).all()
        metrics_by_key = {}
        for m in metrics:
            key = (m.category, m.severity)
            metrics_by_key[key] = m
        return metrics_by_key

    def _preload_pattern_metrics(self) -> Dict[str, PatternMetrics]:
        """
        Batch load all pattern metrics.

        Single query instead of N queries (one per finding).

        Args:
            None

        Returns:
            Dict mapping pattern_type → PatternMetrics
        """
        patterns = self.db.query(PatternMetrics).all()
        patterns_by_type = {}
        for p in patterns:
            patterns_by_type[p.pattern_type] = p
        return patterns_by_type

    # ============================================================================
    # Score Calculation with Preloaded Data
    # ============================================================================

    def get_confidence_score(
        self,
        finding: Finding,
        feedback_by_finding: Dict[int, List[SuggestionFeedback]],
        tuner: Optional[ConfidenceTuner] = None,
    ) -> float:
        """
        Get calibrated confidence score from preloaded feedback.

        Args:
            finding: Finding to score
            feedback_by_finding: Preloaded feedback dict
            tuner: Optional ConfidenceTuner for calibration

        Returns:
            Confidence score (0.0-1.0)
        """
        feedbacks = feedback_by_finding.get(finding.id, [])

        if feedbacks:
            confidences = [f.confidence for f in feedbacks if f.confidence is not None]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if tuner:
                    return tuner.apply_calibration_to_finding(avg_confidence, "balanced")
                return avg_confidence

        return 0.5

    def get_acceptance_rate_score(
        self,
        finding: Finding,
        metrics_by_key: Dict[tuple, LearningMetrics],
    ) -> float:
        """
        Get acceptance rate from preloaded metrics.

        Args:
            finding: Finding to score
            metrics_by_key: Preloaded metrics dict

        Returns:
            Acceptance rate (0.0-1.0)
        """
        key = (finding.category, finding.severity)
        metrics = metrics_by_key.get(key)

        if not metrics:
            return 0.5

        return metrics.accuracy / 100.0 if metrics.accuracy else 0.5

    def get_impact_score(
        self,
        finding: Finding,
        patterns_by_type: Dict[str, PatternMetrics],
    ) -> float:
        """
        Calculate impact score from preloaded patterns.

        Args:
            finding: Finding to score
            patterns_by_type: Preloaded patterns dict

        Returns:
            Impact score (0.0-1.0)
        """
        # Severity scoring
        severity_weight = {
            "critical": 1.0,
            "high": 0.75,
            "medium": 0.50,
            "low": 0.25,
        }
        severity_score = severity_weight.get(finding.severity.value, 0.5)

        # Prevalence scoring from preloaded patterns
        pattern = patterns_by_type.get(finding.title)

        if pattern:
            prevalence_weight = {
                "rare": 0.3,
                "occasional": 0.6,
                "common": 1.0,
            }
            prevalence_score = prevalence_weight.get(
                pattern.team_prevalence.lower(), 0.5
            )
        else:
            prevalence_score = 0.5

        return (severity_score + prevalence_score) / 2

    def get_fix_time_score(
        self,
        finding: Finding,
        metrics_by_key: Dict[tuple, LearningMetrics],
    ) -> float:
        """
        Score based on preloaded metrics.

        Args:
            finding: Finding to score
            metrics_by_key: Preloaded metrics dict

        Returns:
            Time score (0.0-1.0)
        """
        key = (finding.category, finding.severity)
        metrics = metrics_by_key.get(key)

        if not metrics or not metrics.avg_time_to_fix:
            return 0.5

        hours = metrics.avg_time_to_fix
        if hours < 1:
            return 1.0
        elif hours < 4:
            return 0.75
        elif hours < 8:
            return 0.50
        else:
            return 0.25

    def get_team_preference_score(
        self,
        finding: Finding,
        metrics_by_key: Dict[tuple, LearningMetrics],
        patterns_by_type: Dict[str, PatternMetrics],
    ) -> float:
        """
        Score based on preloaded patterns.

        Args:
            finding: Finding to score
            metrics_by_key: Preloaded metrics dict
            patterns_by_type: Preloaded patterns dict

        Returns:
            Preference score (0.0-1.0)
        """
        key = (finding.category, finding.severity)
        metrics = metrics_by_key.get(key)
        acceptance_rate = (metrics.accuracy / 100.0 if metrics and metrics.accuracy else 0.5)

        # Check if this is an anti-pattern
        pattern = patterns_by_type.get(finding.title)

        if pattern and pattern.anti_pattern:
            # Team actively avoiding this pattern → boost score
            return min(acceptance_rate * 1.2, 1.0)
        else:
            return acceptance_rate

    # ============================================================================
    # Optimized Ranking
    # ============================================================================

    def rank_findings(
        self,
        findings: list[Finding],
        tuner: Optional[ConfidenceTuner] = None,
        shown_ids: Optional[List[int]] = None,
    ) -> list[tuple[Finding, float, dict]]:
        """
        Rank findings using preloaded batch data (optimized N+1 elimination).

        Preloads all required data once, then scores findings locally.
        Reduces queries from 7 per finding to ~4 total.

        Args:
            findings: List of findings to rank
            tuner: Optional ConfidenceTuner for calibration
            shown_ids: Optional list of finding IDs already shown (for diversity)

        Returns:
            List of (finding, total_score, component_scores) tuples
            sorted by total_score DESC
        """
        if not findings:
            return []

        finding_ids = [f.id for f in findings]

        # Preload all data in bulk queries
        feedback_by_finding = self._preload_feedback_by_finding(finding_ids)
        metrics_by_key = self._preload_learning_metrics()
        patterns_by_type = self._preload_pattern_metrics()

        dedup_service = DeduplicationService(self.db)
        ranked = []

        for finding in findings:
            scores = {
                "confidence": self.get_confidence_score(finding, feedback_by_finding, tuner),
                "acceptance_rate": self.get_acceptance_rate_score(finding, metrics_by_key),
                "impact_score": self.get_impact_score(finding, patterns_by_type),
                "fix_time": self.get_fix_time_score(finding, metrics_by_key),
                "team_preference": self.get_team_preference_score(
                    finding, metrics_by_key, patterns_by_type
                ),
            }

            total_score = sum(scores[key] * self.weights[key] for key in scores)
            total_score = min(total_score, 1.0)

            # Apply diversity factor if shown_ids provided
            if shown_ids:
                diversity_factor = dedup_service.calculate_diversity_factor(
                    finding.id, shown_ids
                )
                total_score = total_score * diversity_factor
                scores["diversity_factor"] = diversity_factor

            ranked.append((finding, total_score, scores))

        # Sort by score DESC
        return sorted(ranked, key=lambda x: x[1], reverse=True)

    # ============================================================================
    # Custom Weighting
    # ============================================================================

    def set_weights(self, new_weights: dict[str, float]) -> None:
        """
        Set custom ranking weights.

        Args:
            new_weights: Dict of component → weight mappings

        Raises:
            ValueError: If weights don't sum to ~1.0
        """
        weight_sum = sum(new_weights.values())
        if abs(weight_sum - 1.0) > 0.01:
            raise ValueError(
                f"Weights must sum to ~1.0 (got {weight_sum}). "
                f"Components: {new_weights}"
            )
        self.weights = new_weights.copy()

    def get_weights(self) -> dict[str, float]:
        """
        Get current ranking weights.

        Returns:
            Dict of component → weight mappings
        """
        return self.weights.copy()
