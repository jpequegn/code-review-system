"""
Suggestion Ranking Engine - Rank suggestions by multiple factors

Combines multiple scoring components to rank findings by:
- Confidence (calibrated from LLM)
- Acceptance rate (historical feedback)
- Impact score (severity × prevalence)
- Fix time (how quickly developers can fix)
- Team preferences (learned patterns)
- Diversity (avoid redundant suggestions)
"""

from typing import Optional, List
from sqlalchemy.orm import Session

from src.database import (
    Finding,
    SuggestionFeedback,
    LearningMetrics,
    PatternMetrics,
)
from src.learning.confidence_tuner import ConfidenceTuner
from src.learning.deduplication import DeduplicationService


class SuggestionRanker:
    """
    Rank findings by multiple factors to prioritize review focus.

    Problem: Not all suggestions are equally valuable. Some are quick to fix,
    others have high acceptance rates, some are critical security issues.

    Solution: Combine multiple factors with configurable weights to compute
    a composite ranking score for each finding.

    Example:
        ranker = SuggestionRanker(db_session)

        # Rank all findings from a review
        findings = db.query(Finding).all()
        ranked = ranker.rank_findings(findings)

        for finding, score, components in ranked:
            print(f"{finding.title}: {score:.2f} (confidence: {components['confidence']:.2f})")

        # Emphasize security over speed
        ranker.set_weights({
            'confidence': 0.40,
            'acceptance_rate': 0.30,
            'impact_score': 0.20,
            'fix_time': 0.05,
            'team_preference': 0.05
        })
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
    # Individual Score Components
    # ============================================================================

    def get_confidence_score(
        self, finding: Finding, tuner: Optional[ConfidenceTuner] = None
    ) -> float:
        """
        Get calibrated confidence score for a finding.

        Note: Confidence is stored in SuggestionFeedback, not Finding.
        This method returns a default score until feedback is provided.

        Args:
            finding: Finding to score
            tuner: Optional ConfidenceTuner for calibration

        Returns:
            Confidence score (0.0-1.0)
        """
        # Query feedbacks from database for this finding
        feedbacks = (
            self.db.query(SuggestionFeedback)
            .filter(SuggestionFeedback.finding_id == finding.id)
            .all()
        )

        if feedbacks:
            confidences = [f.confidence for f in feedbacks if f.confidence is not None]
            if confidences:
                avg_confidence = sum(confidences) / len(confidences)
                if tuner:
                    return tuner.apply_calibration_to_finding(avg_confidence, "balanced")
                return avg_confidence

        # Default score when no feedback available
        return 0.5

    def get_acceptance_rate_score(self, finding: Finding) -> float:
        """
        Get acceptance rate based on finding's category and severity.

        Looks up historical acceptance rate from LearningMetrics.
        Returns default 0.5 if no history exists.

        Args:
            finding: Finding to score

        Returns:
            Acceptance rate (0.0-1.0)
        """
        metrics = (
            self.db.query(LearningMetrics)
            .filter(
                LearningMetrics.category == finding.category,
                LearningMetrics.severity == finding.severity,
            )
            .first()
        )

        if not metrics:
            return 0.5

        # Return accuracy as proxy for acceptance rate
        return metrics.accuracy / 100.0 if metrics.accuracy else 0.5

    def get_impact_score(self, finding: Finding) -> float:
        """
        Calculate impact score based on severity and prevalence.

        Impact = (severity_score + prevalence_score) / 2

        Severity: CRITICAL=1.0, HIGH=0.75, MEDIUM=0.50, LOW=0.25
        Prevalence: common=1.0, occasional=0.6, rare=0.3

        Args:
            finding: Finding to score

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

        # Prevalence scoring from PatternMetrics
        pattern = (
            self.db.query(PatternMetrics)
            .filter(PatternMetrics.pattern_type == finding.title)
            .first()
        )

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

    def get_fix_time_score(self, finding: Finding) -> float:
        """
        Score based on how long findings of this type take to fix.

        Time-to-fix scores:
        - < 1 hour: 1.0
        - 1-4 hours: 0.75
        - 4-8 hours: 0.50
        - > 8 hours: 0.25

        Args:
            finding: Finding to score

        Returns:
            Time score (0.0-1.0)
        """
        metrics = (
            self.db.query(LearningMetrics)
            .filter(LearningMetrics.category == finding.category)
            .first()
        )

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
        self, finding: Finding, team_id: Optional[str] = None
    ) -> float:
        """
        Score based on team's historical preferences and patterns.

        - Normal pattern: use acceptance rate
        - Anti-pattern: boost acceptance rate × 1.2 (team is avoiding this!)
        - Unknown: default to acceptance rate

        Args:
            finding: Finding to score
            team_id: Optional team identifier (for future multi-team support)

        Returns:
            Preference score (0.0-1.0)
        """
        acceptance_rate = self.get_acceptance_rate_score(finding)

        # Check if this is an anti-pattern
        pattern = (
            self.db.query(PatternMetrics)
            .filter(
                PatternMetrics.pattern_type == finding.title,
                PatternMetrics.anti_pattern == True,
            )
            .first()
        )

        if pattern:
            # Team actively avoiding this pattern → boost score
            return min(acceptance_rate * 1.2, 1.0)
        else:
            return acceptance_rate

    # ============================================================================
    # Composite Ranking
    # ============================================================================

    def calculate_ranking_score(
        self, finding: Finding, tuner: Optional[ConfidenceTuner] = None
    ) -> float:
        """
        Calculate composite ranking score for a finding.

        Combines all components weighted by self.weights:
        score = Σ(component × weight)

        Args:
            finding: Finding to score
            tuner: Optional ConfidenceTuner for calibration

        Returns:
            Composite score (0.0-1.0)
        """
        scores = {
            "confidence": self.get_confidence_score(finding, tuner),
            "acceptance_rate": self.get_acceptance_rate_score(finding),
            "impact_score": self.get_impact_score(finding),
            "fix_time": self.get_fix_time_score(finding),
            "team_preference": self.get_team_preference_score(finding),
        }

        total_score = sum(scores[key] * self.weights[key] for key in scores)

        # Cap at 1.0
        return min(total_score, 1.0)

    def rank_findings(
        self,
        findings: list[Finding],
        tuner: Optional[ConfidenceTuner] = None,
        shown_ids: Optional[List[int]] = None,
    ) -> list[tuple[Finding, float, dict]]:
        """
        Rank findings by composite score, applying diversity factor.

        Returns findings sorted by score (highest first), with component
        scores for transparency and debugging. Applies diversity factor
        to reduce score of similar suggestions already shown.

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

        dedup_service = DeduplicationService(self.db)
        ranked = []

        for finding in findings:
            scores = {
                "confidence": self.get_confidence_score(finding, tuner),
                "acceptance_rate": self.get_acceptance_rate_score(finding),
                "impact_score": self.get_impact_score(finding),
                "fix_time": self.get_fix_time_score(finding),
                "team_preference": self.get_team_preference_score(finding),
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

        Weights must sum to approximately 1.0 and include all components:
        'confidence', 'acceptance_rate', 'impact_score', 'fix_time', 'team_preference'

        Example: Emphasize security impact over speed
        ranker.set_weights({
            'confidence': 0.40,
            'acceptance_rate': 0.30,
            'impact_score': 0.20,
            'fix_time': 0.05,
            'team_preference': 0.05
        })

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
