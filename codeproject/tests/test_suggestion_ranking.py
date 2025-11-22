"""
Suggestion Ranking Engine Tests

Tests for ranking suggestions by multiple factors including confidence,
acceptance rate, impact, fix time, and team preferences.
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database import (
    Base,
    Review,
    Finding,
    SuggestionFeedback,
    LearningMetrics,
    PatternMetrics,
    FindingCategory,
    FindingSeverity,
)
from src.learning.suggestion_ranker import SuggestionRanker


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_db() -> Session:
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys = ON"))
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    yield db
    db.close()


def create_review(db: Session, pr_id: int = 1) -> Review:
    """Helper to create a test review."""
    review = Review(
        pr_id=pr_id,
        repo_url="https://github.com/test/repo",
        commit_sha="abc123def456",
        branch="main",
        created_at=datetime.now(timezone.utc),
    )
    db.add(review)
    db.commit()
    return review


def create_finding(
    db: Session,
    review_id: int,
    title: str = "SQL Injection",
    severity: FindingSeverity = FindingSeverity.CRITICAL,
    category: FindingCategory = FindingCategory.SECURITY,
) -> Finding:
    """Helper to create a test finding."""
    finding = Finding(
        review_id=review_id,
        category=category,
        severity=severity,
        title=title,
        description="Test finding",
        file_path="src/test.py",
        line_number=42,
        created_at=datetime.now(timezone.utc),
    )
    db.add(finding)
    db.commit()
    return finding


def create_feedback(
    db: Session,
    finding_id: int,
    feedback_type: str = "ACCEPTED",
    confidence: float = 0.85,
) -> SuggestionFeedback:
    """Helper to create feedback."""
    feedback = SuggestionFeedback(
        finding_id=finding_id,
        feedback_type=feedback_type,
        confidence=confidence,
        developer_comment="Test feedback",
        created_at=datetime.now(timezone.utc),
    )
    db.add(feedback)
    db.commit()
    return feedback


def create_learning_metrics(
    db: Session,
    category: FindingCategory = FindingCategory.SECURITY,
    severity: FindingSeverity = FindingSeverity.CRITICAL,
    acceptance_rate: float = 0.80,
    avg_time_to_fix: float = 2.5,
) -> LearningMetrics:
    """Helper to create learning metrics."""
    metrics = LearningMetrics(
        repo_url="https://github.com/test/repo",
        category=category,
        severity=severity,
        total_findings=10,
        confirmed_findings=8,
        false_positives=2,
        false_negatives=0,
        accuracy=acceptance_rate * 100.0,
        precision=0.8,
        recall=0.9,
        false_positive_rate=0.2,
        confidence_threshold=0.75,
        avg_time_to_fix=avg_time_to_fix,
    )
    db.add(metrics)
    db.commit()
    return metrics


def create_pattern_metrics(
    db: Session,
    pattern_type: str = "SQL Injection",
    prevalence: str = "common",
    is_anti_pattern: bool = False,
) -> PatternMetrics:
    """Helper to create pattern metrics."""
    import json
    pattern = PatternMetrics(
        pattern_hash="hash123",
        pattern_type=pattern_type,
        occurrences=15,
        files_affected=json.dumps({"src/auth.py": 5, "src/db.py": 10}),
        avg_severity=0.8,
        acceptance_rate=0.85,
        fix_count=12,
        anti_pattern=is_anti_pattern,
        team_prevalence=prevalence,
        recommended_fix="Use parameterized queries",
        created_at=datetime.now(timezone.utc),
    )
    db.add(pattern)
    db.commit()
    return pattern


# ============================================================================
# Confidence Score Tests
# ============================================================================


class TestConfidenceScore:
    """Test confidence score component."""

    def test_get_confidence_score_without_feedback(self, test_db: Session):
        """Return default 0.5 when finding has no feedback."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id)

        ranker = SuggestionRanker(test_db)
        score = ranker.get_confidence_score(finding)

        # No feedback → default 0.5
        assert score == pytest.approx(0.5, rel=0.01)

    def test_get_confidence_score_with_feedback(self, test_db: Session):
        """Return average confidence from feedbacks."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id)
        create_feedback(test_db, finding.id, confidence=0.90)
        create_feedback(test_db, finding.id, confidence=0.80)

        ranker = SuggestionRanker(test_db)
        score = ranker.get_confidence_score(finding)

        # Average of 0.90 and 0.80 = 0.85
        assert score == pytest.approx(0.85, rel=0.01)


# ============================================================================
# Acceptance Rate Score Tests
# ============================================================================


class TestAcceptanceRateScore:
    """Test acceptance rate scoring."""

    def test_acceptance_rate_with_history(self, test_db: Session):
        """Return acceptance rate from learning metrics."""
        review = create_review(test_db)
        finding = create_finding(
            test_db, review.id, title="SQL Injection", severity=FindingSeverity.CRITICAL
        )
        create_learning_metrics(
            test_db,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            acceptance_rate=0.75,
        )

        ranker = SuggestionRanker(test_db)
        score = ranker.get_acceptance_rate_score(finding)

        # Should return acceptance_rate from metrics (75%)
        assert score == pytest.approx(0.75, rel=0.01)

    def test_acceptance_rate_no_history(self, test_db: Session):
        """Return 0.5 default when no history exists."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id, title="Unknown Finding")

        ranker = SuggestionRanker(test_db)
        score = ranker.get_acceptance_rate_score(finding)

        # No metrics → default to 0.5
        assert score == pytest.approx(0.5, rel=0.01)


# ============================================================================
# Impact Score Tests
# ============================================================================


class TestImpactScore:
    """Test impact score calculation."""

    def test_impact_score_critical_common(self, test_db: Session):
        """Critical + common should give high impact score."""
        review = create_review(test_db)
        finding = create_finding(
            test_db, review.id, title="SQL Injection", severity=FindingSeverity.CRITICAL
        )
        create_pattern_metrics(test_db, pattern_type="SQL Injection", prevalence="common")

        ranker = SuggestionRanker(test_db)
        score = ranker.get_impact_score(finding)

        # Critical (1.0) + common (1.0) / 2 = 1.0
        assert score >= 0.8

    def test_impact_score_low_rare(self, test_db: Session):
        """Low + rare should give low impact score."""
        review = create_review(test_db)
        finding = create_finding(
            test_db, review.id, title="Minor Style Issue", severity=FindingSeverity.LOW
        )
        create_pattern_metrics(
            test_db, pattern_type="Minor Style Issue", prevalence="rare"
        )

        ranker = SuggestionRanker(test_db)
        score = ranker.get_impact_score(finding)

        # Low (0.25) + rare (0.3) / 2 = 0.275
        assert score < 0.5

    def test_impact_score_medium_occasional(self, test_db: Session):
        """Medium + occasional should give medium impact score."""
        review = create_review(test_db)
        finding = create_finding(
            test_db,
            review.id,
            title="Resource Leak",
            severity=FindingSeverity.MEDIUM,
        )
        create_pattern_metrics(
            test_db, pattern_type="Resource Leak", prevalence="occasional"
        )

        ranker = SuggestionRanker(test_db)
        score = ranker.get_impact_score(finding)

        # Medium (0.5) + occasional (0.6) / 2 = 0.55
        assert 0.4 < score < 0.7


# ============================================================================
# Fix Time Score Tests
# ============================================================================


class TestFixTimeScore:
    """Test fix time scoring."""

    def test_fix_time_score_quick_under_1_hour(self, test_db: Session):
        """Fix time < 1 hour should score 1.0."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id, title="Quick Fix")
        create_learning_metrics(
            test_db, avg_time_to_fix=0.5
        )  # 0.5 hours

        ranker = SuggestionRanker(test_db)
        score = ranker.get_fix_time_score(finding)

        # Time < 1 hour → 1.0
        assert score == pytest.approx(1.0, rel=0.01)

    def test_fix_time_score_1_to_4_hours(self, test_db: Session):
        """Fix time 1-4 hours should score 0.75."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id, title="Medium Fix")
        create_learning_metrics(test_db, avg_time_to_fix=2.5)  # 2.5 hours

        ranker = SuggestionRanker(test_db)
        score = ranker.get_fix_time_score(finding)

        # 1 < time < 4 → 0.75
        assert score == pytest.approx(0.75, rel=0.01)

    def test_fix_time_score_4_to_8_hours(self, test_db: Session):
        """Fix time 4-8 hours should score 0.50."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id, title="Slow Fix")
        create_learning_metrics(test_db, avg_time_to_fix=6.0)  # 6 hours

        ranker = SuggestionRanker(test_db)
        score = ranker.get_fix_time_score(finding)

        # 4 < time < 8 → 0.50
        assert score == pytest.approx(0.50, rel=0.01)

    def test_fix_time_score_over_8_hours(self, test_db: Session):
        """Fix time > 8 hours should score 0.25."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id, title="Very Slow Fix")
        create_learning_metrics(test_db, avg_time_to_fix=12.0)  # 12 hours

        ranker = SuggestionRanker(test_db)
        score = ranker.get_fix_time_score(finding)

        # time > 8 → 0.25
        assert score == pytest.approx(0.25, rel=0.01)

    def test_fix_time_score_no_history(self, test_db: Session):
        """No fix time history returns default 0.5."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id, title="No History")

        ranker = SuggestionRanker(test_db)
        score = ranker.get_fix_time_score(finding)

        # No metrics → 0.5
        assert score == pytest.approx(0.5, rel=0.01)


# ============================================================================
# Team Preference Score Tests
# ============================================================================


class TestTeamPreferenceScore:
    """Test team preference scoring."""

    def test_team_preference_normal_pattern(self, test_db: Session):
        """Normal pattern gets acceptance rate score."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id, title="SQL Injection")
        create_pattern_metrics(
            test_db,
            pattern_type="SQL Injection",
            prevalence="common",
            is_anti_pattern=False,
        )
        create_learning_metrics(test_db, acceptance_rate=0.80)

        ranker = SuggestionRanker(test_db)
        score = ranker.get_team_preference_score(finding)

        # Normal pattern → acceptance_rate
        assert 0.7 <= score <= 1.0

    def test_team_preference_anti_pattern_boost(self, test_db: Session):
        """Anti-pattern gets boost (acceptance_rate × 1.2)."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id, title="Bad Pattern")
        create_pattern_metrics(
            test_db,
            pattern_type="Bad Pattern",
            prevalence="common",
            is_anti_pattern=True,
        )
        create_learning_metrics(test_db, acceptance_rate=0.70)

        ranker = SuggestionRanker(test_db)
        score = ranker.get_team_preference_score(finding)

        # Anti-pattern with 0.70 acceptance → 0.70 × 1.2 = 0.84 (capped at 1.0)
        assert score >= 0.80


# ============================================================================
# Ranking Score Tests
# ============================================================================


class TestRankingScore:
    """Test composite ranking score calculation."""

    def test_calculate_ranking_score_all_high(self, test_db: Session):
        """High scores on all components should give high ranking."""
        review = create_review(test_db)
        finding = create_finding(
            test_db,
            review.id,
            title="SQL Injection",
            severity=FindingSeverity.CRITICAL,
        )
        create_learning_metrics(
            test_db,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            acceptance_rate=0.95,
            avg_time_to_fix=0.5,
        )
        create_pattern_metrics(
            test_db, pattern_type="SQL Injection", prevalence="common"
        )

        ranker = SuggestionRanker(test_db)
        score = ranker.calculate_ranking_score(finding)

        # All high scores → high total
        assert score > 0.8

    def test_calculate_ranking_score_all_low(self, test_db: Session):
        """Low scores on all components should give low ranking."""
        review = create_review(test_db)
        finding = create_finding(
            test_db,
            review.id,
            title="Trivial Issue",
            severity=FindingSeverity.LOW,
        )
        create_learning_metrics(
            test_db,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.LOW,
            acceptance_rate=0.2,
            avg_time_to_fix=12.0,
        )
        create_pattern_metrics(
            test_db, pattern_type="Trivial Issue", prevalence="rare"
        )

        ranker = SuggestionRanker(test_db)
        score = ranker.calculate_ranking_score(finding)

        # All low scores → low total
        assert score < 0.5


# ============================================================================
# Ranking List Tests
# ============================================================================


class TestRankingList:
    """Test ranking multiple findings."""

    def test_rank_findings_order(self, test_db: Session):
        """Findings should be ranked by composite score (highest first)."""
        review = create_review(test_db)

        # High-priority finding (will have higher impact)
        f_high = create_finding(
            test_db,
            review.id,
            title="Critical Vuln",
            severity=FindingSeverity.CRITICAL,
        )
        create_feedback(test_db, f_high.id, confidence=0.95)
        create_learning_metrics(
            test_db,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            acceptance_rate=0.9,
            avg_time_to_fix=0.5,
        )

        # Low-priority finding (will have lower impact)
        f_low = create_finding(
            test_db,
            review.id,
            title="Minor Style",
            severity=FindingSeverity.LOW,
        )
        create_feedback(test_db, f_low.id, confidence=0.3)
        create_learning_metrics(
            test_db,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.LOW,
            acceptance_rate=0.2,
            avg_time_to_fix=12.0,
        )

        ranker = SuggestionRanker(test_db)
        ranked = ranker.rank_findings([f_low, f_high])

        # Should be ranked: high score first, low score second
        assert ranked[0][0].id == f_high.id
        assert ranked[1][0].id == f_low.id
        assert ranked[0][1] > ranked[1][1]  # Score comparison

    def test_rank_findings_returns_scores(self, test_db: Session):
        """rank_findings should return (finding, score, component_scores)."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id)

        ranker = SuggestionRanker(test_db)
        ranked = ranker.rank_findings([finding])

        assert len(ranked) == 1
        finding_obj, total_score, component_scores = ranked[0]

        assert finding_obj.id == finding.id
        assert 0.0 <= total_score <= 1.0
        assert "confidence" in component_scores
        assert "acceptance_rate" in component_scores
        assert "impact_score" in component_scores
        assert "fix_time" in component_scores
        assert "team_preference" in component_scores


# ============================================================================
# Custom Weights Tests
# ============================================================================


class TestCustomWeights:
    """Test custom weight configuration."""

    def test_set_weights_valid(self, test_db: Session):
        """Setting valid weights should succeed."""
        ranker = SuggestionRanker(test_db)
        new_weights = {
            "confidence": 0.40,
            "acceptance_rate": 0.30,
            "impact_score": 0.20,
            "fix_time": 0.05,
            "team_preference": 0.05,
        }

        ranker.set_weights(new_weights)
        assert ranker.get_weights() == new_weights

    def test_set_weights_must_sum_to_1(self, test_db: Session):
        """Weights must sum to approximately 1.0."""
        ranker = SuggestionRanker(test_db)
        bad_weights = {
            "confidence": 0.50,
            "acceptance_rate": 0.30,
            "impact_score": 0.10,
            "fix_time": 0.05,
            "team_preference": 0.06,  # 0.06 makes total 1.01, which should fail
        }

        with pytest.raises(ValueError):
            ranker.set_weights(bad_weights)

    def test_custom_weights_affect_ranking(self, test_db: Session):
        """Using different weights should result in different scores."""
        review = create_review(test_db)

        # Finding with high confidence, low acceptance
        f1 = create_finding(
            test_db,
            review.id,
            title="High Conf",
            severity=FindingSeverity.CRITICAL,
        )
        create_feedback(test_db, f1.id, confidence=0.95)
        create_learning_metrics(
            test_db,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            acceptance_rate=0.2,
            avg_time_to_fix=1.0,
        )

        ranker = SuggestionRanker(test_db)

        # Get score with default weights (confidence-heavy)
        default_score = ranker.calculate_ranking_score(f1)

        # Change to acceptance-heavy weights
        ranker.set_weights(
            {
                "confidence": 0.05,
                "acceptance_rate": 0.65,
                "impact_score": 0.15,
                "fix_time": 0.10,
                "team_preference": 0.05,
            }
        )
        custom_score = ranker.calculate_ranking_score(f1)

        # Scores should differ significantly (confidence-heavy > acceptance-heavy for high conf/low acceptance)
        assert default_score != custom_score
        assert default_score > custom_score  # Confidence score (0.95) dominates default, acceptance (0.2) dominates custom


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_findings_list(self, test_db: Session):
        """Ranking empty list returns empty list."""
        ranker = SuggestionRanker(test_db)
        ranked = ranker.rank_findings([])

        assert ranked == []

    def test_single_finding(self, test_db: Session):
        """Ranking single finding returns it."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id)

        ranker = SuggestionRanker(test_db)
        ranked = ranker.rank_findings([finding])

        assert len(ranked) == 1
        assert ranked[0][0].id == finding.id

    def test_ranking_score_never_exceeds_1(self, test_db: Session):
        """Ranking score should never exceed 1.0."""
        review = create_review(test_db)
        finding = create_finding(test_db, review.id)

        ranker = SuggestionRanker(test_db)
        score = ranker.calculate_ranking_score(finding)

        assert score <= 1.0
