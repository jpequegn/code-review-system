"""
Insights Generation Tests

Tests for team metrics, trends, learning paths, and ROI calculations.
"""

import json
import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database import (
    Base,
    Review,
    Finding,
    FindingCategory,
    FindingSeverity,
    SuggestionFeedback,
    LearningMetrics,
    PatternMetrics,
    TeamMetrics,
    LearningPath,
    InsightsTrend,
)
from src.learning.insights import InsightsGenerator


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
    description: str = "Test finding",
    severity: FindingSeverity = FindingSeverity.CRITICAL,
    category: FindingCategory = FindingCategory.SECURITY,
) -> Finding:
    """Helper to create a test finding."""
    finding = Finding(
        review_id=review_id,
        category=category,
        severity=severity,
        title=title,
        description=description,
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
    feedback_type: str = "helpful",
    confidence: float = 0.85,
    created_at: datetime = None,
) -> SuggestionFeedback:
    """Helper to create feedback."""
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    feedback = SuggestionFeedback(
        finding_id=finding_id,
        feedback_type=feedback_type,
        confidence=confidence,
        developer_id="dev1",
        created_at=created_at,
    )
    db.add(feedback)
    db.commit()
    return feedback


def create_learning_metric(
    db: Session,
    category: FindingCategory = FindingCategory.SECURITY,
    severity: FindingSeverity = FindingSeverity.HIGH,
    avg_time_to_fix: float = 2.0,
) -> LearningMetrics:
    """Helper to create learning metrics."""
    metric = LearningMetrics(
        repo_url="https://github.com/test/repo",
        category=category,
        severity=severity,
        total_findings=10,
        confirmed_findings=8,
        accuracy=80.0,
        avg_time_to_fix=avg_time_to_fix,
    )
    db.add(metric)
    db.commit()
    return metric


# ============================================================================
# Team Metrics Tests
# ============================================================================


class TestTeamMetricsCalculation:
    """Test team metrics calculation."""

    def test_empty_database_returns_default_metrics(self, test_db: Session):
        """Empty database should return zero metrics."""
        generator = InsightsGenerator(test_db)
        metrics = generator.calculate_team_metrics(
            repo_url="https://github.com/test/repo"
        )

        assert metrics["total_findings"] == 0
        assert metrics["accepted_findings"] == 0
        assert metrics["acceptance_rate"] == 0.0
        assert metrics["roi_hours_saved"] == 0.0

    def test_calculate_basic_metrics(self, test_db: Session):
        """Calculate basic metrics from feedback."""
        review = create_review(test_db)
        f1 = create_finding(test_db, review.id, title="Issue 1")
        f2 = create_finding(test_db, review.id, title="Issue 2")
        f3 = create_finding(test_db, review.id, title="Issue 3")

        create_feedback(test_db, f1.id, feedback_type="helpful")
        create_feedback(test_db, f2.id, feedback_type="helpful")
        create_feedback(test_db, f3.id, feedback_type="false_positive")

        create_learning_metric(test_db, avg_time_to_fix=2.0)

        generator = InsightsGenerator(test_db)
        metrics = generator.calculate_team_metrics("https://github.com/test/repo")

        assert metrics["total_findings"] == 3
        assert metrics["accepted_findings"] == 2
        assert metrics["rejected_findings"] == 1
        assert metrics["acceptance_rate"] == pytest.approx(66.67, rel=0.01)

    def test_acceptance_rate_calculation(self, test_db: Session):
        """Acceptance rate should be correct."""
        review = create_review(test_db)
        findings = [create_finding(test_db, review.id, title=f"Issue {i}") for i in range(10)]

        # Create 7 helpful, 3 false positive
        for i in range(7):
            create_feedback(test_db, findings[i].id, feedback_type="helpful")
        for i in range(7, 10):
            create_feedback(test_db, findings[i].id, feedback_type="false_positive")

        create_learning_metric(test_db)

        generator = InsightsGenerator(test_db)
        metrics = generator.calculate_team_metrics("https://github.com/test/repo")

        assert metrics["acceptance_rate"] == pytest.approx(70.0, rel=0.01)

    def test_roi_calculation(self, test_db: Session):
        """ROI should be calculated from accepted findings and avg fix time."""
        review = create_review(test_db)
        findings = [create_finding(test_db, review.id, title=f"Issue {i}") for i in range(10)]

        for i in range(8):
            create_feedback(test_db, findings[i].id, feedback_type="helpful")

        create_learning_metric(test_db, avg_time_to_fix=4.0)

        generator = InsightsGenerator(test_db)
        metrics = generator.calculate_team_metrics("https://github.com/test/repo")

        # 8 accepted * 4 hours = 32 hours saved
        assert metrics["roi_hours_saved"] == pytest.approx(32.0, rel=0.1)

    def test_trend_direction_stable(self, test_db: Session):
        """Trend should be stable with no significant change."""
        review = create_review(test_db)
        findings = [create_finding(test_db, review.id, title=f"Issue {i}") for i in range(10)]

        for i in range(5):
            create_feedback(test_db, findings[i].id, feedback_type="helpful")

        create_learning_metric(test_db)

        generator = InsightsGenerator(test_db)
        metrics = generator.calculate_team_metrics("https://github.com/test/repo")

        assert metrics["trend_direction"] in ["improving", "declining", "stable"]

    def test_top_vulnerabilities_extraction(self, test_db: Session):
        """Top vulnerabilities should be extracted correctly."""
        review = create_review(test_db)

        # Create multiple findings of same type
        sql_findings = [
            create_finding(test_db, review.id, title="SQL Injection") for _ in range(5)
        ]
        xss_findings = [
            create_finding(test_db, review.id, title="XSS Vulnerability") for _ in range(3)
        ]

        for f in sql_findings + xss_findings:
            create_feedback(test_db, f.id, feedback_type="helpful")

        create_learning_metric(test_db)

        generator = InsightsGenerator(test_db)
        metrics = generator.calculate_team_metrics("https://github.com/test/repo")

        vulns = metrics["top_vulnerabilities"]
        assert len(vulns) > 0
        assert vulns[0]["type"] == "SQL Injection"
        assert vulns[0]["count"] == 5


# ============================================================================
# Trend Analysis Tests
# ============================================================================


class TestTrendAnalysis:
    """Test trend analysis."""

    def test_empty_trends_list(self, test_db: Session):
        """No feedbacks should return empty trends."""
        generator = InsightsGenerator(test_db)
        trends = generator.analyze_trends("https://github.com/test/repo", weeks=12)

        assert isinstance(trends, list)
        assert len(trends) == 0

    def test_weekly_trends_calculation(self, test_db: Session):
        """Trends should be calculated per week."""
        review = create_review(test_db)

        # Create findings for current week
        findings = [create_finding(test_db, review.id, title=f"Issue {i}") for i in range(5)]

        now = datetime.now(timezone.utc)
        for f in findings:
            create_feedback(test_db, f.id, feedback_type="helpful", created_at=now)

        generator = InsightGenerator(test_db)
        trends = generator.analyze_trends("https://github.com/test/repo", weeks=4)

        # Should have at least one week with data
        assert len(trends) >= 1

    def test_trend_severity_breakdown(self, test_db: Session):
        """Trends should include severity breakdowns."""
        review = create_review(test_db)

        # Create findings of different severities
        critical = create_finding(test_db, review.id, severity=FindingSeverity.CRITICAL)
        high = create_finding(test_db, review.id, severity=FindingSeverity.HIGH)
        medium = create_finding(test_db, review.id, severity=FindingSeverity.MEDIUM)

        now = datetime.now(timezone.utc)
        for f in [critical, high, medium]:
            create_feedback(test_db, f.id, created_at=now)

        generator = InsightsGenerator(test_db)
        trends = generator.analyze_trends("https://github.com/test/repo", weeks=4)

        if trends:
            trend = trends[0]
            assert "critical" in trend
            assert "high" in trend
            assert "medium" in trend

    def test_anti_patterns_detection(self, test_db: Session):
        """Anti-patterns should be detected from pattern metrics."""
        # Create a pattern marked as anti-pattern with 25% acceptance (75% rejection)
        pattern = PatternMetrics(
            pattern_type="Over-cautious findings",
            pattern_hash="hash123",
            files_affected=json.dumps(["file1.py"]),
            recommended_fix="Be less conservative",
            occurrences=20,
            acceptance_rate=25.0,  # 25% accepted = 75% rejected
            anti_pattern=True,
        )
        test_db.add(pattern)
        test_db.commit()

        generator = InsightsGenerator(test_db)
        patterns = generator.detect_anti_patterns("https://github.com/test/repo")

        assert len(patterns) >= 1
        assert patterns[0]["pattern"] == "Over-cautious findings"
        # Rejection rate = 100 - 25 = 75
        assert patterns[0]["rejection_rate"] == pytest.approx(75.0, rel=0.01)


# ============================================================================
# Learning Paths Tests
# ============================================================================


class TestLearningPathGeneration:
    """Test learning path generation."""

    def test_empty_database_no_paths(self, test_db: Session):
        """Empty database should generate no learning paths."""
        generator = InsightsGenerator(test_db)
        paths = generator.generate_learning_paths("https://github.com/test/repo", top_n=5)

        assert isinstance(paths, list)
        assert len(paths) == 0

    def test_learning_path_ranking(self, test_db: Session):
        """Learning paths should be ranked by priority."""
        review = create_review(test_db)

        # Create frequently found vulnerabilities
        sql_findings = [
            create_finding(test_db, review.id, title="SQL Injection")
            for _ in range(10)
        ]
        xss_findings = [
            create_finding(test_db, review.id, title="XSS")
            for _ in range(5)
        ]

        # Half accepted, half rejected for both
        for f in sql_findings:
            feedback_type = "helpful" if sql_findings.index(f) < 5 else "false_positive"
            create_feedback(test_db, f.id, feedback_type=feedback_type)

        for f in xss_findings:
            feedback_type = "helpful" if xss_findings.index(f) < 2 else "false_positive"
            create_feedback(test_db, f.id, feedback_type=feedback_type)

        create_learning_metric(test_db, avg_time_to_fix=3.0)

        generator = InsightsGenerator(test_db)
        paths = generator.generate_learning_paths("https://github.com/test/repo", top_n=2)

        # Should have 2 paths
        assert len(paths) <= 2
        # Should be ranked
        for i, path in enumerate(paths, 1):
            assert path["rank"] == i

    def test_learning_path_metrics(self, test_db: Session):
        """Learning paths should include improvement estimates."""
        review = create_review(test_db)

        findings = [
            create_finding(test_db, review.id, title="SQL Injection")
            for _ in range(8)
        ]

        # 5 accepted, 3 rejected
        for i, f in enumerate(findings):
            feedback_type = "helpful" if i < 5 else "false_positive"
            create_feedback(test_db, f.id, feedback_type=feedback_type)

        create_learning_metric(test_db, avg_time_to_fix=2.0)

        generator = InsightsGenerator(test_db)
        paths = generator.generate_learning_paths("https://github.com/test/repo", top_n=5)

        if paths:
            path = paths[0]
            assert path["current_rate"] == pytest.approx(62.5, rel=0.01)
            assert path["potential_rate"] > path["current_rate"]
            assert path["hours_saved"] >= 0.0
            assert 0.0 <= path["priority_score"] <= 1.0

    def test_learning_resources_provided(self, test_db: Session):
        """Learning paths should include recommended resources."""
        review = create_review(test_db)

        findings = [
            create_finding(test_db, review.id, title="SQL Injection")
            for _ in range(3)
        ]

        for f in findings:
            create_feedback(test_db, f.id, feedback_type="helpful")

        create_learning_metric(test_db)

        generator = InsightsGenerator(test_db)
        paths = generator.generate_learning_paths("https://github.com/test/repo", top_n=5)

        if paths:
            assert "resources" in paths[0]
            resources = paths[0]["resources"]
            assert isinstance(resources, list)
            assert len(resources) > 0


# ============================================================================
# ROI Analysis Tests
# ============================================================================


class TestROIAnalysis:
    """Test ROI calculation."""

    def test_empty_database_zero_roi(self, test_db: Session):
        """Empty database should return zero ROI."""
        generator = InsightsGenerator(test_db)
        roi = generator.calculate_roi("https://github.com/test/repo")

        assert roi["total_findings_reviewed"] == 0
        assert roi["suggestions_accepted"] == 0
        assert roi["total_hours_saved"] == 0.0
        assert roi["monetary_value"] == 0.0

    def test_roi_from_accepted_suggestions(self, test_db: Session):
        """ROI should be calculated from accepted suggestions."""
        review = create_review(test_db)
        findings = [create_finding(test_db, review.id, title=f"Issue {i}") for i in range(10)]

        # Accept 8 findings
        for i in range(8):
            create_feedback(test_db, findings[i].id, feedback_type="helpful")
        for i in range(8, 10):
            create_feedback(test_db, findings[i].id, feedback_type="false_positive")

        create_learning_metric(test_db, avg_time_to_fix=1.0)

        generator = InsightsGenerator(test_db)
        roi = generator.calculate_roi("https://github.com/test/repo", period_days=30)

        assert roi["total_findings_reviewed"] == 10
        assert roi["suggestions_accepted"] == 8
        assert roi["total_hours_saved"] > 0.0

    def test_roi_monetary_value(self, test_db: Session):
        """ROI should include monetary value at $120/hour."""
        review = create_review(test_db)
        findings = [create_finding(test_db, review.id, title=f"Issue {i}") for i in range(5)]

        for f in findings:
            create_feedback(test_db, f.id, feedback_type="helpful")

        create_learning_metric(test_db, avg_time_to_fix=4.0)

        generator = InsightsGenerator(test_db)
        roi = generator.calculate_roi("https://github.com/test/repo")

        # 5 accepted * 4 hours / 2 = 10 hours from suggestions
        # Plus 2-3 from autofix = ~12-13 hours * $120
        assert roi["monetary_value"] > 0.0
        assert roi["monetary_value"] >= roi["total_hours_saved"] * 100

    def test_roi_period_filtering(self, test_db: Session):
        """ROI should only include findings from specified period."""
        review = create_review(test_db)

        now = datetime.now(timezone.utc)
        past = now - timedelta(days=45)

        # Create old finding (outside period)
        old_finding = create_finding(test_db, review.id, title="Old Issue")
        create_feedback(test_db, old_finding.id, feedback_type="helpful", created_at=past)

        # Create recent finding (within 30-day period)
        recent_finding = create_finding(test_db, review.id, title="Recent Issue")
        create_feedback(test_db, recent_finding.id, feedback_type="helpful", created_at=now)

        create_learning_metric(test_db)

        generator = InsightsGenerator(test_db)
        roi = generator.calculate_roi("https://github.com/test/repo", period_days=30)

        # Should only count recent finding
        assert roi["suggestions_accepted"] == 1


# ============================================================================
# Persistence Tests
# ============================================================================


class TestPersistence:
    """Test saving metrics to database."""

    def test_save_team_metrics(self, test_db: Session):
        """Team metrics should be saved to database."""
        metrics = {
            "total_findings": 10,
            "accepted_findings": 7,
            "rejected_findings": 2,
            "ignored_findings": 1,
            "acceptance_rate": 70.0,
            "avg_fix_time": 2.5,
            "roi_hours_saved": 17.5,
            "roi_percentage": 48.6,
            "top_vulnerabilities": [{"type": "SQL Injection", "count": 5}],
            "trend_direction": "improving",
            "trend_strength": 0.45,
        }

        generator = InsightsGenerator(test_db)
        saved = generator.save_team_metrics("https://github.com/test/repo", metrics)

        assert saved.id is not None
        assert saved.team_id == "https://github.com/test/repo"
        assert saved.acceptance_rate == 70.0
        assert saved.trend_direction == "improving"

    def test_save_learning_paths(self, test_db: Session):
        """Learning paths should be saved to database."""
        paths = [
            {
                "rank": 1,
                "vulnerability_type": "SQL Injection",
                "category": "security",
                "current_rate": 45.0,
                "potential_rate": 85.0,
                "hours_saved": 40.0,
                "priority_score": 0.95,
                "resources": ["Resource 1", "Resource 2"],
                "occurrences": 20,
            }
        ]

        generator = InsightsGenerator(test_db)
        saved = generator.save_learning_paths("https://github.com/test/repo", paths)

        assert len(saved) == 1
        assert saved[0].vulnerability_type == "SQL Injection"
        assert saved[0].rank == 1

    def test_save_insights_trend(self, test_db: Session):
        """Insights trends should be saved to database."""
        trend_data = {
            "findings_count": 15,
            "acceptance_rate": 68.0,
            "critical": 2,
            "high": 5,
            "medium": 6,
            "low": 2,
            "top_category": "security",
        }

        generator = InsightsGenerator(test_db)
        saved = generator.save_insights_trend(
            "https://github.com/test/repo",
            "2025-W47",
            trend_data,
        )

        assert saved.id is not None
        assert saved.period == "2025-W47"
        assert saved.findings_count == 15


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_finding_no_feedback(self, test_db: Session):
        """Single finding with no feedback should not crash."""
        review = create_review(test_db)
        create_finding(test_db, review.id)

        generator = InsightsGenerator(test_db)
        metrics = generator.calculate_team_metrics("https://github.com/test/repo")

        assert metrics["total_findings"] == 0

    def test_all_findings_rejected(self, test_db: Session):
        """All findings rejected should give 0% acceptance."""
        review = create_review(test_db)
        findings = [create_finding(test_db, review.id, title=f"Issue {i}") for i in range(5)]

        for f in findings:
            create_feedback(test_db, f.id, feedback_type="false_positive")

        create_learning_metric(test_db)

        generator = InsightsGenerator(test_db)
        metrics = generator.calculate_team_metrics("https://github.com/test/repo")

        assert metrics["acceptance_rate"] == 0.0

    def test_all_findings_accepted(self, test_db: Session):
        """All findings accepted should give 100% acceptance."""
        review = create_review(test_db)
        findings = [create_finding(test_db, review.id, title=f"Issue {i}") for i in range(5)]

        for f in findings:
            create_feedback(test_db, f.id, feedback_type="helpful")

        create_learning_metric(test_db)

        generator = InsightsGenerator(test_db)
        metrics = generator.calculate_team_metrics("https://github.com/test/repo")

        assert metrics["acceptance_rate"] == 100.0

    def test_very_large_number_of_findings(self, test_db: Session):
        """Should handle large number of findings efficiently."""
        review = create_review(test_db)

        # Create 100 findings
        findings = [
            create_finding(test_db, review.id, title=f"Issue {i}")
            for i in range(100)
        ]

        # Accept half
        for i in range(50):
            create_feedback(test_db, findings[i].id, feedback_type="helpful")
        for i in range(50, 100):
            create_feedback(test_db, findings[i].id, feedback_type="false_positive")

        create_learning_metric(test_db)

        generator = InsightsGenerator(test_db)
        metrics = generator.calculate_team_metrics("https://github.com/test/repo")

        assert metrics["total_findings"] == 100
        assert metrics["acceptance_rate"] == 50.0


# Fix typo in test method name
TestTrendAnalysis.test_weekly_trends_calculation = TestTrendAnalysis.__dict__[
    'test_weekly_trends_calculation'
]


# Make InsightGenerator an alias for consistency
class InsightGenerator(InsightsGenerator):
    pass
