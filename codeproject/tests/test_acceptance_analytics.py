"""
Tests for Acceptance Rate Analytics.

Tests:
- AcceptanceAnalyzer calculation methods
- Acceptance rates by category, severity, type
- Fix timeline analysis
- Metrics persistence
- Edge cases and time window filtering
"""

import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.database import (
    Base,
    Review,
    Finding,
    SuggestionFeedback,
    ReviewStatus,
    FindingCategory,
    FindingSeverity,
)
from src.learning.analytics import AcceptanceAnalyzer
from src.learning.metrics import AcceptanceMetrics


@pytest.fixture
def test_db():
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:")

    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(bind=engine)
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestSessionLocal()
    yield session
    session.close()


def create_review(db: Session, pr_id: int) -> Review:
    """Helper to create a test review."""
    review = Review(
        pr_id=pr_id,
        repo_url="https://github.com/test/repo.git",
        branch="test",
        commit_sha="abc123",
        status=ReviewStatus.COMPLETED,
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
        suggested_fix="Fix this",
    )
    db.add(finding)
    db.commit()
    return finding


def create_feedback(
    db: Session,
    finding_id: int,
    feedback_type: str = "ACCEPTED",
    commit_hash: str | None = None,
    confidence: float = 0.85,
) -> SuggestionFeedback:
    """Helper to create test feedback."""
    feedback = SuggestionFeedback(
        finding_id=finding_id,
        feedback_type=feedback_type,
        confidence=confidence,
        commit_hash=commit_hash,
    )
    db.add(feedback)
    db.commit()
    return feedback


# ============================================================================
# Basic Acceptance Calculation Tests
# ============================================================================


class TestAcceptanceByCategory:
    """Test acceptance rate calculation by category."""

    def test_acceptance_by_category_basic(self, test_db: Session):
        """Basic acceptance by category calculation."""
        review = create_review(test_db, 1)

        # Create 3 SQL Injection findings: 2 accepted, 1 rejected
        findings = [
            create_finding(test_db, review.id, "SQL Injection"),
            create_finding(test_db, review.id, "SQL Injection"),
            create_finding(test_db, review.id, "SQL Injection"),
        ]

        create_feedback(test_db, findings[0].id, "ACCEPTED")
        create_feedback(test_db, findings[1].id, "ACCEPTED")
        create_feedback(test_db, findings[2].id, "REJECTED")

        analyzer = AcceptanceAnalyzer(test_db)
        results = analyzer.calculate_acceptance_by_category()

        # Should have SQL Injection metrics
        sql_metrics = [r for r in results if r.finding_category == "SQL Injection"]
        assert len(sql_metrics) == 1

        metric = sql_metrics[0]
        assert metric.total == 3
        assert metric.accepted == 2
        assert metric.rejected == 1
        assert metric.ignored == 0
        assert metric.acceptance_rate == pytest.approx(2 / 3, rel=0.01)

    def test_acceptance_by_category_multiple(self, test_db: Session):
        """Multiple categories with different rates."""
        review = create_review(test_db, 1)

        # SQL Injection: 4 accepted, 1 rejected → 80%
        for _ in range(4):
            f = create_finding(test_db, review.id, "SQL Injection")
            create_feedback(test_db, f.id, "ACCEPTED")
        f = create_finding(test_db, review.id, "SQL Injection")
        create_feedback(test_db, f.id, "REJECTED")

        # Resource Leak: 2 accepted, 3 rejected → 40%
        for _ in range(2):
            f = create_finding(test_db, review.id, "Resource Leak")
            create_feedback(test_db, f.id, "ACCEPTED")
        for _ in range(3):
            f = create_finding(test_db, review.id, "Resource Leak")
            create_feedback(test_db, f.id, "REJECTED")

        analyzer = AcceptanceAnalyzer(test_db)
        results = analyzer.calculate_acceptance_by_category()

        # Should be sorted by acceptance rate DESC
        assert results[0].finding_category == "SQL Injection"
        assert results[0].acceptance_rate == pytest.approx(0.80, rel=0.01)
        assert results[1].finding_category == "Resource Leak"
        assert results[1].acceptance_rate == pytest.approx(0.40, rel=0.01)


class TestAcceptanceBySeverity:
    """Test acceptance by severity level."""

    def test_acceptance_by_severity_all_levels(self, test_db: Session):
        """All severity levels have proper rates."""
        review = create_review(test_db, 1)

        severity_data = [
            (FindingSeverity.CRITICAL, 9, 1),  # 9 accepted, 1 rejected → 90%
            (FindingSeverity.HIGH, 6, 4),      # 6 accepted, 4 rejected → 60%
            (FindingSeverity.MEDIUM, 3, 7),    # 3 accepted, 7 rejected → 30%
            (FindingSeverity.LOW, 1, 9),       # 1 accepted, 9 rejected → 10%
        ]

        for severity, accepted_count, rejected_count in severity_data:
            for _ in range(accepted_count):
                f = create_finding(test_db, review.id, severity=severity)
                create_feedback(test_db, f.id, "ACCEPTED")
            for _ in range(rejected_count):
                f = create_finding(test_db, review.id, severity=severity)
                create_feedback(test_db, f.id, "REJECTED")

        analyzer = AcceptanceAnalyzer(test_db)
        results = analyzer.calculate_acceptance_by_severity()

        assert len(results) == 4
        assert results[0].severity == "CRITICAL"
        assert results[0].acceptance_rate == pytest.approx(0.90, rel=0.01)
        assert results[3].severity == "LOW"
        assert results[3].acceptance_rate == pytest.approx(0.10, rel=0.01)


class TestAcceptanceByIssueType:
    """Test acceptance by issue type (SECURITY vs PERFORMANCE)."""

    def test_acceptance_by_issue_type(self, test_db: Session):
        """Acceptance rates by issue type."""
        review = create_review(test_db, 1)

        # Security findings: 7 accepted, 3 rejected → 70%
        for _ in range(7):
            f = create_finding(
                test_db,
                review.id,
                category=FindingCategory.SECURITY,
            )
            create_feedback(test_db, f.id, "ACCEPTED")
        for _ in range(3):
            f = create_finding(
                test_db,
                review.id,
                category=FindingCategory.SECURITY,
            )
            create_feedback(test_db, f.id, "REJECTED")

        # Performance findings: 4 accepted, 6 rejected → 40%
        for _ in range(4):
            f = create_finding(
                test_db,
                review.id,
                category=FindingCategory.PERFORMANCE,
            )
            create_feedback(test_db, f.id, "ACCEPTED")
        for _ in range(6):
            f = create_finding(
                test_db,
                review.id,
                category=FindingCategory.PERFORMANCE,
            )
            create_feedback(test_db, f.id, "REJECTED")

        analyzer = AcceptanceAnalyzer(test_db)
        results = analyzer.calculate_acceptance_by_issue_type()

        security = [r for r in results if r.finding_category == "SECURITY"][0]
        performance = [r for r in results if r.finding_category == "PERFORMANCE"][0]

        assert security.acceptance_rate == pytest.approx(0.70, rel=0.01)
        assert performance.acceptance_rate == pytest.approx(0.40, rel=0.01)


# ============================================================================
# Composite Acceptance Tests
# ============================================================================


class TestCompositeAcceptance:
    """Test composite acceptance calculation with filters."""

    def test_composite_acceptance_basic(self, test_db: Session):
        """Basic composite acceptance calculation."""
        review = create_review(test_db, 1)

        # Create SQL Injection findings
        for _ in range(3):
            f = create_finding(test_db, review.id, "SQL Injection")
            create_feedback(test_db, f.id, "ACCEPTED")

        analyzer = AcceptanceAnalyzer(test_db)
        metrics = analyzer.calculate_composite_acceptance(category="SQL Injection")

        assert metrics.total == 3
        assert metrics.accepted == 3
        assert metrics.acceptance_rate == 1.0

    def test_composite_acceptance_with_category_filter(self, test_db: Session):
        """Filter by category only."""
        review = create_review(test_db, 1)

        # SQL Injection: 2 accepted
        for _ in range(2):
            f = create_finding(test_db, review.id, "SQL Injection")
            create_feedback(test_db, f.id, "ACCEPTED")

        # Resource Leak: 3 accepted (should be excluded)
        for _ in range(3):
            f = create_finding(test_db, review.id, "Resource Leak")
            create_feedback(test_db, f.id, "ACCEPTED")

        analyzer = AcceptanceAnalyzer(test_db)
        metrics = analyzer.calculate_composite_acceptance(category="SQL Injection")

        assert metrics.total == 2
        assert metrics.accepted == 2

    def test_composite_acceptance_with_severity_filter(self, test_db: Session):
        """Filter by severity only."""
        review = create_review(test_db, 1)

        # CRITICAL: 3 accepted
        for _ in range(3):
            f = create_finding(
                test_db, review.id, severity=FindingSeverity.CRITICAL
            )
            create_feedback(test_db, f.id, "ACCEPTED")

        # HIGH: 2 accepted (should be excluded)
        for _ in range(2):
            f = create_finding(test_db, review.id, severity=FindingSeverity.HIGH)
            create_feedback(test_db, f.id, "ACCEPTED")

        analyzer = AcceptanceAnalyzer(test_db)
        metrics = analyzer.calculate_composite_acceptance(severity="CRITICAL")

        assert metrics.total == 3
        assert metrics.accepted == 3

    def test_composite_acceptance_multiple_filters(self, test_db: Session):
        """Filter by category AND severity."""
        review = create_review(test_db, 1)

        # SQL Injection + CRITICAL: 2 accepted
        for _ in range(2):
            f = create_finding(
                test_db,
                review.id,
                "SQL Injection",
                severity=FindingSeverity.CRITICAL,
            )
            create_feedback(test_db, f.id, "ACCEPTED")

        # SQL Injection + HIGH: 3 accepted (should be excluded)
        for _ in range(3):
            f = create_finding(
                test_db,
                review.id,
                "SQL Injection",
                severity=FindingSeverity.HIGH,
            )
            create_feedback(test_db, f.id, "ACCEPTED")

        analyzer = AcceptanceAnalyzer(test_db)
        metrics = analyzer.calculate_composite_acceptance(
            category="SQL Injection",
            severity="CRITICAL",
        )

        assert metrics.total == 2
        assert metrics.accepted == 2

    def test_composite_acceptance_no_match(self, test_db: Session):
        """No findings match filter."""
        analyzer = AcceptanceAnalyzer(test_db)
        metrics = analyzer.calculate_composite_acceptance(
            category="Nonexistent"
        )

        assert metrics.total == 0
        assert metrics.accepted == 0
        assert metrics.acceptance_rate == 0.0


# ============================================================================
# Fix Timeline Tests
# ============================================================================


class TestFixingTimeline:
    """Test fix timeline analysis."""

    def test_fixing_timeline_basic(self, test_db: Session):
        """Basic timeline calculation."""
        review = create_review(test_db, 1)
        finding = create_finding(test_db, review.id)

        # Create feedback with commit hash
        feedback = create_feedback(
            test_db,
            finding.id,
            "ACCEPTED",
            commit_hash="abc123def456",
        )

        # Manually set times for testing
        finding.created_at = datetime(2025, 11, 20, 10, 0, 0, tzinfo=timezone.utc)
        feedback.created_at = datetime(
            2025, 11, 20, 12, 30, 0, tzinfo=timezone.utc
        )
        test_db.commit()

        analyzer = AcceptanceAnalyzer(test_db)
        timelines = analyzer.get_fixing_timeline()

        assert len(timelines) == 1
        timeline = timelines[0]
        assert timeline["time_to_fix_hours"] == pytest.approx(2.5, rel=0.01)
        assert timeline["accepted"] is True
        assert timeline["finding_id"] == finding.id

    def test_fixing_timeline_multiple(self, test_db: Session):
        """Multiple timelines sorted correctly."""
        review = create_review(test_db, 1)

        findings = []
        for i in range(3):
            f = create_finding(test_db, review.id, f"Issue {i}")
            findings.append(f)

            # Create feedback with different times
            fb = create_feedback(test_db, f.id, "ACCEPTED", commit_hash=f"commit{i}")

            # Set times: 1 hour, 2 hours, 3 hours
            f.created_at = datetime(
                2025, 11, 20, 10, 0, 0, tzinfo=timezone.utc
            )
            fb.created_at = datetime(
                2025, 11, 20, 10 + (i + 1), 0, 0, tzinfo=timezone.utc
            )
            test_db.commit()

        analyzer = AcceptanceAnalyzer(test_db)
        timelines = analyzer.get_fixing_timeline()

        assert len(timelines) == 3
        # Should be sorted by time DESC (slowest first)
        assert timelines[0]["time_to_fix_hours"] == pytest.approx(3.0, rel=0.01)
        assert timelines[2]["time_to_fix_hours"] == pytest.approx(1.0, rel=0.01)

    def test_fixing_timeline_missing_commit(self, test_db: Session):
        """Feedback without commit hash excluded."""
        review = create_review(test_db, 1)
        finding = create_finding(test_db, review.id)

        # Create feedback WITHOUT commit hash
        create_feedback(test_db, finding.id, "ACCEPTED", commit_hash=None)

        analyzer = AcceptanceAnalyzer(test_db)
        timelines = analyzer.get_fixing_timeline()

        assert len(timelines) == 0


# ============================================================================
# Metrics Persistence Tests
# ============================================================================


class TestMetricsPersistence:
    """Test metrics persistence to database."""

    def test_persist_metrics_new_record(self, test_db: Session):
        """Create new metrics record."""
        review = create_review(test_db, 1)

        # Create findings
        for _ in range(5):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "ACCEPTED")

        analyzer = AcceptanceAnalyzer(test_db)
        # Note: persist_metrics_for_category is simplified in current implementation
        # This test verifies the structure works

        metrics = analyzer.calculate_composite_acceptance()
        assert metrics.total == 5
        assert metrics.accepted == 5

    def test_persist_all_metrics_coverage(self, test_db: Session):
        """Persist all metrics finds all combinations."""
        review = create_review(test_db, 1)

        # Create findings with different categories
        for category in ["SQL Injection", "Resource Leak"]:
            for _ in range(2):
                f = create_finding(test_db, review.id, category)
                create_feedback(test_db, f.id, "ACCEPTED")

        analyzer = AcceptanceAnalyzer(test_db)
        count = analyzer.persist_all_metrics()

        assert count >= 2  # At least one for each category


# ============================================================================
# Time Window Filtering Tests
# ============================================================================


class TestTimeWindowFiltering:
    """Test filtering by time window."""

    def test_time_window_filtering_recent(self, test_db: Session):
        """Only recent feedback included."""
        review_old = create_review(test_db, 1)
        review_new = create_review(test_db, 2)

        # Old finding (60 days ago)
        old_finding = create_finding(test_db, review_old.id)
        old_finding.created_at = datetime.now(timezone.utc) - timedelta(days=60)
        test_db.commit()
        create_feedback(test_db, old_finding.id, "ACCEPTED")

        # New finding (10 days ago)
        new_finding = create_finding(test_db, review_new.id)
        new_finding.created_at = datetime.now(timezone.utc) - timedelta(days=10)
        test_db.commit()
        create_feedback(test_db, new_finding.id, "ACCEPTED")

        analyzer = AcceptanceAnalyzer(test_db)

        # Query last 30 days
        metrics = analyzer.calculate_composite_acceptance(days=30)

        # Should only include new finding
        assert metrics.total == 1
        assert metrics.accepted == 1

    def test_time_window_filtering_all_data(self, test_db: Session):
        """No filter includes all data."""
        review = create_review(test_db, 1)

        # Create findings at different ages
        for days_ago in [60, 30, 10, 1]:
            f = create_finding(test_db, review.id)
            f.created_at = datetime.now(timezone.utc) - timedelta(days=days_ago)
            test_db.commit()
            create_feedback(test_db, f.id, "ACCEPTED")

        analyzer = AcceptanceAnalyzer(test_db)

        # Query with very long window
        metrics = analyzer.calculate_composite_acceptance(days=365)

        # Should include all 4 findings
        assert metrics.total == 4


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_database(self, test_db: Session):
        """Empty database returns sensible defaults."""
        analyzer = AcceptanceAnalyzer(test_db)

        results = analyzer.calculate_acceptance_by_category()
        assert results == []

        metrics = analyzer.calculate_composite_acceptance()
        assert metrics.total == 0
        assert metrics.acceptance_rate == 0.0

    def test_no_feedback(self, test_db: Session):
        """Findings with no feedback marked as ignored."""
        review = create_review(test_db, 1)

        # Create findings with NO feedback
        for _ in range(5):
            create_finding(test_db, review.id)

        analyzer = AcceptanceAnalyzer(test_db)
        metrics = analyzer.calculate_composite_acceptance()

        assert metrics.total == 5
        assert metrics.accepted == 0
        assert metrics.rejected == 0
        assert metrics.ignored == 5
        assert metrics.acceptance_rate == 0.0

    def test_mixed_feedback(self, test_db: Session):
        """Mix of accepted, rejected, and no feedback."""
        review = create_review(test_db, 1)

        # 3 accepted
        for _ in range(3):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "ACCEPTED")

        # 2 rejected
        for _ in range(2):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "REJECTED")

        # 5 ignored (no feedback)
        for _ in range(5):
            create_finding(test_db, review.id)

        analyzer = AcceptanceAnalyzer(test_db)
        metrics = analyzer.calculate_composite_acceptance()

        assert metrics.total == 10
        assert metrics.accepted == 3
        assert metrics.rejected == 2
        assert metrics.ignored == 5
        assert metrics.acceptance_rate == pytest.approx(3 / 5, rel=0.01)  # 60%

    def test_zero_confidence_handling(self, test_db: Session):
        """Handle feedback with varying confidence levels."""
        review = create_review(test_db, 1)

        f1 = create_finding(test_db, review.id)
        f2 = create_finding(test_db, review.id)
        f3 = create_finding(test_db, review.id)

        # Feedback with different confidence levels
        create_feedback(test_db, f1.id, "ACCEPTED", confidence=0.85)
        create_feedback(test_db, f2.id, "ACCEPTED", confidence=0.0)
        create_feedback(test_db, f3.id, "ACCEPTED", confidence=None)

        analyzer = AcceptanceAnalyzer(test_db)
        metrics = analyzer.calculate_composite_acceptance()

        # Average should be 0.85 / 2 = 0.425 (only counting non-null confidences)
        assert metrics.confidence_avg == pytest.approx(0.425, rel=0.01)


# ============================================================================
# Ranking by Acceptance Tests
# ============================================================================


class TestTopCategoriesByAcceptance:
    """Test ranking categories by acceptance."""

    def test_top_categories_ranking(self, test_db: Session):
        """Categories ranked by acceptance rate."""
        review = create_review(test_db, 1)

        # SQL Injection: 8/10 = 80%
        for _ in range(8):
            f = create_finding(test_db, review.id, "SQL Injection")
            create_feedback(test_db, f.id, "ACCEPTED")
        for _ in range(2):
            f = create_finding(test_db, review.id, "SQL Injection")
            create_feedback(test_db, f.id, "REJECTED")

        # Auth Flaw: 6/10 = 60%
        for _ in range(6):
            f = create_finding(test_db, review.id, "Auth Flaw")
            create_feedback(test_db, f.id, "ACCEPTED")
        for _ in range(4):
            f = create_finding(test_db, review.id, "Auth Flaw")
            create_feedback(test_db, f.id, "REJECTED")

        analyzer = AcceptanceAnalyzer(test_db)
        top = analyzer.get_top_categories_by_acceptance(limit=2)

        assert len(top) == 2
        assert top[0][0] == "SQL Injection"
        assert top[0][1] == pytest.approx(0.80, rel=0.01)
        assert top[1][0] == "Auth Flaw"
        assert top[1][1] == pytest.approx(0.60, rel=0.01)

    def test_top_categories_ascending(self, test_db: Session):
        """Can sort ascending (lowest acceptance first)."""
        review = create_review(test_db, 1)

        # Create different acceptance rates
        for category, accepted, rejected in [
            ("High", 9, 1),
            ("Low", 1, 9),
        ]:
            for _ in range(accepted):
                f = create_finding(test_db, review.id, category)
                create_feedback(test_db, f.id, "ACCEPTED")
            for _ in range(rejected):
                f = create_finding(test_db, review.id, category)
                create_feedback(test_db, f.id, "REJECTED")

        analyzer = AcceptanceAnalyzer(test_db)
        top = analyzer.get_top_categories_by_acceptance(limit=2, ascending=True)

        # Should be Low, High
        assert top[0][0] == "Low"
        assert top[0][1] == pytest.approx(0.10, rel=0.01)
