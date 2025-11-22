"""
Pattern Learning & Detection Tests

Tests for learning and detecting recurring patterns, anti-patterns,
and team-specific best practices from code review feedback.
"""

import pytest
import json
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database import (
    Base,
    Review,
    Finding,
    SuggestionFeedback,
    PatternMetrics,
    FindingCategory,
    FindingSeverity,
)
from src.learning.pattern_learner import PatternLearner


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
) -> SuggestionFeedback:
    """Helper to create feedback."""
    feedback = SuggestionFeedback(
        finding_id=finding_id,
        feedback_type=feedback_type,
        developer_comment="Test feedback",
        created_at=datetime.now(timezone.utc),
    )
    db.add(feedback)
    db.commit()
    return feedback


# ============================================================================
# Pattern Detection Tests
# ============================================================================


class TestPatternDetection:
    """Test detection of recurring patterns."""

    def test_detect_pattern_multiple_findings(self, test_db: Session):
        """Detect pattern when finding appears multiple times."""
        review1 = create_review(test_db, pr_id=1)
        review2 = create_review(test_db, pr_id=2)

        # Create multiple findings with same title (SQL Injection)
        f1 = create_finding(test_db, review1.id, title="SQL Injection")
        f2 = create_finding(test_db, review2.id, title="SQL Injection")

        learner = PatternLearner(test_db)
        patterns = learner.detect_patterns()

        assert len(patterns) > 0
        sql_pattern = next((p for p in patterns if p["pattern_type"] == "SQL Injection"), None)
        assert sql_pattern is not None
        assert sql_pattern["occurrences"] == 2

    def test_detect_pattern_with_feedback(self, test_db: Session):
        """Pattern detection includes feedback analysis."""
        review = create_review(test_db)

        # Create findings with feedback
        for i in range(3):
            f = create_finding(test_db, review.id, title="Null Check")
            if i < 2:
                create_feedback(test_db, f.id, "ACCEPTED")
            else:
                create_feedback(test_db, f.id, "REJECTED")

        learner = PatternLearner(test_db)
        patterns = learner.detect_patterns()

        null_pattern = next((p for p in patterns if p["pattern_type"] == "Null Check"), None)
        assert null_pattern is not None
        assert null_pattern["acceptance_count"] == 2
        assert null_pattern["rejection_count"] == 1
        assert null_pattern["acceptance_rate"] >= 0.5

    def test_pattern_detection_empty_database(self, test_db: Session):
        """Pattern detection returns empty when no findings exist."""
        learner = PatternLearner(test_db)
        patterns = learner.detect_patterns()

        assert patterns == []

    def test_pattern_detection_single_finding(self, test_db: Session):
        """Single finding doesn't create pattern (needs minimum threshold)."""
        review = create_review(test_db)
        create_finding(test_db, review.id, title="Unique Finding")

        learner = PatternLearner(test_db)
        patterns = learner.detect_patterns(min_occurrences=2)

        assert len(patterns) == 0


# ============================================================================
# Anti-Pattern Detection Tests
# ============================================================================


class TestAntiPatternDetection:
    """Test detection of anti-patterns (patterns developers avoid)."""

    def test_detect_anti_pattern_low_acceptance(self, test_db: Session):
        """Anti-pattern has low acceptance rate (< 0.3)."""
        review = create_review(test_db)

        # Create findings that are mostly rejected
        for i in range(10):
            f = create_finding(test_db, review.id, title="Bad Practice")
            if i < 2:
                create_feedback(test_db, f.id, "ACCEPTED")
            else:
                create_feedback(test_db, f.id, "REJECTED")

        learner = PatternLearner(test_db)
        anti_patterns = learner.detect_anti_patterns(rejection_threshold=0.5)

        assert len(anti_patterns) > 0
        bad_practice = next((p for p in anti_patterns if p["pattern_type"] == "Bad Practice"), None)
        assert bad_practice is not None
        assert bad_practice["is_anti_pattern"] is True

    def test_not_anti_pattern_high_acceptance(self, test_db: Session):
        """Pattern with high acceptance is not anti-pattern."""
        review = create_review(test_db)

        # Create findings that are mostly accepted
        for i in range(10):
            f = create_finding(test_db, review.id, title="Good Practice")
            if i < 8:
                create_feedback(test_db, f.id, "ACCEPTED")
            else:
                create_feedback(test_db, f.id, "REJECTED")

        learner = PatternLearner(test_db)
        anti_patterns = learner.detect_anti_patterns(rejection_threshold=0.5)

        bad_practice = next((p for p in anti_patterns if p["pattern_type"] == "Good Practice"), None)
        assert bad_practice is None  # Should not be detected as anti-pattern


# ============================================================================
# Pattern Ranking Tests
# ============================================================================


class TestPatternRanking:
    """Test ranking patterns by various metrics."""

    def test_rank_patterns_by_occurrence(self, test_db: Session):
        """Rank patterns by number of occurrences."""
        review = create_review(test_db)

        # Common pattern (5 times)
        for _ in range(5):
            f = create_finding(test_db, review.id, title="Common Issue")
            create_feedback(test_db, f.id, "ACCEPTED")

        # Rare pattern (2 times, minimum for detection)
        for _ in range(2):
            f = create_finding(test_db, review.id, title="Rare Issue")
            create_feedback(test_db, f.id, "ACCEPTED")

        learner = PatternLearner(test_db)
        patterns = learner.detect_patterns()
        ranked = learner.rank_patterns(patterns, by="occurrences")

        assert ranked[0]["pattern_type"] == "Common Issue"
        assert ranked[1]["pattern_type"] == "Rare Issue"

    def test_rank_patterns_by_acceptance_rate(self, test_db: Session):
        """Rank patterns by acceptance rate."""
        review = create_review(test_db)

        # High acceptance pattern
        for i in range(5):
            f = create_finding(test_db, review.id, title="Good Finding")
            create_feedback(test_db, f.id, "ACCEPTED" if i < 4 else "REJECTED")

        # Low acceptance pattern
        for i in range(5):
            f = create_finding(test_db, review.id, title="Bad Finding")
            create_feedback(test_db, f.id, "REJECTED" if i < 4 else "ACCEPTED")

        learner = PatternLearner(test_db)
        patterns = learner.detect_patterns()
        ranked = learner.rank_patterns(patterns, by="acceptance_rate", descending=True)

        assert ranked[0]["pattern_type"] == "Good Finding"
        assert ranked[0]["acceptance_rate"] > ranked[1]["acceptance_rate"]


# ============================================================================
# Pattern Persistence Tests
# ============================================================================


class TestPatternPersistence:
    """Test persistence of patterns to database."""

    def test_persist_pattern_creates_record(self, test_db: Session):
        """Persisting pattern creates PatternMetrics record."""
        review = create_review(test_db)
        f = create_finding(test_db, review.id, title="SQL Injection")
        create_feedback(test_db, f.id, "ACCEPTED")

        learner = PatternLearner(test_db)
        pattern = {
            "pattern_type": "SQL Injection",
            "occurrences": 1,
            "acceptance_rate": 1.0,
            "is_anti_pattern": False,
            "files_affected": {"src/auth.py": 1},
        }

        record = learner.persist_pattern(pattern)

        assert record.pattern_type == "SQL Injection"
        assert record.occurrences == 1
        assert record.anti_pattern is False

    def test_persist_all_patterns(self, test_db: Session):
        """Persisting all patterns saves multiple records."""
        review = create_review(test_db)

        # Create 3 patterns with 2 occurrences each (minimum for detection)
        for i in range(3):
            for _ in range(2):
                f = create_finding(test_db, review.id, title=f"Pattern {i}")
                create_feedback(test_db, f.id, "ACCEPTED")

        learner = PatternLearner(test_db)
        patterns = learner.detect_patterns()
        count = learner.persist_all_patterns(patterns)

        assert count == 3

        # Verify in database
        records = test_db.query(PatternMetrics).all()
        assert len(records) == 3


# ============================================================================
# Best Practice Identification Tests
# ============================================================================


class TestBestPracticeIdentification:
    """Test identification of team best practices."""

    def test_identify_best_practice_high_acceptance(self, test_db: Session):
        """Best practice has high acceptance rate and low variance."""
        review = create_review(test_db)

        # Consistent high acceptance
        for _ in range(8):
            f = create_finding(test_db, review.id, title="Best Practice")
            create_feedback(test_db, f.id, "ACCEPTED")

        for _ in range(2):
            f = create_finding(test_db, review.id, title="Best Practice")
            create_feedback(test_db, f.id, "REJECTED")

        learner = PatternLearner(test_db)
        best_practices = learner.identify_best_practices(acceptance_threshold=0.75)

        assert len(best_practices) > 0
        bp = next((p for p in best_practices if p["pattern_type"] == "Best Practice"), None)
        assert bp is not None
        assert bp["acceptance_rate"] >= 0.75

    def test_best_practice_sorted_by_acceptance(self, test_db: Session):
        """Best practices ranked by acceptance rate (highest first)."""
        review = create_review(test_db)

        # Practice A: 80% acceptance
        for i in range(10):
            f = create_finding(test_db, review.id, title="Practice A")
            create_feedback(test_db, f.id, "ACCEPTED" if i < 8 else "REJECTED")

        # Practice B: 90% acceptance
        for i in range(10):
            f = create_finding(test_db, review.id, title="Practice B")
            create_feedback(test_db, f.id, "ACCEPTED" if i < 9 else "REJECTED")

        learner = PatternLearner(test_db)
        best_practices = learner.identify_best_practices(acceptance_threshold=0.75)

        assert best_practices[0]["pattern_type"] == "Practice B"
        assert best_practices[1]["pattern_type"] == "Practice A"


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_pattern_with_no_feedback(self, test_db: Session):
        """Pattern without feedback is detected but has no acceptance data."""
        review = create_review(test_db)

        # Finding with no feedback
        create_finding(test_db, review.id, title="No Feedback")

        learner = PatternLearner(test_db)
        patterns = learner.detect_patterns()

        # Should still detect the pattern occurrence
        pattern = next((p for p in patterns if p["pattern_type"] == "No Feedback"), None)
        assert pattern is None  # Single occurrence, requires threshold

    def test_pattern_case_sensitivity(self, test_db: Session):
        """Pattern detection treats titles with different cases as different patterns."""
        review = create_review(test_db)

        create_finding(test_db, review.id, title="SQL Injection")
        create_finding(test_db, review.id, title="sql injection")

        learner = PatternLearner(test_db)
        patterns = learner.detect_patterns(min_occurrences=1)

        assert len(patterns) == 2  # Two different patterns due to case

    def test_empty_pattern_list_returns_empty(self, test_db: Session):
        """Ranking empty pattern list returns empty."""
        learner = PatternLearner(test_db)
        ranked = learner.rank_patterns([])

        assert ranked == []
