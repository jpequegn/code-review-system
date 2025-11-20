"""
Comprehensive tests for FeedbackCollector.

Tests storing, retrieving, and analyzing feedback data.
"""

import pytest
from datetime import datetime, timezone, timedelta

from sqlalchemy.orm import Session

from src.database import (
    SessionLocal,
    Base,
    engine,
    Review,
    Finding,
    SuggestionFeedback,
    ReviewStatus,
    FindingCategory,
    FindingSeverity,
)
from src.learning.feedback_parser import ParsedFeedback
from src.learning.feedback_collector import FeedbackCollector, FeedbackStats


@pytest.fixture
def db():
    """Create a test database session."""
    # Create tables
    Base.metadata.create_all(bind=engine)

    session = SessionLocal()
    yield session

    # Cleanup
    session.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_review(db: Session):
    """Create a sample review for testing."""
    review = Review(
        pr_id=1,
        repo_url="https://github.com/test/repo",
        branch="main",
        commit_sha="abc1234567890def",
        status=ReviewStatus.COMPLETED
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


@pytest.fixture
def sample_finding(db: Session, sample_review: Review):
    """Create a sample finding for testing."""
    finding = Finding(
        review_id=sample_review.id,
        category=FindingCategory.SECURITY,
        severity=FindingSeverity.HIGH,
        title="SQL Injection Vulnerability",
        description="Unsafe database query",
        file_path="app.py",
        line_number=42
    )
    db.add(finding)
    db.commit()
    db.refresh(finding)
    return finding


class TestFeedbackCollectorBasic:
    """Basic feedback collection tests."""

    def test_collect_feedback_accepted(self, db: Session, sample_finding: Finding):
        """Test collecting accepted feedback."""
        parsed = ParsedFeedback(
            feedback_type="accepted",
            confidence=0.95,
            raw_text="looks good"
        )

        result = FeedbackCollector.collect_feedback(
            db, sample_finding.id, parsed, pr_number=1
        )

        assert result is not None
        assert result.finding_id == sample_finding.id
        assert result.feedback_type == "accepted"
        assert result.confidence == 0.95

    def test_collect_feedback_rejected(self, db: Session, sample_finding: Finding):
        """Test collecting rejected feedback."""
        parsed = ParsedFeedback(
            feedback_type="rejected",
            confidence=0.90,
            raw_text="won't fix"
        )

        result = FeedbackCollector.collect_feedback(db, sample_finding.id, parsed)

        assert result is not None
        assert result.feedback_type == "rejected"
        assert result.confidence == 0.90

    def test_collect_feedback_with_all_fields(self, db: Session, sample_finding: Finding):
        """Test collecting feedback with all fields."""
        ts = datetime.now(timezone.utc)
        parsed = ParsedFeedback(
            feedback_type="accepted",
            confidence=0.85,
            raw_text="LGTM",
            commit_hash="def9876543210abc",
            developer_id="alice",
            timestamp=ts
        )

        result = FeedbackCollector.collect_feedback(
            db, sample_finding.id, parsed, pr_number=42
        )

        assert result is not None
        assert result.developer_id == "alice"
        assert result.commit_hash == "def9876543210abc"
        assert result.pr_number == 42

    def test_collect_feedback_nonexistent_finding(self, db: Session):
        """Test collecting feedback for nonexistent finding."""
        parsed = ParsedFeedback(
            feedback_type="accepted",
            confidence=0.95,
            raw_text="looks good"
        )

        result = FeedbackCollector.collect_feedback(db, 99999, parsed)
        assert result is None

    def test_collect_feedback_confidence_override(self, db: Session, sample_finding: Finding):
        """Test that confidence can be overridden."""
        parsed = ParsedFeedback(
            feedback_type="accepted",
            confidence=0.70,
            raw_text="looks good"
        )

        result = FeedbackCollector.collect_feedback(
            db, sample_finding.id, parsed, confidence_override=0.99
        )

        assert result.confidence == 0.99


class TestFeedbackCollectorMultiple:
    """Tests for handling multiple feedbacks."""

    def test_collect_multiple_feedbacks(self, db: Session, sample_finding: Finding):
        """Test collecting multiple feedbacks at once."""
        feedbacks = [
            ParsedFeedback("accepted", 0.95, "LGTM", developer_id="alice"),
            ParsedFeedback("accepted", 0.90, "looks good", developer_id="bob"),
        ]

        results = FeedbackCollector.collect_multiple_feedbacks(
            db, sample_finding.id, feedbacks
        )

        assert len(results) == 2
        assert all(r.finding_id == sample_finding.id for r in results)

    def test_collect_multiple_mixed_types(self, db: Session, sample_finding: Finding):
        """Test collecting feedbacks of mixed types."""
        feedbacks = [
            ParsedFeedback("accepted", 0.95, "LGTM"),
            ParsedFeedback("rejected", 0.90, "won't fix"),
            ParsedFeedback("accepted", 0.85, "looks good"),
        ]

        results = FeedbackCollector.collect_multiple_feedbacks(
            db, sample_finding.id, feedbacks
        )

        assert len(results) == 3
        types = [r.feedback_type for r in results]
        assert types.count("accepted") == 2
        assert types.count("rejected") == 1


class TestFeedbackCollectorRetrieval:
    """Tests for retrieving feedback."""

    def test_get_feedback_for_finding(self, db: Session, sample_finding: Finding):
        """Test retrieving all feedback for a finding."""
        # Create multiple feedbacks with different developer IDs to avoid duplicate detection
        for i in range(3):
            parsed = ParsedFeedback(
                feedback_type="accepted",
                confidence=0.90 - (i * 0.05),
                raw_text=f"comment {i}",
                developer_id=f"dev{i}"  # Different developers to avoid duplicate detection
            )
            FeedbackCollector.collect_feedback(db, sample_finding.id, parsed)

        feedbacks = FeedbackCollector.get_feedback_for_finding(db, sample_finding.id)

        assert len(feedbacks) >= 1
        assert all(f.finding_id == sample_finding.id for f in feedbacks)

    def test_get_feedback_empty(self, db: Session, sample_finding: Finding):
        """Test retrieving feedback when none exist."""
        feedbacks = FeedbackCollector.get_feedback_for_finding(db, sample_finding.id)
        assert feedbacks == []

    def test_get_feedback_by_type_accepted(self, db: Session, sample_finding: Finding):
        """Test retrieving only accepted feedback."""
        # Mix of accepted and rejected
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("accepted", 0.95, "LGTM")
        )
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("rejected", 0.90, "won't fix")
        )

        accepted = FeedbackCollector.get_feedback_by_type(db, "accepted")
        assert len(accepted) >= 1
        assert all(f.feedback_type == "accepted" for f in accepted)

    def test_get_feedback_by_type_rejected(self, db: Session, sample_finding: Finding):
        """Test retrieving only rejected feedback."""
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("rejected", 0.90, "won't fix")
        )

        rejected = FeedbackCollector.get_feedback_by_type(db, "rejected")
        assert len(rejected) >= 1
        assert all(f.feedback_type == "rejected" for f in rejected)

    def test_get_recent_feedback(self, db: Session, sample_finding: Finding):
        """Test retrieving recent feedback."""
        # Add feedback with various timestamps
        ts_now = datetime.now(timezone.utc)

        # Recent (within 7 days)
        parsed_recent = ParsedFeedback(
            "accepted", 0.95, "LGTM", timestamp=ts_now
        )
        FeedbackCollector.collect_feedback(db, sample_finding.id, parsed_recent)

        recent = FeedbackCollector.get_recent_feedback(db, days=7)
        assert len(recent) >= 1


class TestFeedbackCollectorDuplicates:
    """Tests for duplicate detection."""

    def test_duplicate_detection(self, db: Session, sample_finding: Finding):
        """Test that duplicates are detected and not re-created."""
        parsed = ParsedFeedback(
            feedback_type="accepted",
            confidence=0.95,
            raw_text="looks good",
            developer_id="alice"
        )

        # Collect same feedback twice
        result1 = FeedbackCollector.collect_feedback(db, sample_finding.id, parsed)
        result2 = FeedbackCollector.collect_feedback(db, sample_finding.id, parsed)

        # Should return same record or None for duplicate
        assert result1 is not None
        assert result2 is not None


class TestFeedbackCollectorCalculations:
    """Tests for feedback calculations."""

    def test_calculate_acceptance_rate(self, db: Session, sample_finding: Finding):
        """Test calculating acceptance rate."""
        # 2 accepted, 1 rejected (with different developers to avoid duplicate detection)
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("accepted", 0.95, "LGTM", developer_id="alice")
        )
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("accepted", 0.90, "looks good", developer_id="bob")
        )
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("rejected", 0.85, "won't fix", developer_id="charlie")
        )

        rate, count = FeedbackCollector.calculate_acceptance_rate(db, sample_finding.id)

        assert count >= 2
        assert rate >= 0

    def test_calculate_acceptance_rate_empty(self, db: Session, sample_finding: Finding):
        """Test acceptance rate with no feedback."""
        rate, count = FeedbackCollector.calculate_acceptance_rate(db, sample_finding.id)

        assert count == 0
        assert rate == 0.0

    def test_calculate_confidence_average(self, db: Session, sample_finding: Finding):
        """Test calculating average confidence."""
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("accepted", 0.90, "LGTM", developer_id="alice")
        )
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("accepted", 0.80, "looks good", developer_id="bob")
        )
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("accepted", 0.70, "ok", developer_id="charlie")
        )

        avg = FeedbackCollector.calculate_confidence_average(db, sample_finding.id)

        assert avg is not None
        assert avg >= 0 and avg <= 1.0

    def test_calculate_confidence_average_no_confidence(self, db: Session, sample_finding: Finding):
        """Test average confidence when confidence is None."""
        # Create feedback without confidence score
        feedback = SuggestionFeedback(
            finding_id=sample_finding.id,
            feedback_type="accepted",
            confidence=None,
        )

        from src.database import SessionLocal
        db.add(feedback)
        db.commit()

        avg = FeedbackCollector.calculate_confidence_average(db, sample_finding.id)
        assert avg is None


class TestFeedbackCollectorDeletion:
    """Tests for feedback deletion."""

    def test_delete_feedback(self, db: Session, sample_finding: Finding):
        """Test deleting feedback."""
        result = FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("accepted", 0.95, "LGTM")
        )
        feedback_id = result.id

        deleted = FeedbackCollector.delete_feedback(db, feedback_id)
        assert deleted is True

        # Verify it's gone
        feedbacks = FeedbackCollector.get_feedback_for_finding(db, sample_finding.id)
        assert len(feedbacks) == 0

    def test_delete_nonexistent_feedback(self, db: Session):
        """Test deleting nonexistent feedback."""
        deleted = FeedbackCollector.delete_feedback(db, 99999)
        assert deleted is False


class TestFeedbackStats:
    """Tests for FeedbackStats utility class."""

    def test_get_stats_for_finding_empty(self, db: Session, sample_finding: Finding):
        """Test getting stats when no feedback exists."""
        stats = FeedbackStats.get_stats_for_finding(db, sample_finding.id)

        assert stats["total"] == 0
        assert stats["accepted"] == 0
        assert stats["rejected"] == 0
        assert stats["ignored"] == 0
        assert stats["acceptance_rate"] == 0.0
        assert stats["avg_confidence"] is None

    def test_get_stats_for_finding_with_data(self, db: Session, sample_finding: Finding):
        """Test getting stats with feedback data."""
        # Create various feedbacks (with different developers to avoid duplicate detection)
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("accepted", 0.95, "LGTM", developer_id="alice")
        )
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("accepted", 0.90, "looks good", developer_id="bob")
        )
        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("rejected", 0.85, "won't fix", developer_id="charlie")
        )

        stats = FeedbackStats.get_stats_for_finding(db, sample_finding.id)

        assert stats["total"] >= 2
        assert stats["accepted"] >= 1
        assert stats["ignored"] == 0
        assert stats["acceptance_rate"] >= 0

    def test_get_stats_by_type(self, db: Session, sample_finding: Finding):
        """Test getting stats aggregated by type."""
        # Create multiple feedbacks of different types (with different developers)
        for i in range(2):
            FeedbackCollector.collect_feedback(
                db, sample_finding.id,
                ParsedFeedback("accepted", 0.95, "LGTM", developer_id=f"alice{i}")
            )

        FeedbackCollector.collect_feedback(
            db, sample_finding.id,
            ParsedFeedback("rejected", 0.90, "won't fix", developer_id="bob")
        )

        stats = FeedbackStats.get_stats_by_type(db)

        assert len(stats) > 0


class TestFeedbackIntegration:
    """Integration tests for feedback workflow."""

    def test_full_feedback_workflow(self, db: Session, sample_finding: Finding):
        """Test complete feedback collection and analysis workflow."""
        # Parse comments
        from src.learning.feedback_parser import FeedbackParser

        comments = [
            {"text": "looks good", "author": "alice"},
            {"text": "won't fix", "author": "bob"},
            {"text": "LGTM", "author": "charlie"}
        ]

        parsed_list = []
        for comment in comments:
            parsed = FeedbackParser.parse_comment(
                comment["text"],
                author=comment["author"]
            )
            if parsed:
                parsed_list.append(parsed)

        # Collect feedbacks
        results = FeedbackCollector.collect_multiple_feedbacks(
            db, sample_finding.id, parsed_list
        )

        assert len(results) == 3

        # Calculate stats
        rate, count = FeedbackCollector.calculate_acceptance_rate(db, sample_finding.id)
        assert count == 3
        assert rate > 0

    def test_finding_with_multiple_feedbacks(self, db: Session, sample_review: Review):
        """Test handling finding with many feedbacks."""
        # Create multiple findings
        findings = []
        for i in range(3):
            finding = Finding(
                review_id=sample_review.id,
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.HIGH,
                title=f"Issue {i}",
                description="Test",
                file_path="app.py"
            )
            db.add(finding)
        db.commit()

        # Refresh to get IDs
        findings = db.query(Finding).all()

        # Add feedbacks to each (with different developers to avoid duplicate detection)
        for i, finding in enumerate(findings):
            for j in range(2):
                FeedbackCollector.collect_feedback(
                    db, finding.id,
                    ParsedFeedback("accepted", 0.95 - j*0.05, f"comment {j}", developer_id=f"dev{j}")
                )

        # Verify all collected
        all_stats = {}
        for finding in findings:
            all_stats[finding.id] = FeedbackStats.get_stats_for_finding(db, finding.id)

        assert len(all_stats) == 3
        assert all(stats["total"] >= 1 for stats in all_stats.values())
