"""
Confidence Calibration & Tuning Tests

Tests for recalibrating confidence thresholds based on actual acceptance outcomes.
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
    FindingCategory,
    FindingSeverity,
    ConfidenceCalibration,
)
from src.learning.confidence_tuner import ConfidenceTuner
from src.learning.metrics import ConfidenceBin


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_db() -> Session:
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    # Enable foreign keys for SQLite
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
    confidence: float = None,
    commit_hash: str = None,
) -> SuggestionFeedback:
    """Helper to create feedback."""
    feedback = SuggestionFeedback(
        finding_id=finding_id,
        feedback_type=feedback_type,
        confidence=confidence,
        commit_hash=commit_hash,
        developer_comment="Test feedback",
        created_at=datetime.now(timezone.utc),
    )
    db.add(feedback)
    db.commit()
    return feedback


# ============================================================================
# Confidence Binning Tests
# ============================================================================


class TestConfidenceBinning:
    """Test confidence score to bin mapping."""

    def test_bin_confidence_boundaries(self, test_db: Session):
        """Test binning at bin boundaries."""
        tuner = ConfidenceTuner(test_db)

        # Bin 0: 0.0-0.1
        assert tuner.bin_confidence(0.0) == 0
        assert tuner.bin_confidence(0.05) == 0
        assert tuner.bin_confidence(0.09) == 0

        # Bin 5: 0.5-0.6
        assert tuner.bin_confidence(0.5) == 5
        assert tuner.bin_confidence(0.55) == 5
        assert tuner.bin_confidence(0.59) == 5

        # Bin 9: 0.9-1.0
        assert tuner.bin_confidence(0.9) == 9
        assert tuner.bin_confidence(0.95) == 9
        assert tuner.bin_confidence(1.0) == 9

    def test_bin_confidence_edge_cases(self, test_db: Session):
        """Test edge cases for binning."""
        tuner = ConfidenceTuner(test_db)

        # Below 0.1 goes to bin 0
        assert tuner.bin_confidence(0.001) == 0
        assert tuner.bin_confidence(0.099) == 0

        # 1.0 goes to bin 9
        assert tuner.bin_confidence(1.0) == 9

    def test_get_confidence_range_for_bin(self, test_db: Session):
        """Test getting confidence range from bin ID."""
        tuner = ConfidenceTuner(test_db)

        min_0, max_0 = tuner.get_confidence_range_for_bin(0)
        assert abs(min_0 - 0.0) < 0.001 and abs(max_0 - 0.1) < 0.001

        min_5, max_5 = tuner.get_confidence_range_for_bin(5)
        assert abs(min_5 - 0.5) < 0.001 and abs(max_5 - 0.6) < 0.001

        min_9, max_9 = tuner.get_confidence_range_for_bin(9)
        assert abs(min_9 - 0.9) < 0.001 and abs(max_9 - 1.0) < 0.001


# ============================================================================
# Bin Statistics Calculation Tests
# ============================================================================


class TestBinStatistics:
    """Test calculation of statistics for confidence bins."""

    def test_calculate_bin_statistics_single_bin(self, test_db: Session):
        """Calculate statistics for one bin with mixed feedback."""
        review = create_review(test_db)

        # Create findings with confidence in bin 9 (0.9-1.0)
        # 7 accepted, 2 rejected, 1 no feedback
        for _ in range(7):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "ACCEPTED", confidence=0.95)

        for _ in range(2):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "REJECTED", confidence=0.92)

        f = create_finding(test_db, review.id)
        # No feedback for this one

        tuner = ConfidenceTuner(test_db)
        bin_stats = tuner.calculate_bin_statistics(9)

        assert bin_stats.bin_id == 9
        assert bin_stats.sample_size == 9  # Total feedback items (7 accepted + 2 rejected)
        assert bin_stats.original_acceptance_rate >= 0.0
        assert bin_stats.precision >= 0.0
        assert bin_stats.recall >= 0.0
        assert bin_stats.f1_score >= 0.0

    def test_calculate_all_bin_statistics(self, test_db: Session):
        """Calculate statistics for all bins."""
        review = create_review(test_db)

        # Create findings across different bins
        # Bin 0-1 (low confidence, all rejected)
        for _ in range(3):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "REJECTED", confidence=0.05)

        # Bin 8-9 (high confidence, all accepted)
        for _ in range(5):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "ACCEPTED", confidence=0.95)

        tuner = ConfidenceTuner(test_db)
        bins = tuner.calculate_all_bin_statistics()

        assert len(bins) == 10
        assert all(isinstance(b, ConfidenceBin) for b in bins)
        assert bins[0].bin_id == 0
        assert bins[9].bin_id == 9

    def test_bin_statistics_empty_bin(self, test_db: Session):
        """Calculate statistics for empty bin returns zeros."""
        tuner = ConfidenceTuner(test_db)
        bin_stats = tuner.calculate_bin_statistics(5)  # Empty bin

        assert bin_stats.bin_id == 5
        assert bin_stats.sample_size == 0
        assert bin_stats.precision == 0.0
        assert bin_stats.recall == 0.0
        assert bin_stats.f1_score == 0.0


# ============================================================================
# Threshold Computation Tests
# ============================================================================


class TestThresholdComputation:
    """Test optimal threshold computation."""

    def test_compute_optimal_threshold_defaults(self, test_db: Session):
        """Compute threshold with default targets (precision=0.85, recall=0.70)."""
        review = create_review(test_db)

        # Create diverse feedback
        for i in range(10):
            f = create_finding(test_db, review.id)
            confidence = 0.5 + (i * 0.04)  # 0.5 to 0.86
            feedback_type = "ACCEPTED" if i < 7 else "REJECTED"
            create_feedback(test_db, f.id, feedback_type, confidence=confidence)

        tuner = ConfidenceTuner(test_db)
        threshold = tuner.compute_optimal_threshold()

        assert threshold.bin_id >= 0
        assert threshold.bin_id <= 9
        assert threshold.precision >= 0.0
        assert threshold.recall >= 0.0

    def test_compute_optimal_threshold_custom_targets(self, test_db: Session):
        """Compute threshold with custom precision/recall targets."""
        review = create_review(test_db)

        # Create feedback
        for i in range(8):
            f = create_finding(test_db, review.id)
            confidence = 0.6 + (i * 0.04)
            feedback_type = "ACCEPTED" if i < 6 else "REJECTED"
            create_feedback(test_db, f.id, feedback_type, confidence=confidence)

        tuner = ConfidenceTuner(test_db)
        threshold = tuner.compute_optimal_threshold(
            target_precision=0.90, target_recall=0.80
        )

        assert 0 <= threshold.bin_id <= 9

    def test_suggest_calibrated_thresholds(self, test_db: Session):
        """Suggest thresholds for aggressive/balanced/conservative."""
        review = create_review(test_db)

        # Create feedback across multiple confidence levels
        for i in range(20):
            f = create_finding(test_db, review.id)
            confidence = i / 20.0  # 0.0 to 0.95
            feedback_type = "ACCEPTED" if (i % 3) != 0 else "REJECTED"
            create_feedback(test_db, f.id, feedback_type, confidence=confidence)

        tuner = ConfidenceTuner(test_db)
        thresholds = tuner.suggest_calibrated_thresholds()

        assert "aggressive" in thresholds
        assert "balanced" in thresholds
        assert "conservative" in thresholds

        # aggressive should be lowest, conservative should be highest
        assert thresholds["aggressive"] <= thresholds["balanced"]
        assert thresholds["balanced"] <= thresholds["conservative"]


# ============================================================================
# Calibration Persistence Tests
# ============================================================================


class TestCalibrationPersistence:
    """Test persistence of calibration to database."""

    def test_persist_calibration_single_bin(self, test_db: Session):
        """Persist calibration for one bin."""
        review = create_review(test_db)

        # Create feedback for bin 5
        for _ in range(5):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "ACCEPTED", confidence=0.55)

        tuner = ConfidenceTuner(test_db)
        record = tuner.persist_calibration(5)

        assert record.bin_id == 5
        assert record.sample_size >= 0
        assert record.precision >= 0.0
        assert record.recall >= 0.0

        # Verify in database
        persisted = (
            test_db.query(ConfidenceCalibration)
            .filter(ConfidenceCalibration.bin_id == 5)
            .first()
        )
        assert persisted is not None
        assert persisted.precision == record.precision

    def test_persist_all_calibrations(self, test_db: Session):
        """Persist calibration for all 10 bins."""
        review = create_review(test_db)

        # Create feedback across all bins
        for i in range(10):
            f = create_finding(test_db, review.id)
            confidence = (i * 0.1) + 0.05
            create_feedback(test_db, f.id, "ACCEPTED", confidence=confidence)

        tuner = ConfidenceTuner(test_db)
        count = tuner.persist_all_calibrations()

        assert count == 10

        # Verify all bins persisted
        for bin_id in range(10):
            record = (
                test_db.query(ConfidenceCalibration)
                .filter(ConfidenceCalibration.bin_id == bin_id)
                .first()
            )
            assert record is not None

    def test_persist_calibration_update_existing(self, test_db: Session):
        """Persisting same bin twice updates existing record."""
        review = create_review(test_db)

        # Create initial feedback
        f = create_finding(test_db, review.id)
        create_feedback(test_db, f.id, "ACCEPTED", confidence=0.55)

        tuner = ConfidenceTuner(test_db)
        tuner.persist_calibration(5)

        # Should have 1 record initially
        initial_record = (
            test_db.query(ConfidenceCalibration)
            .filter(ConfidenceCalibration.bin_id == 5)
            .first()
        )
        initial_sample_size = initial_record.sample_size

        # Add more feedback
        f = create_finding(test_db, review.id)
        create_feedback(test_db, f.id, "REJECTED", confidence=0.55)

        # Persist again
        record2 = tuner.persist_calibration(5)

        # Should still only have 1 record (updated, not duplicated)
        count = (
            test_db.query(ConfidenceCalibration)
            .filter(ConfidenceCalibration.bin_id == 5)
            .count()
        )
        assert count == 1

        # Sample size should increase
        assert record2.sample_size > initial_sample_size


# ============================================================================
# Calibration Application Tests
# ============================================================================


class TestCalibrationApplication:
    """Test applying calibration to adjust confidence scores."""

    def test_apply_calibration_balanced(self, test_db: Session):
        """Apply balanced calibration to a confidence score."""
        review = create_review(test_db)

        # Create feedback for bin 9
        for _ in range(8):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "ACCEPTED", confidence=0.95)

        for _ in range(2):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "REJECTED", confidence=0.95)

        tuner = ConfidenceTuner(test_db)
        tuner.persist_all_calibrations()

        # Apply calibration
        original = 0.95
        calibrated = tuner.apply_calibration_to_finding(
            original, recalibration_mode="balanced"
        )

        # Calibrated should reflect actual acceptance rate
        assert 0.0 <= calibrated <= 1.0

    def test_apply_calibration_aggressive(self, test_db: Session):
        """Apply aggressive calibration (slightly boost score)."""
        review = create_review(test_db)

        for _ in range(5):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "ACCEPTED", confidence=0.75)

        tuner = ConfidenceTuner(test_db)
        tuner.persist_all_calibrations()

        original = 0.75
        balanced = tuner.apply_calibration_to_finding(original, "balanced")
        aggressive = tuner.apply_calibration_to_finding(original, "aggressive")

        # Aggressive should be >= balanced (boosted)
        assert aggressive >= balanced

    def test_apply_calibration_conservative(self, test_db: Session):
        """Apply conservative calibration (slightly reduce score)."""
        review = create_review(test_db)

        for _ in range(5):
            f = create_finding(test_db, review.id)
            create_feedback(test_db, f.id, "ACCEPTED", confidence=0.75)

        tuner = ConfidenceTuner(test_db)
        tuner.persist_all_calibrations()

        original = 0.75
        balanced = tuner.apply_calibration_to_finding(original, "balanced")
        conservative = tuner.apply_calibration_to_finding(original, "conservative")

        # Conservative should be <= balanced (reduced)
        assert conservative <= balanced

    def test_apply_calibration_without_persistence(self, test_db: Session):
        """Apply calibration to score when bin not yet calibrated."""
        tuner = ConfidenceTuner(test_db)

        # No feedback/calibration created
        original = 0.85
        calibrated = tuner.apply_calibration_to_finding(original)

        # Should fallback to original
        assert calibrated == original


# ============================================================================
# Reporting Tests
# ============================================================================


class TestCalibrationReporting:
    """Test calibration report generation."""

    def test_get_calibration_report(self, test_db: Session):
        """Generate comprehensive calibration report."""
        review = create_review(test_db)

        # Create feedback
        for i in range(15):
            f = create_finding(test_db, review.id)
            confidence = 0.3 + (i * 0.05)
            feedback_type = "ACCEPTED" if i < 10 else "REJECTED"
            create_feedback(test_db, f.id, feedback_type, confidence=confidence)

        tuner = ConfidenceTuner(test_db)
        tuner.persist_all_calibrations()
        report = tuner.get_calibration_report()

        assert report is not None
        assert "bins" in report
        assert "recommended_thresholds" in report
        assert "improvement_summary" in report
        assert len(report["bins"]) == 10

    def test_calibration_report_empty_database(self, test_db: Session):
        """Generate report from empty database."""
        tuner = ConfidenceTuner(test_db)
        report = tuner.get_calibration_report()

        assert report is not None
        assert len(report["bins"]) == 10
