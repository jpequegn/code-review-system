"""
Tests for feedback collection and learning system.

Tests cover feedback recording, accuracy calculations, learning metrics,
and adaptive severity adjustments.
"""

import pytest
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session

from src.database import (
    Review, Finding, FindingFeedback, ProductionIssue, LearningMetrics,
    ReviewStatus, FindingCategory, FindingSeverity, FeedbackType, IssueValidation,
    SessionLocal, init_db, engine, Base
)
from src.feedback.feedback import FeedbackCollector
from src.learning.learner import (
    LearningEngine, HistoricalAccuracy, PatternLearner, PersonalThresholdCalculator
)
from src.analysis.adaptive_severity import AdaptiveSeverityAdjuster
from src.reporting.learning_report import LearningReportGenerator


@pytest.fixture(scope="function")
def test_db():
    """Create a test database and session."""
    # Create tables
    Base.metadata.create_all(bind=engine)

    db = SessionLocal()
    yield db

    # Cleanup
    db.close()
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def sample_review(test_db):
    """Create a sample review with findings."""
    review = Review(
        pr_id=1,
        repo_url="https://github.com/test/repo.git",
        branch="main",
        commit_sha="abc123def456",
        status=ReviewStatus.COMPLETED
    )
    test_db.add(review)
    test_db.commit()
    test_db.refresh(review)
    return review


@pytest.fixture
def sample_findings(test_db, sample_review):
    """Create sample findings."""
    findings = []
    for i in range(5):
        category = FindingCategory.SECURITY if i % 2 == 0 else FindingCategory.PERFORMANCE
        severity = FindingSeverity.HIGH if i < 3 else FindingSeverity.MEDIUM

        finding = Finding(
            review_id=sample_review.id,
            category=category,
            severity=severity,
            title=f"Finding {i}",
            description=f"Test finding {i}",
            file_path="test.py",
            line_number=10 + i
        )
        test_db.add(finding)

    test_db.commit()
    findings = test_db.query(Finding).all()
    return findings


class TestFeedbackCollection:
    """Test feedback collection functionality."""

    def test_record_finding_feedback(self, test_db, sample_findings):
        """Test recording finding feedback."""
        collector = FeedbackCollector(test_db)
        finding = sample_findings[0]

        feedback = collector.record_finding_feedback(
            finding_id=finding.id,
            feedback_type=FeedbackType.HELPFUL,
            validation=IssueValidation.CONFIRMED,
            user_notes="Good catch!",
            helpful=True
        )

        assert feedback.finding_id == finding.id
        assert feedback.feedback_type == FeedbackType.HELPFUL
        assert feedback.validation == IssueValidation.CONFIRMED
        assert feedback.helpful is True

    def test_record_issue_validation(self, test_db, sample_findings):
        """Test recording issue validation."""
        collector = FeedbackCollector(test_db)
        finding = sample_findings[0]

        # First validation
        feedback = collector.record_issue_validation(
            finding_id=finding.id,
            validation=IssueValidation.CONFIRMED,
            helpful=True
        )

        assert feedback.validation == IssueValidation.CONFIRMED

        # Update validation
        updated = collector.record_issue_validation(
            finding_id=finding.id,
            validation=IssueValidation.FIXED,
            helpful=True
        )

        assert updated.validation == IssueValidation.FIXED

    def test_record_production_bug(self, test_db):
        """Test recording production bugs."""
        collector = FeedbackCollector(test_db)

        issue = collector.record_production_bug(
            repo_url="https://github.com/test/repo.git",
            description="Critical auth bug in production",
            severity=FindingSeverity.CRITICAL,
            date_discovered=datetime.now(timezone.utc),
            file_path="auth.py",
            time_to_fix_minutes=120,
            related_finding_ids=[1, 2, 3]
        )

        assert issue.description == "Critical auth bug in production"
        assert issue.severity == FindingSeverity.CRITICAL
        assert "1,2,3" in issue.related_finding_ids

    def test_get_feedback_for_finding(self, test_db, sample_findings):
        """Test retrieving feedback for specific finding."""
        collector = FeedbackCollector(test_db)
        finding = sample_findings[0]

        # Record feedback
        collector.record_finding_feedback(
            finding_id=finding.id,
            feedback_type=FeedbackType.FALSE_POSITIVE
        )

        # Retrieve feedback
        feedback = collector.get_feedback_for_finding(finding.id)
        assert feedback is not None
        assert feedback.finding_id == finding.id

    def test_get_false_positives(self, test_db, sample_findings):
        """Test getting false positive findings."""
        collector = FeedbackCollector(test_db)

        # Record mix of feedback
        collector.record_finding_feedback(
            finding_id=sample_findings[0].id,
            feedback_type=FeedbackType.FALSE_POSITIVE
        )
        collector.record_finding_feedback(
            finding_id=sample_findings[1].id,
            feedback_type=FeedbackType.HELPFUL
        )

        false_positives = collector.get_false_positives()
        assert len(false_positives) == 1
        assert false_positives[0].finding_id == sample_findings[0].id


class TestHistoricalAccuracy:
    """Test historical accuracy tracking."""

    def test_calculate_accuracy(self, test_db, sample_findings):
        """Test accuracy calculation."""
        collector = FeedbackCollector(test_db)
        accuracy_tracker = HistoricalAccuracy(test_db)

        # Record feedback with mix of outcomes
        collector.record_finding_feedback(
            finding_id=sample_findings[0].id,
            feedback_type=FeedbackType.HELPFUL,
            validation=IssueValidation.CONFIRMED
        )
        collector.record_finding_feedback(
            finding_id=sample_findings[1].id,
            feedback_type=FeedbackType.FALSE_POSITIVE,
            validation=IssueValidation.FALSE
        )
        collector.record_finding_feedback(
            finding_id=sample_findings[2].id,
            feedback_type=FeedbackType.HELPFUL,
            validation=IssueValidation.CONFIRMED
        )

        accuracy = accuracy_tracker.calculate_accuracy()

        assert accuracy["total_findings"] == 3
        assert accuracy["confirmed"] == 2
        assert accuracy["false_positives"] == 1
        assert accuracy["accuracy"] > 0

    def test_per_category_accuracy(self, test_db, sample_findings):
        """Test per-category accuracy calculation."""
        accuracy_tracker = HistoricalAccuracy(test_db)

        category_accuracy = accuracy_tracker.get_per_category_accuracy()

        # Should return dict for all categories
        assert FindingCategory.SECURITY in category_accuracy
        assert FindingCategory.PERFORMANCE in category_accuracy
        assert FindingCategory.BEST_PRACTICE in category_accuracy

    def test_per_severity_accuracy(self, test_db, sample_findings):
        """Test per-severity accuracy calculation."""
        accuracy_tracker = HistoricalAccuracy(test_db)

        severity_accuracy = accuracy_tracker.get_per_severity_accuracy()

        # Should return dict for all severities
        assert FindingSeverity.CRITICAL in severity_accuracy
        assert FindingSeverity.HIGH in severity_accuracy
        assert FindingSeverity.MEDIUM in severity_accuracy
        assert FindingSeverity.LOW in severity_accuracy


class TestPatternLearner:
    """Test pattern learning."""

    def test_get_acted_on_patterns(self, test_db, sample_findings):
        """Test identifying patterns in acted-on findings."""
        collector = FeedbackCollector(test_db)
        learner = PatternLearner(test_db)

        # Record feedback
        for i, finding in enumerate(sample_findings[:3]):
            collector.record_finding_feedback(
                finding_id=finding.id,
                feedback_type=FeedbackType.HELPFUL,
                helpful=True
            )

        patterns = learner.get_acted_on_patterns()

        # Should have patterns for all categories
        assert len(patterns) > 0
        for category, pattern_data in patterns.items():
            assert "total" in pattern_data
            assert "acted_on" in pattern_data


class TestPersonalThresholds:
    """Test personal threshold calculation."""

    def test_calculate_confidence_threshold(self, test_db, sample_findings):
        """Test calculating confidence threshold."""
        calculator = PersonalThresholdCalculator(test_db)

        threshold = calculator.calculate_confidence_threshold(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH
        )

        # Should return a value between 0 and 1
        assert 0.0 <= threshold <= 1.0

    def test_should_include_finding(self, test_db, sample_findings):
        """Test deciding whether to include finding."""
        calculator = PersonalThresholdCalculator(test_db)

        # High confidence should be included
        should_include = calculator.should_include_finding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            confidence=0.95
        )

        assert should_include is True

    def test_get_personal_thresholds(self, test_db):
        """Test getting all personal thresholds."""
        calculator = PersonalThresholdCalculator(test_db)

        thresholds = calculator.get_personal_thresholds()

        # Should have thresholds for all category/severity combos
        assert len(thresholds) == len(FindingCategory) * len(FindingSeverity)


class TestLearningEngine:
    """Test complete learning engine."""

    def test_update_learning_metrics(self, test_db, sample_findings):
        """Test updating learning metrics."""
        collector = FeedbackCollector(test_db)
        engine = LearningEngine(test_db)

        # Record feedback
        for finding in sample_findings[:3]:
            collector.record_finding_feedback(
                finding_id=finding.id,
                feedback_type=FeedbackType.HELPFUL,
                validation=IssueValidation.CONFIRMED
            )

        # Update metrics
        engine.update_learning_metrics(
            repo_url="https://github.com/test/repo.git",
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH
        )

        # Check metrics were created/updated
        metrics = test_db.query(LearningMetrics).filter(
            LearningMetrics.repo_url == "https://github.com/test/repo.git"
        ).first()

        assert metrics is not None

    def test_get_learning_report(self, test_db, sample_findings):
        """Test generating learning report."""
        collector = FeedbackCollector(test_db)
        engine = LearningEngine(test_db)

        # Record feedback
        for finding in sample_findings:
            collector.record_finding_feedback(
                finding_id=finding.id,
                feedback_type=FeedbackType.HELPFUL,
                validation=IssueValidation.CONFIRMED
            )

        report = engine.get_learning_report()

        assert "overall" in report
        assert "by_category" in report
        assert "patterns" in report
        assert "thresholds" in report


class TestAdaptiveSeverity:
    """Test adaptive severity adjustment."""

    def test_adjust_severity_high_fp_rate(self, test_db, sample_review):
        """Test severity demotion with high false positive rate."""
        collector = FeedbackCollector(test_db)
        adjuster = AdaptiveSeverityAdjuster(
            test_db,
            "https://github.com/test/repo.git"
        )

        # Create metrics with high FP rate
        metrics = LearningMetrics(
            repo_url="https://github.com/test/repo.git",
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            total_findings=10,
            confirmed_findings=2,
            false_positives=6,
            accuracy=20.0,
            precision=25.0,
            recall=20.0,
            false_positive_rate=60.0
        )
        test_db.add(metrics)
        test_db.commit()

        # High FP rate should demote severity
        adjusted, factor = adjuster.adjust_severity(
            category=FindingCategory.SECURITY,
            base_severity=FindingSeverity.CRITICAL
        )

        # Should be demoted from CRITICAL
        assert adjusted != FindingSeverity.CRITICAL or factor < 1.0

    def test_adjust_confidence(self, test_db):
        """Test confidence adjustment."""
        adjuster = AdaptiveSeverityAdjuster(
            test_db,
            "https://github.com/test/repo.git"
        )

        adjusted = adjuster.adjust_confidence(
            category=FindingCategory.SECURITY,
            base_severity=FindingSeverity.HIGH,
            confidence=0.8
        )

        # Should return adjusted value
        assert 0.0 <= adjusted <= 1.0

    def test_should_include_finding(self, test_db):
        """Test deciding whether to include finding."""
        adjuster = AdaptiveSeverityAdjuster(
            test_db,
            "https://github.com/test/repo.git"
        )

        # High confidence should be included
        should_include = adjuster.should_include_finding(
            category=FindingCategory.SECURITY,
            base_severity=FindingSeverity.CRITICAL,
            confidence=0.95
        )

        assert should_include is True


class TestLearningReport:
    """Test learning report generation."""

    def test_generate_weekly_report(self, test_db, sample_findings):
        """Test generating weekly report."""
        collector = FeedbackCollector(test_db)
        generator = LearningReportGenerator(test_db)

        # Record some feedback
        for finding in sample_findings[:2]:
            collector.record_finding_feedback(
                finding_id=finding.id,
                feedback_type=FeedbackType.HELPFUL
            )

        report = generator.generate_weekly_report()

        assert "period" in report
        assert report["period"] == "Weekly"
        assert "accuracy_metrics" in report
        assert "feedback_counts" in report
        assert "summary" in report

    def test_generate_monthly_report(self, test_db, sample_findings):
        """Test generating monthly report."""
        collector = FeedbackCollector(test_db)
        generator = LearningReportGenerator(test_db)

        # Record feedback
        for finding in sample_findings:
            collector.record_finding_feedback(
                finding_id=finding.id,
                feedback_type=FeedbackType.HELPFUL
            )

        report = generator.generate_monthly_report()

        assert report["period"] == "Monthly"
        assert "recommendations" in report

    def test_generate_overall_report(self, test_db, sample_findings):
        """Test generating overall report."""
        generator = LearningReportGenerator(test_db)

        report = generator.generate_overall_report()

        assert report["period"] == "Overall"
        assert "date_range" in report


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
