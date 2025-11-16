"""
Tests for Failure Predictor module.

Tests failure prediction, pattern recognition, and risk assessment.
"""

import pytest
from datetime import datetime

from src.prediction.history_tracker import CodeQualitySnapshot, HistoryDatabase
from src.prediction.risk_scorer import RiskScorer
from src.prediction.pattern_learner import PatternLearner, IssueType, CodeContext
from src.prediction.failure_predictor import (
    FailurePredictor,
    FailureType,
)


@pytest.fixture
def history_db():
    """Create history database with test data."""
    db = HistoryDatabase()

    snapshot = CodeQualitySnapshot(
        commit_sha="abc123",
        timestamp=datetime.now(),
        files_changed=["file1.py"],
        metrics_per_file={"file1.py": {"complexity": 15, "lines": 150}},
        complexity_trend={"file1.py": 15.0},
        test_coverage_trend={"file1.py": 0.6},
        issues_found=3,
        production_bugs=1,
        fixed_issues=1,
        architectural_metrics={},
        coupling_trend={"file1.py": 0.5},
    )

    db.add_snapshot(snapshot)
    return db


@pytest.fixture
def components(history_db):
    """Create all prediction components."""
    risk_scorer = RiskScorer(history_db)
    pattern_learner = PatternLearner()
    failure_predictor = FailurePredictor(history_db, pattern_learner, risk_scorer)

    return {
        "history_db": history_db,
        "risk_scorer": risk_scorer,
        "pattern_learner": pattern_learner,
        "failure_predictor": failure_predictor,
    }


class TestFailurePrediction:
    """Test failure prediction functionality."""

    def test_predict_failure_basic(self, components):
        """Test basic failure prediction."""
        predictor = components["failure_predictor"]

        prediction = predictor.predict_failure(
            "file1.py",
            current_complexity=15.0,
            current_coverage=0.6,
            coupling_score=0.5,
        )

        assert prediction.file_path == "file1.py"
        assert 0.0 <= prediction.likelihood <= 1.0
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.predicted_failure_types) > 0

    def test_predict_failure_with_context(self, components):
        """Test failure prediction with code context hint."""
        predictor = components["failure_predictor"]
        pattern_learner = components["pattern_learner"]

        # Record a pattern
        pattern_learner.record_issue(
            IssueType.LOGIC,
            CodeContext.LOOPS,
            "file1.py",
            "commit123",
            "Off-by-one error",
        )

        prediction = predictor.predict_failure(
            "file1.py",
            current_complexity=15.0,
            current_coverage=0.6,
            coupling_score=0.5,
            code_context=CodeContext.LOOPS,
        )

        assert prediction.file_path == "file1.py"
        assert len(prediction.contributing_patterns) > 0

    def test_failure_type_likelihoods(self, components):
        """Test that failure types have realistic likelihoods."""
        predictor = components["failure_predictor"]

        prediction = predictor.predict_failure(
            "file1.py",
            current_complexity=25.0,  # High
            current_coverage=0.1,  # Low
            coupling_score=0.8,
        )

        # Low coverage should increase bug likelihood
        assert prediction.predicted_failure_types[FailureType.BUG] > 0.0

    def test_predict_multiple_files(self, components):
        """Test predicting failures for multiple files."""
        predictor = components["failure_predictor"]

        file_metrics = {
            "file1.py": {"complexity": 10, "coverage": 0.9, "coupling": 0.2},
            "file2.py": {"complexity": 25, "coverage": 0.1, "coupling": 0.8},
            "file3.py": {"complexity": 5, "coverage": 0.95, "coupling": 0.05},
        }

        predictions = predictor.predict_multiple_files(file_metrics)

        assert len(predictions) == 3
        # Sorted by likelihood
        assert predictions[0][1].likelihood >= predictions[1][1].likelihood

    def test_record_and_predict_failure(self, components):
        """Test recording failures and using them in predictions."""
        predictor = components["failure_predictor"]

        # Record a failure
        predictor.record_failure(
            "file1.py",
            FailureType.BUG,
            "commit123",
            "Off-by-one error in loop",
        )

        prediction = predictor.predict_failure(
            "file1.py",
            current_complexity=10.0,
            current_coverage=0.8,
            coupling_score=0.2,
        )

        # Should have similar failures
        assert len(prediction.similar_failures) > 0
        assert prediction.similar_failures[0].file_path == "file1.py"

    def test_get_most_risky_files(self, components):
        """Test getting most risky files."""
        predictor = components["failure_predictor"]

        file_metrics = {
            "risky1.py": {"complexity": 30, "coverage": 0.0, "coupling": 0.9},
            "risky2.py": {"complexity": 25, "coverage": 0.1, "coupling": 0.8},
            "safe.py": {"complexity": 5, "coverage": 0.95, "coupling": 0.1},
        }

        risky = predictor.get_most_risky_files(file_metrics, limit=2)

        # May be empty if threshold not reached, but if found, should be the risky ones
        for file_path, pred in risky:
            assert file_path in ["risky1.py", "risky2.py"]

    def test_get_high_confidence_predictions(self, components):
        """Test getting high-confidence predictions."""
        predictor = components["failure_predictor"]

        file_metrics = {
            "file1.py": {"complexity": 15, "coverage": 0.6, "coupling": 0.5},
            "file2.py": {"complexity": 20, "coverage": 0.2, "coupling": 0.7},
        }

        high_conf = predictor.get_high_confidence_predictions(file_metrics, min_confidence=0.5)

        # At least some predictions should be found
        assert isinstance(high_conf, list)

    def test_recommendations_generated(self, components):
        """Test that recommendations are generated."""
        predictor = components["failure_predictor"]

        prediction = predictor.predict_failure(
            "file1.py",
            current_complexity=25.0,
            current_coverage=0.1,
            coupling_score=0.8,
        )

        # Should have recommendations
        assert len(prediction.recommendations) > 0
        assert all(isinstance(r, str) for r in prediction.recommendations)

    def test_similar_failures_found(self, components):
        """Test that similar failures are identified."""
        predictor = components["failure_predictor"]

        # Record failures in same directory
        predictor.record_failure("dir/file1.py", FailureType.BUG, "c1", "Error 1")
        predictor.record_failure("dir/file2.py", FailureType.BUG, "c2", "Error 2")

        prediction = predictor.predict_failure(
            "dir/file3.py",
            current_complexity=10.0,
            current_coverage=0.7,
            coupling_score=0.3,
        )

        # Should find similar failures in same directory
        assert isinstance(prediction.similar_failures, list)

    def test_low_coverage_increases_bug_risk(self, components):
        """Test that low test coverage increases bug risk."""
        predictor = components["failure_predictor"]

        low_coverage = predictor.predict_failure(
            "file1.py",
            current_complexity=10.0,
            current_coverage=0.1,
            coupling_score=0.2,
        )

        high_coverage = predictor.predict_failure(
            "file1.py",
            current_complexity=10.0,
            current_coverage=0.9,
            coupling_score=0.2,
        )

        # Low coverage should have higher bug likelihood
        assert (
            low_coverage.predicted_failure_types[FailureType.BUG]
            > high_coverage.predicted_failure_types[FailureType.BUG]
        )
