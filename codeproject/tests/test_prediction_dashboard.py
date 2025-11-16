"""
Tests for Prediction Dashboard and Reporting.

Tests report generation, formatting, and recommendations.
"""

import pytest
from datetime import datetime

from src.prediction.history_tracker import CodeQualitySnapshot, HistoryDatabase
from src.prediction.risk_scorer import RiskScorer
from src.prediction.failure_predictor import FailurePredictor
from src.prediction.temporal_analysis import TemporalAnalyzer
from src.prediction.debt_predictor import DebtPredictor
from src.prediction.pattern_learner import PatternLearner, IssueType, CodeContext
from src.reporting.predictions import PredictionReportGenerator


@pytest.fixture
def setup_reporting():
    """Set up complete reporting system."""
    # Create history database
    history_db = HistoryDatabase()

    snapshot = CodeQualitySnapshot(
        commit_sha="abc123",
        timestamp=datetime.now(),
        files_changed=["file1.py", "file2.py"],
        metrics_per_file={
            "file1.py": {"complexity": 15, "lines": 150},
            "file2.py": {"complexity": 8, "lines": 80},
        },
        complexity_trend={"file1.py": 15.0, "file2.py": 8.0},
        test_coverage_trend={"file1.py": 0.6, "file2.py": 0.85},
        issues_found=3,
        production_bugs=1,
        fixed_issues=0,
        architectural_metrics={"density": 0.3},
        coupling_trend={"file1.py": 0.5, "file2.py": 0.2},
    )

    history_db.add_snapshot(snapshot)

    # Create components
    risk_scorer = RiskScorer(history_db)
    pattern_learner = PatternLearner()
    failure_predictor = FailurePredictor(history_db, pattern_learner, risk_scorer)
    temporal_analyzer = TemporalAnalyzer(history_db)
    debt_predictor = DebtPredictor(history_db)

    # Create report generator
    generator = PredictionReportGenerator(
        history_db,
        risk_scorer,
        failure_predictor,
        temporal_analyzer,
        debt_predictor,
        pattern_learner,
    )

    return {
        "generator": generator,
        "history_db": history_db,
        "risk_scorer": risk_scorer,
        "pattern_learner": pattern_learner,
        "debt_predictor": debt_predictor,
    }


class TestPredictionReporting:
    """Test prediction report generation."""

    def test_generate_weekly_report(self, setup_reporting):
        """Test generating a weekly report."""
        generator = setup_reporting["generator"]

        file_metrics = {
            "file1.py": {"complexity": 15, "coverage": 0.6, "coupling": 0.5},
            "file2.py": {"complexity": 8, "coverage": 0.85, "coupling": 0.2},
        }

        report = generator.generate_weekly_report(file_metrics)

        assert report.report_type == "weekly"
        assert "Week of" in report.report_period
        assert report.generated_at is not None
        assert isinstance(report.at_risk_files, list)
        assert isinstance(report.recommendations, list)

    def test_generate_monthly_report(self, setup_reporting):
        """Test generating a monthly report."""
        generator = setup_reporting["generator"]

        file_metrics = {
            "file1.py": {"complexity": 15, "coverage": 0.6, "coupling": 0.5},
            "file2.py": {"complexity": 8, "coverage": 0.85, "coupling": 0.2},
        }

        report = generator.generate_monthly_report(file_metrics)

        assert report.report_type == "monthly"
        assert report.report_period  # Should have month/year
        assert report.generated_at is not None
        assert isinstance(report.at_risk_files, list)
        assert isinstance(report.failure_predictions, list)

    def test_report_includes_debt_metrics(self, setup_reporting):
        """Test that reports include debt metrics."""
        generator = setup_reporting["generator"]

        file_metrics = {
            "file1.py": {"complexity": 15, "coverage": 0.6, "coupling": 0.5},
        }

        report = generator.generate_weekly_report(file_metrics)

        assert report.debt_metrics is not None
        assert 0.0 <= report.debt_metrics.total_debt_score <= 1.0
        assert report.debt_trajectory is not None

    def test_report_includes_patterns(self, setup_reporting):
        """Test that reports include learned patterns."""
        generator = setup_reporting["generator"]
        pattern_learner = setup_reporting["pattern_learner"]

        # Record a pattern
        pattern_learner.record_issue(
            IssueType.LOGIC,
            CodeContext.LOOPS,
            "file1.py",
            "commit1",
            "Off-by-one error",
        )

        file_metrics = {
            "file1.py": {"complexity": 15, "coverage": 0.6, "coupling": 0.5},
        }

        report = generator.generate_weekly_report(file_metrics)

        assert len(report.personal_patterns) > 0

    def test_report_includes_recommendations(self, setup_reporting):
        """Test that reports include actionable recommendations."""
        generator = setup_reporting["generator"]

        file_metrics = {
            "risky.py": {"complexity": 25, "coverage": 0.1, "coupling": 0.8},
        }

        report = generator.generate_weekly_report(file_metrics)

        assert len(report.recommendations) > 0
        assert all(isinstance(r, str) for r in report.recommendations)

    def test_format_report_as_text(self, setup_reporting):
        """Test formatting report as readable text."""
        generator = setup_reporting["generator"]

        file_metrics = {
            "file1.py": {"complexity": 15, "coverage": 0.6, "coupling": 0.5},
            "file2.py": {"complexity": 8, "coverage": 0.85, "coupling": 0.2},
        }

        report = generator.generate_weekly_report(file_metrics)
        text = generator.format_report_as_text(report)

        # Check that text report includes key sections
        assert "PREDICTION REPORT" in text
        assert "FILES AT RISK" in text
        assert "TECHNICAL DEBT" in text
        assert isinstance(text, str)
        assert len(text) > 100

    def test_report_has_temporal_insights(self, setup_reporting):
        """Test that reports include temporal insights."""
        generator = setup_reporting["generator"]

        file_metrics = {
            "file1.py": {"complexity": 15, "coverage": 0.6, "coupling": 0.5},
        }

        report = generator.generate_weekly_report(file_metrics)

        # Should have temporal insights (even if placeholder)
        assert isinstance(report.temporal_insights, list)

    def test_report_includes_at_risk_files(self, setup_reporting):
        """Test that high-risk files are included in reports."""
        generator = setup_reporting["generator"]

        file_metrics = {
            "risky1.py": {"complexity": 30, "coverage": 0.0, "coupling": 0.9},
            "risky2.py": {"complexity": 25, "coverage": 0.1, "coupling": 0.8},
            "safe.py": {"complexity": 5, "coverage": 0.95, "coupling": 0.1},
        }

        report = generator.generate_weekly_report(file_metrics)

        # At-risk files should be detected
        at_risk_paths = [f[0] for f in report.at_risk_files]
        # If any files are included, should be the risky ones
        for path in at_risk_paths:
            assert path in ["risky1.py", "risky2.py"]

    def test_report_includes_failure_predictions(self, setup_reporting):
        """Test that failure predictions are included."""
        generator = setup_reporting["generator"]

        file_metrics = {
            "file1.py": {"complexity": 15, "coverage": 0.6, "coupling": 0.5},
        }

        report = generator.generate_weekly_report(file_metrics)

        assert isinstance(report.failure_predictions, list)

    def test_debt_trajectory_in_report(self, setup_reporting):
        """Test that debt trajectory is included."""
        generator = setup_reporting["generator"]

        file_metrics = {
            "file1.py": {"complexity": 15, "coverage": 0.6, "coupling": 0.5},
        }

        report = generator.generate_weekly_report(file_metrics)

        assert report.debt_trajectory is not None
        assert "dates" in report.debt_trajectory
        assert "total_debt" in report.debt_trajectory
        assert len(report.debt_trajectory["dates"]) > 0

    def test_report_text_formatting_valid(self, setup_reporting):
        """Test that formatted report text is valid."""
        generator = setup_reporting["generator"]

        file_metrics = {
            "file1.py": {"complexity": 15, "coverage": 0.6, "coupling": 0.5},
        }

        report = generator.generate_weekly_report(file_metrics)
        text = generator.format_report_as_text(report)

        # Should not have placeholder values
        assert "None" not in text
        assert len(text.split("\n")) > 10  # Substantial report

    def test_monthly_report_more_detailed(self, setup_reporting):
        """Test that monthly reports are more detailed than weekly."""
        generator = setup_reporting["generator"]

        file_metrics = {
            "file1.py": {"complexity": 15, "coverage": 0.6, "coupling": 0.5},
            "file2.py": {"complexity": 8, "coverage": 0.85, "coupling": 0.2},
        }

        weekly = generator.generate_weekly_report(file_metrics)
        monthly = generator.generate_monthly_report(file_metrics)

        # Monthly should have more predictions
        assert len(monthly.failure_predictions) >= len(weekly.failure_predictions)
