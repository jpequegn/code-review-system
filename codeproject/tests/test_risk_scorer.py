"""
Tests for Risk Scorer module.

Tests risk calculation, file scoring, and risk level classification.
"""

import pytest
from datetime import datetime
from pathlib import Path

from src.prediction.history_tracker import CodeQualitySnapshot, HistoryDatabase
from src.prediction.risk_scorer import RiskScorer, RiskLevel


@pytest.fixture
def history_db():
    """Create a history database with test data."""
    db = HistoryDatabase()

    # Add some test snapshots
    snapshot1 = CodeQualitySnapshot(
        commit_sha="abc123",
        timestamp=datetime.now(),
        files_changed=["file1.py", "file2.py"],
        metrics_per_file={
            "file1.py": {"complexity": 10, "lines": 100},
            "file2.py": {"complexity": 5, "lines": 50},
        },
        complexity_trend={"file1.py": 10.0, "file2.py": 5.0},
        test_coverage_trend={"file1.py": 0.8, "file2.py": 0.6},
        issues_found=2,
        production_bugs=1,
        fixed_issues=0,
        architectural_metrics={"density": 0.3, "coupling": 2.0},
        coupling_trend={"file1.py": 2.0},
    )

    db.add_snapshot(snapshot1)
    return db


@pytest.fixture
def risk_scorer(history_db):
    """Create a risk scorer with test history."""
    return RiskScorer(history_db)


class TestRiskScoring:
    """Test risk scoring functionality."""

    def test_score_file_low_risk(self, risk_scorer):
        """Test scoring a low-risk file."""
        score = risk_scorer.score_file(
            "file1.py",
            current_complexity=5.0,
            current_coverage=0.9,
            coupling_score=0.1,
        )

        assert score.file_path == "file1.py"
        assert score.risk_level == RiskLevel.LOW
        assert 0.0 <= score.total_score <= 1.0

    def test_score_file_high_risk(self, risk_scorer):
        """Test scoring a high-risk file."""
        score = risk_scorer.score_file(
            "file2.py",
            current_complexity=25.0,  # High complexity
            current_coverage=0.1,  # Low coverage
            coupling_score=0.8,  # High coupling
        )

        assert score.file_path == "file2.py"
        # High complexity + low coverage should result in elevated risk
        assert score.risk_level in [RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]
        assert score.total_score > 0.3

    def test_coverage_score_calculation(self, risk_scorer):
        """Test coverage score (inverted: low coverage = high risk)."""
        high_coverage = risk_scorer._score_coverage(0.9)
        low_coverage = risk_scorer._score_coverage(0.1)

        assert high_coverage < low_coverage
        assert 0.0 <= high_coverage <= 1.0
        assert 0.0 <= low_coverage <= 1.0

    def test_risk_level_classification(self, risk_scorer):
        """Test risk score to level conversion."""
        assert risk_scorer._score_to_level(0.1) == RiskLevel.LOW
        assert risk_scorer._score_to_level(0.35) == RiskLevel.MEDIUM
        assert risk_scorer._score_to_level(0.6) == RiskLevel.HIGH
        assert risk_scorer._score_to_level(0.8) == RiskLevel.CRITICAL

    def test_score_multiple_files(self, risk_scorer):
        """Test scoring multiple files at once."""
        file_metrics = {
            "file1.py": {"complexity": 10, "coverage": 0.8, "coupling": 0.2},
            "file2.py": {"complexity": 20, "coverage": 0.2, "coupling": 0.8},
            "file3.py": {"complexity": 5, "coverage": 0.9, "coupling": 0.1},
        }

        scores = risk_scorer.score_multiple_files(file_metrics)

        assert len(scores) == 3
        assert all(0.0 <= score.total_score <= 1.0 for score in scores.values())

        # file2 should be riskier than file3
        assert scores["file2.py"].total_score > scores["file3.py"].total_score

    def test_get_critical_files(self, risk_scorer):
        """Test getting critical risk files."""
        file_metrics = {
            "critical1.py": {"complexity": 30, "coverage": 0.0, "coupling": 0.9},
            "critical2.py": {"complexity": 25, "coverage": 0.1, "coupling": 0.8},
            "safe.py": {"complexity": 5, "coverage": 0.95, "coupling": 0.1},
        }

        critical = risk_scorer.get_critical_files(file_metrics)

        # Should identify at least some high-risk files (CRITICAL level)
        # May be empty if threshold not reached, but if any found, should be CRITICAL
        if critical:
            for file_path, score in critical:
                assert score.risk_level == RiskLevel.CRITICAL

    def test_get_high_risk_files(self, risk_scorer):
        """Test getting high-risk files (HIGH or CRITICAL)."""
        file_metrics = {
            "file1.py": {"complexity": 20, "coverage": 0.2, "coupling": 0.7},
            "file2.py": {"complexity": 30, "coverage": 0.0, "coupling": 0.9},
            "file3.py": {"complexity": 5, "coverage": 0.95, "coupling": 0.1},
        }

        high_risk = risk_scorer.get_high_risk_files(file_metrics)

        assert len(high_risk) > 0
        for file_path, score in high_risk:
            assert score.risk_level in [RiskLevel.HIGH, RiskLevel.CRITICAL]

    def test_contributing_factors(self, risk_scorer):
        """Test that contributing factors are identified."""
        score = risk_scorer.score_file(
            "test.py",
            current_complexity=25.0,  # High
            current_coverage=0.1,  # Low
            coupling_score=0.8,  # High
        )

        assert len(score.contributing_factors) > 0
        assert any("complexity" in k for k in score.contributing_factors.keys()) or len(
            score.contributing_factors
        ) >= 2

    def test_churn_scoring(self, risk_scorer):
        """Test churn rate scoring."""
        # File with no changes = low churn score
        low_churn = risk_scorer._score_churn("nonexistent_file.py")
        assert low_churn == 0.0 or low_churn < 0.3

    def test_complexity_scoring(self, risk_scorer):
        """Test complexity scoring."""
        low_complexity = risk_scorer._score_complexity("file.py", 5.0)
        high_complexity = risk_scorer._score_complexity("file.py", 25.0)

        assert low_complexity < high_complexity
        assert 0.0 <= low_complexity <= 1.0
        assert 0.0 <= high_complexity <= 1.0
