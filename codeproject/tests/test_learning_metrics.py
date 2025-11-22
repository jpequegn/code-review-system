"""
Tests for learning metrics data structures and database schema.

Tests:
- AcceptanceMetrics dataclass validation
- ConfidenceBin calibration metrics
- PatternInfo pattern detection structures
- FixPattern tracking
- Database schema and relationships
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.learning.metrics import (
    AcceptanceMetrics,
    ConfidenceBin,
    PatternInfo,
    FixPattern,
    CalibrationReport,
    PatternReport,
    RankingScores,
)
from src.database import (
    Base,
    ConfidenceCalibration,
    PatternMetrics,
    Finding,
    Review,
    FindingCategory,
    FindingSeverity,
    ReviewStatus,
)


@pytest.fixture
def test_db():
    """
    Create an in-memory SQLite database for testing.

    Provides a fresh database for each test and cleans up afterward.
    """
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")

    # Enable foreign key constraints for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create session factory
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Yield session for test
    session = TestSessionLocal()
    yield session

    # Cleanup
    session.close()


# ============================================================================
# AcceptanceMetrics Tests
# ============================================================================


class TestAcceptanceMetrics:
    """Test AcceptanceMetrics dataclass."""

    def test_create_valid_metrics(self):
        """Create valid acceptance metrics."""
        metrics = AcceptanceMetrics(
            finding_category="SQL Injection",
            severity="CRITICAL",
            total=10,
            accepted=8,
            rejected=2,
            ignored=0,
            acceptance_rate=0.80,
            confidence_avg=0.85,
            fix_rate=0.75,
        )
        assert metrics.finding_category == "SQL Injection"
        assert metrics.acceptance_rate == 0.80
        assert metrics.fix_rate == 0.75

    def test_acceptance_rate_validation(self):
        """Acceptance rate must be 0.0-1.0."""
        with pytest.raises(ValueError, match="acceptance_rate must be 0.0-1.0"):
            AcceptanceMetrics(
                finding_category="Test",
                severity="HIGH",
                total=10,
                accepted=8,
                rejected=2,
                ignored=0,
                acceptance_rate=1.5,  # Invalid
                confidence_avg=0.8,
                fix_rate=0.7,
            )

    def test_confidence_avg_validation(self):
        """Confidence average must be 0.0-1.0."""
        with pytest.raises(ValueError, match="confidence_avg must be 0.0-1.0"):
            AcceptanceMetrics(
                finding_category="Test",
                severity="HIGH",
                total=10,
                accepted=8,
                rejected=2,
                ignored=0,
                acceptance_rate=0.8,
                confidence_avg=-0.1,  # Invalid
                fix_rate=0.7,
            )

    def test_fix_rate_validation(self):
        """Fix rate must be 0.0-1.0."""
        with pytest.raises(ValueError, match="fix_rate must be 0.0-1.0"):
            AcceptanceMetrics(
                finding_category="Test",
                severity="HIGH",
                total=10,
                accepted=8,
                rejected=2,
                ignored=0,
                acceptance_rate=0.8,
                confidence_avg=0.8,
                fix_rate=2.0,  # Invalid
            )

    def test_boundary_values(self):
        """Test boundary values (0.0 and 1.0)."""
        metrics = AcceptanceMetrics(
            finding_category="Test",
            severity="LOW",
            total=10,
            accepted=10,
            rejected=0,
            ignored=0,
            acceptance_rate=1.0,  # Edge case
            confidence_avg=0.0,  # Edge case
            fix_rate=0.5,
        )
        assert metrics.acceptance_rate == 1.0
        assert metrics.confidence_avg == 0.0


# ============================================================================
# ConfidenceBin Tests
# ============================================================================


class TestConfidenceBin:
    """Test ConfidenceBin calibration structure."""

    def test_create_valid_bin(self):
        """Create valid confidence bin."""
        bin_data = ConfidenceBin(
            bin_id=5,
            confidence_range="0.5-0.6",
            original_acceptance_rate=0.55,
            actual_acceptance_rate=0.60,
            precision=0.75,
            recall=0.70,
            f1_score=0.72,
            calibrated_threshold=0.60,
            sample_size=100,
        )
        assert bin_data.bin_id == 5
        assert bin_data.f1_score == 0.72
        assert bin_data.sample_size == 100

    def test_bin_id_validation(self):
        """Bin ID must be 0-9."""
        with pytest.raises(ValueError, match="bin_id must be 0-9"):
            ConfidenceBin(
                bin_id=15,  # Invalid
                confidence_range="0.5-0.6",
                original_acceptance_rate=0.55,
                actual_acceptance_rate=0.60,
                precision=0.75,
                recall=0.70,
                f1_score=0.72,
                calibrated_threshold=0.60,
                sample_size=100,
            )

    def test_precision_validation(self):
        """Precision must be 0.0-1.0."""
        with pytest.raises(ValueError, match="precision must be 0.0-1.0"):
            ConfidenceBin(
                bin_id=5,
                confidence_range="0.5-0.6",
                original_acceptance_rate=0.55,
                actual_acceptance_rate=0.60,
                precision=1.5,  # Invalid
                recall=0.70,
                f1_score=0.72,
                calibrated_threshold=0.60,
                sample_size=100,
            )

    def test_all_metrics_validation(self):
        """All metric fields validated."""
        with pytest.raises(ValueError, match="recall must be 0.0-1.0"):
            ConfidenceBin(
                bin_id=5,
                confidence_range="0.5-0.6",
                original_acceptance_rate=0.55,
                actual_acceptance_rate=0.60,
                precision=0.75,
                recall=-0.1,  # Invalid
                f1_score=0.72,
                calibrated_threshold=0.60,
                sample_size=100,
            )

    def test_all_bins_valid(self):
        """All 10 bins can be created."""
        for bin_id in range(10):
            bin_data = ConfidenceBin(
                bin_id=bin_id,
                confidence_range=f"0.{bin_id}-0.{bin_id+1}",
                original_acceptance_rate=0.5,
                actual_acceptance_rate=0.5 + (bin_id * 0.05),
                precision=0.7,
                recall=0.7,
                f1_score=0.7,
                calibrated_threshold=0.5,
                sample_size=10,
            )
            assert bin_data.bin_id == bin_id


# ============================================================================
# PatternInfo Tests
# ============================================================================


class TestPatternInfo:
    """Test PatternInfo pattern detection structure."""

    def test_create_valid_pattern(self):
        """Create valid pattern info."""
        pattern = PatternInfo(
            pattern_type="unclosed_file_handle",
            occurrences=12,
            files=["src/db.py", "src/api.py"],
            avg_severity=0.85,
            acceptance_rate=0.90,
            fix_count=10,
            anti_pattern=False,
            team_prevalence="common",
            recommended_fix="Use context managers: with open(...) as f:",
        )
        assert pattern.pattern_type == "unclosed_file_handle"
        assert pattern.occurrences == 12
        assert pattern.team_prevalence == "common"

    def test_occurrences_validation(self):
        """Occurrences must be >= 1."""
        with pytest.raises(ValueError, match="occurrences must be >= 1"):
            PatternInfo(
                pattern_type="test",
                occurrences=0,  # Invalid
                files=["test.py"],
                avg_severity=0.5,
                acceptance_rate=0.5,
                fix_count=0,
                anti_pattern=False,
                team_prevalence="rare",
                recommended_fix="test",
            )

    def test_prevalence_validation(self):
        """Team prevalence must be rare/occasional/common."""
        with pytest.raises(ValueError, match="team_prevalence must be"):
            PatternInfo(
                pattern_type="test",
                occurrences=5,
                files=["test.py"],
                avg_severity=0.5,
                acceptance_rate=0.5,
                fix_count=0,
                anti_pattern=False,
                team_prevalence="frequent",  # Invalid
                recommended_fix="test",
            )

    def test_empty_files_validation(self):
        """Files list cannot be empty."""
        with pytest.raises(ValueError, match="files list cannot be empty"):
            PatternInfo(
                pattern_type="test",
                occurrences=5,
                files=[],  # Invalid
                avg_severity=0.5,
                acceptance_rate=0.5,
                fix_count=0,
                anti_pattern=False,
                team_prevalence="occasional",
                recommended_fix="test",
            )

    def test_empty_recommended_fix_validation(self):
        """Recommended fix cannot be empty."""
        with pytest.raises(ValueError, match="recommended_fix cannot be empty"):
            PatternInfo(
                pattern_type="test",
                occurrences=5,
                files=["test.py"],
                avg_severity=0.5,
                acceptance_rate=0.5,
                fix_count=0,
                anti_pattern=False,
                team_prevalence="occasional",
                recommended_fix="",  # Invalid
            )

    def test_prevalence_levels(self):
        """All three prevalence levels valid."""
        for prevalence in ['rare', 'occasional', 'common']:
            pattern = PatternInfo(
                pattern_type="test",
                occurrences=5,
                files=["test.py"],
                avg_severity=0.5,
                acceptance_rate=0.5,
                fix_count=0,
                anti_pattern=False,
                team_prevalence=prevalence,
                recommended_fix="test",
            )
            assert pattern.team_prevalence == prevalence


# ============================================================================
# FixPattern Tests
# ============================================================================


class TestFixPattern:
    """Test FixPattern tracking."""

    def test_create_valid_fix_pattern(self):
        """Create valid fix pattern."""
        pattern = FixPattern(
            suggestion_fix="Use parameterized queries: cursor.execute(..., (user_id,))",
            acceptance_count=8,
            rejection_count=2,
            fix_time_avg_hours=2.5,
            effectiveness=0.80,
        )
        assert pattern.acceptance_count == 8
        assert pattern.effectiveness == 0.80

    def test_effectiveness_calculation(self):
        """Effectiveness should match acceptance rate."""
        # 8 accepted, 2 rejected → 80% effectiveness
        pattern = FixPattern(
            suggestion_fix="test",
            acceptance_count=8,
            rejection_count=2,
            fix_time_avg_hours=2.5,
            effectiveness=0.80,
        )
        assert pattern.effectiveness == 0.80

    def test_negative_counts_validation(self):
        """Counts must be non-negative."""
        with pytest.raises(ValueError, match="acceptance_count must be >= 0"):
            FixPattern(
                suggestion_fix="test",
                acceptance_count=-1,  # Invalid
                rejection_count=2,
                fix_time_avg_hours=2.5,
                effectiveness=0.80,
            )


# ============================================================================
# CalibrationReport Tests
# ============================================================================


class TestCalibrationReport:
    """Test calibration report structure."""

    def test_create_valid_report(self):
        """Create valid calibration report."""
        bins = [
            ConfidenceBin(
                bin_id=i,
                confidence_range=f"0.{i}-0.{i+1}",
                original_acceptance_rate=0.5,
                actual_acceptance_rate=0.5,
                precision=0.7,
                recall=0.7,
                f1_score=0.7,
                calibrated_threshold=0.5,
                sample_size=10,
            )
            for i in range(10)
        ]

        report = CalibrationReport(
            bins=bins,
            recommended_thresholds={
                'aggressive': 0.60,
                'balanced': 0.75,
                'conservative': 0.85,
            },
            improvement_summary="Calibration reduces false positives by 15%",
        )
        assert len(report.bins) == 10
        assert report.recommended_thresholds['balanced'] == 0.75

    def test_must_have_10_bins(self):
        """Report must have exactly 10 bins."""
        bins = [
            ConfidenceBin(
                bin_id=0,
                confidence_range="0.0-0.1",
                original_acceptance_rate=0.5,
                actual_acceptance_rate=0.5,
                precision=0.7,
                recall=0.7,
                f1_score=0.7,
                calibrated_threshold=0.5,
                sample_size=10,
            )
        ]

        with pytest.raises(ValueError, match="Must have exactly 10 bins"):
            CalibrationReport(
                bins=bins,  # Only 1 bin
                recommended_thresholds={
                    'aggressive': 0.60,
                    'balanced': 0.75,
                    'conservative': 0.85,
                },
                improvement_summary="test",
            )

    def test_must_have_all_thresholds(self):
        """Report must have all three threshold types."""
        bins = [
            ConfidenceBin(
                bin_id=i,
                confidence_range=f"0.{i}-0.{i+1}",
                original_acceptance_rate=0.5,
                actual_acceptance_rate=0.5,
                precision=0.7,
                recall=0.7,
                f1_score=0.7,
                calibrated_threshold=0.5,
                sample_size=10,
            )
            for i in range(10)
        ]

        with pytest.raises(ValueError, match="Must have thresholds for"):
            CalibrationReport(
                bins=bins,
                recommended_thresholds={
                    'balanced': 0.75,  # Missing aggressive and conservative
                },
                improvement_summary="test",
            )


# ============================================================================
# PatternReport Tests
# ============================================================================


class TestPatternReport:
    """Test pattern report structure."""

    def test_create_valid_report(self):
        """Create valid pattern report."""
        report = PatternReport(
            total_patterns=23,
            anti_patterns=5,
            best_practices=8,
            common_patterns=[
                {
                    'type': 'SQL Injection',
                    'occurrences': 15,
                    'acceptance_rate': 0.85,
                }
            ],
            files_affected={'src/db.py': 12, 'src/api.py': 8},
        )
        assert report.total_patterns == 23
        assert report.anti_patterns == 5

    def test_anti_plus_best_cannot_exceed_total(self):
        """Anti-patterns + best practices cannot exceed total."""
        with pytest.raises(ValueError, match="cannot exceed total_patterns"):
            PatternReport(
                total_patterns=10,
                anti_patterns=8,
                best_practices=5,  # 8 + 5 = 13 > 10
                common_patterns=[],
                files_affected={},
            )


# ============================================================================
# RankingScores Tests
# ============================================================================


class TestRankingScores:
    """Test ranking scores breakdown."""

    def test_create_valid_scores(self):
        """Create valid ranking scores."""
        scores = RankingScores(
            confidence=0.85,
            acceptance_rate=0.80,
            impact_score=0.90,
            fix_time=0.75,
            team_preference=0.70,
            total_score=0.80,
        )
        assert scores.confidence == 0.85
        assert scores.total_score == 0.80

    def test_all_scores_must_be_0_to_1(self):
        """All scores must be 0.0-1.0."""
        with pytest.raises(ValueError, match="confidence must be 0.0-1.0"):
            RankingScores(
                confidence=1.5,  # Invalid
                acceptance_rate=0.80,
                impact_score=0.90,
                fix_time=0.75,
                team_preference=0.70,
                total_score=0.80,
            )


# ============================================================================
# Database Schema Tests
# ============================================================================


class TestConfidenceCalibrationSchema:
    """Test ConfidenceCalibration database table."""

    def test_create_calibration_record(self, test_db: Session):
        """Create a confidence calibration record."""
        record = ConfidenceCalibration(
            bin_id=5,
            original_confidence_range="0.5-0.6",
            sample_size=100,
            original_acceptance_rate=0.55,
            actual_acceptance_rate=0.60,
            precision=0.75,
            recall=0.70,
            f1_score=0.72,
            calibrated_threshold=0.60,
        )
        test_db.add(record)
        test_db.commit()

        retrieved = test_db.query(ConfidenceCalibration).filter(
            ConfidenceCalibration.bin_id == 5
        ).first()
        assert retrieved is not None
        assert retrieved.f1_score == 0.72
        assert retrieved.sample_size == 100

    def test_bin_id_unique_constraint(self, test_db: Session):
        """Bin ID must be unique."""
        record1 = ConfidenceCalibration(
            bin_id=3,
            original_confidence_range="0.3-0.4",
            sample_size=50,
            original_acceptance_rate=0.50,
            actual_acceptance_rate=0.55,
            precision=0.70,
            recall=0.65,
            f1_score=0.67,
            calibrated_threshold=0.55,
        )
        test_db.add(record1)
        test_db.commit()

        # Try to add another with same bin_id
        record2 = ConfidenceCalibration(
            bin_id=3,  # Duplicate
            original_confidence_range="0.3-0.4",
            sample_size=60,
            original_acceptance_rate=0.50,
            actual_acceptance_rate=0.55,
            precision=0.70,
            recall=0.65,
            f1_score=0.67,
            calibrated_threshold=0.55,
        )
        test_db.add(record2)

        with pytest.raises(Exception):  # IntegrityError or similar
            test_db.commit()

    def test_calibration_timestamps(self, test_db: Session):
        """Calibration record has proper timestamps."""
        record = ConfidenceCalibration(
            bin_id=7,
            original_confidence_range="0.7-0.8",
            sample_size=100,
            original_acceptance_rate=0.70,
            actual_acceptance_rate=0.75,
            precision=0.80,
            recall=0.75,
            f1_score=0.77,
            calibrated_threshold=0.75,
        )
        test_db.add(record)
        test_db.commit()

        assert record.created_at is not None
        assert record.last_updated is not None
        assert isinstance(record.created_at, datetime)


class TestPatternMetricsSchema:
    """Test PatternMetrics database table."""

    def test_create_pattern_record(self, test_db: Session):
        """Create a pattern metrics record."""
        import json

        record = PatternMetrics(
            pattern_hash="abc123def456",
            pattern_type="unclosed_file_handle",
            occurrences=12,
            files_affected=json.dumps(["src/db.py", "src/api.py"]),
            avg_severity=0.85,
            acceptance_rate=0.90,
            fix_count=10,
            anti_pattern=False,
            team_prevalence="common",
            recommended_fix="Use context managers: with open(...) as f:",
        )
        test_db.add(record)
        test_db.commit()

        retrieved = test_db.query(PatternMetrics).filter(
            PatternMetrics.pattern_type == "unclosed_file_handle"
        ).first()
        assert retrieved is not None
        assert retrieved.occurrences == 12
        assert retrieved.acceptance_rate == 0.90

    def test_pattern_hash_unique_constraint(self, test_db: Session):
        """Pattern hash must be unique."""
        import json

        record1 = PatternMetrics(
            pattern_hash="hash123",
            pattern_type="sql_injection",
            occurrences=5,
            files_affected=json.dumps(["src/api.py"]),
            avg_severity=0.9,
            acceptance_rate=0.8,
            fix_count=4,
            anti_pattern=False,
            team_prevalence="occasional",
            recommended_fix="Use parameterized queries",
        )
        test_db.add(record1)
        test_db.commit()

        # Try to add another with same hash
        record2 = PatternMetrics(
            pattern_hash="hash123",  # Duplicate
            pattern_type="sql_injection",
            occurrences=6,
            files_affected=json.dumps(["src/api.py"]),
            avg_severity=0.9,
            acceptance_rate=0.8,
            fix_count=5,
            anti_pattern=False,
            team_prevalence="occasional",
            recommended_fix="Use parameterized queries",
        )
        test_db.add(record2)

        with pytest.raises(Exception):  # IntegrityError or similar
            test_db.commit()

    def test_pattern_metrics_repr(self, test_db: Session):
        """Pattern metrics has proper string representation."""
        import json

        record = PatternMetrics(
            pattern_hash="xyz789",
            pattern_type="n_plus_one_query",
            occurrences=8,
            files_affected=json.dumps(["src/db.py"]),
            avg_severity=0.75,
            acceptance_rate=0.85,
            fix_count=6,
            anti_pattern=True,
            team_prevalence="occasional",
            recommended_fix="Batch load relationships",
        )
        test_db.add(record)
        test_db.commit()

        repr_str = repr(record)
        assert "n_plus_one_query" in repr_str
        assert "8x" in repr_str
        assert "85" in repr_str  # 0.85 as percentage


# ============================================================================
# Integration Tests
# ============================================================================


class TestSchemaIntegration:
    """Test schema integration and relationships."""

    def test_all_tables_created(self, test_db: Session):
        """All required tables exist in database."""
        # Get all table names from metadata
        table_names = {table.name for table in Base.metadata.sorted_tables}

        required_tables = {
            'reviews',
            'findings',
            'suggestion_feedback',
            'suggestion_impact',
            'confidence_calibration',
            'pattern_metrics',
        }

        for table in required_tables:
            assert table in table_names, f"Table '{table}' not found in schema"

    def test_database_initialization(self, test_db: Session):
        """Database can be initialized without errors."""
        # Tables should already be created by the test_db fixture
        # Just verify we can query them
        assert test_db.query(ConfidenceCalibration).count() == 0
        assert test_db.query(PatternMetrics).count() == 0
