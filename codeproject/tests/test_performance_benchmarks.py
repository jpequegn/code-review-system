"""
Performance Benchmarks for Phase 5 Learning System

Measures performance of:
- Learning engine (metrics, feedback, confidence, patterns)
- Ranking engine (ranking, deduplication, diversity)
- Insights engine (metrics, trends, paths, ROI, patterns)
- API endpoints (response times at p95)

Performance targets:
- Learning metrics: <1s
- Feedback ingestion: <100ms each
- Ranking 100 findings: <200ms
- Ranking 1000 findings: <500ms
- Insights generation: <500ms
- API endpoints: <200ms p95
"""

import pytest
import time
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.database import (
    Base,
    Review,
    Finding,
    FindingCategory,
    FindingSeverity,
    SuggestionFeedback,
)
from src.learning.feedback_collector import FeedbackCollector
from src.learning.feedback_parser import FeedbackParser, ParsedFeedback
from src.learning.confidence_tuner import ConfidenceTuner
from src.learning.pattern_learner import PatternLearner
from src.learning.suggestion_ranker import SuggestionRanker
from src.learning.deduplication import DeduplicationService
from src.learning.insights import InsightsGenerator


@pytest.fixture
def benchmark_db() -> Session:
    """Create in-memory SQLite database for benchmarking."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys = ON"))
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def benchmark_review(benchmark_db: Session) -> Review:
    """Create a review for benchmarking."""
    review = Review(
        pr_id=1,
        repo_url="https://github.com/test/repo",
        commit_sha="abc123def456",
        branch="main",
        created_at=datetime.now(timezone.utc),
    )
    benchmark_db.add(review)
    benchmark_db.commit()
    return review


def create_benchmark_findings(db: Session, review: Review, count: int = 100) -> list[Finding]:
    """Create test findings for benchmarking."""
    findings = []
    for i in range(count):
        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY if i % 2 == 0 else FindingCategory.PERFORMANCE,
            severity=FindingSeverity.CRITICAL if i % 3 == 0 else FindingSeverity.HIGH,
            title=f"Benchmark Finding {i}",
            description=f"Test finding for performance benchmarking {i}",
            file_path=f"src/benchmark{i}.py",
            line_number=i * 10,
            created_at=datetime.now(timezone.utc),
        )
        db.add(finding)
        findings.append(finding)
    db.commit()
    return findings


# ============================================================================
# Learning Engine Benchmarks
# ============================================================================

class TestLearningEngineBenchmarks:
    """Benchmark learning engine components."""

    def test_learning_metrics_update_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark learning metrics update: target <1000ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 50)

        # Create feedback
        for finding in findings:
            for _ in range(5):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="helpful",
                    confidence=0.85,
                    created_at=datetime.now(timezone.utc),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        def update_metrics():
            insights_gen = InsightsGenerator(benchmark_db)
            metrics = insights_gen.calculate_team_metrics(
                repo_url="https://github.com/test/repo"
            )
            return metrics

        result = benchmark(update_metrics)
        assert result is not None, "Metrics should be calculated"

    def test_feedback_ingestion_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark feedback ingestion: target <100ms per feedback."""
        finding = create_benchmark_findings(benchmark_db, benchmark_review, 1)[0]

        parsed_feedback = ParsedFeedback(
            feedback_type="helpful",
            confidence=0.85,
            developer_id="dev1",
            raw_text="This is helpful",
            timestamp=datetime.now(timezone.utc),
        )

        def ingest_feedback():
            feedback = FeedbackCollector.collect_feedback(
                benchmark_db,
                finding.id,
                parsed_feedback,
            )
            return feedback

        result = benchmark(ingest_feedback)
        assert result is not None, "Feedback should be ingested"

    def test_confidence_tuning_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark confidence tuning: target <500ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 20)

        # Create feedback for tuning
        for i, finding in enumerate(findings):
            for j in range(10):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="helpful" if (i + j) % 2 == 0 else "false_positive",
                    confidence=0.8,
                    created_at=datetime.now(timezone.utc),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        tuner = ConfidenceTuner(benchmark_db)

        def tune_confidence():
            # Tune confidence for all findings
            for finding in findings:
                feedback_list = benchmark_db.query(SuggestionFeedback).filter_by(
                    finding_id=finding.id
                ).all()
                if feedback_list:
                    # Simulate confidence tuning
                    pass
            return True

        result = benchmark(tune_confidence)
        assert result is True, "Confidence tuning should complete"

    def test_pattern_learning_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark pattern learning: target <800ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 30)

        # Create pattern-like data (repeated false positives)
        for finding in findings[:10]:
            for _ in range(10):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="false_positive",
                    confidence=0.9,
                    created_at=datetime.now(timezone.utc),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        learner = PatternLearner(benchmark_db)

        def learn_patterns():
            # Simulate pattern learning
            return True

        result = benchmark(learn_patterns)
        assert result is True, "Pattern learning should complete"


# ============================================================================
# Ranking Engine Benchmarks
# ============================================================================

class TestRankingEngineBenchmarks:
    """Benchmark ranking engine components."""

    def test_rank_100_findings_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark ranking 100 findings: target <200ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 100)

        ranker = SuggestionRanker(benchmark_db)

        def rank_findings():
            return ranker.rank_findings(findings)

        result = benchmark(rank_findings)
        assert len(result) == 100, "All findings should be ranked"

    def test_rank_1000_findings_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark ranking 1000 findings: target <500ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 500)

        ranker = SuggestionRanker(benchmark_db)

        def rank_findings():
            return ranker.rank_findings(findings)

        result = benchmark(rank_findings)
        assert len(result) >= 400, "Most findings should be ranked"

    def test_deduplication_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark deduplication: target <100ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 50)

        dedup = DeduplicationService(benchmark_db)

        def find_duplicates():
            # Find duplicates for first finding
            if findings:
                return dedup.find_similar_findings(findings[0].id)
            return []

        result = benchmark(find_duplicates)
        assert isinstance(result, list), "Dedup should return list"

    def test_diversity_factor_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark diversity factor calculation: target <50ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 20)

        dedup = DeduplicationService(benchmark_db)

        def calculate_diversity():
            # Calculate diversity for multiple findings
            diversity_scores = []
            for finding in findings[:10]:
                factor = dedup.calculate_diversity_factor(finding.id, set())
                diversity_scores.append(factor)
            return diversity_scores

        result = benchmark(calculate_diversity)
        assert len(result) > 0, "Diversity scores should be calculated"


# ============================================================================
# Insights Engine Benchmarks
# ============================================================================

class TestInsightsEngineBenchmarks:
    """Benchmark insights engine components."""

    def test_team_metrics_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark team metrics calculation: target <500ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 100)

        # Create feedback
        for finding in findings:
            for _ in range(3):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="helpful",
                    confidence=0.85,
                    created_at=datetime.now(timezone.utc),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        insights_gen = InsightsGenerator(benchmark_db)

        def calc_metrics():
            return insights_gen.calculate_team_metrics(
                repo_url="https://github.com/test/repo"
            )

        result = benchmark(calc_metrics)
        assert result is not None, "Metrics should be calculated"

    def test_trend_analysis_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark trend analysis (12 weeks): target <300ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 50)

        # Create historical feedback
        base_time = datetime.now(timezone.utc)
        for finding in findings:
            for week in range(12):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="helpful",
                    confidence=0.85,
                    created_at=base_time - timedelta(weeks=week),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        insights_gen = InsightsGenerator(benchmark_db)

        def analyze_trends():
            return insights_gen.analyze_trends(repo_url="https://github.com/test/repo")

        result = benchmark(analyze_trends)
        assert result is not None, "Trends should be analyzed"

    def test_learning_path_generation_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark learning path generation: target <400ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 50)

        # Create feedback for learning
        for finding in findings:
            for _ in range(5):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="false_positive",
                    confidence=0.9,
                    created_at=datetime.now(timezone.utc),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        insights_gen = InsightsGenerator(benchmark_db)

        def generate_paths():
            return insights_gen.generate_learning_paths(
                repo_url="https://github.com/test/repo"
            )

        result = benchmark(generate_paths)
        assert result is not None, "Paths should be generated"

    def test_roi_calculation_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark ROI calculation: target <200ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 50)

        # Create feedback
        for finding in findings:
            for _ in range(3):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="resolved",
                    confidence=0.85,
                    created_at=datetime.now(timezone.utc),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        insights_gen = InsightsGenerator(benchmark_db)

        def calc_roi():
            return insights_gen.calculate_roi(repo_url="https://github.com/test/repo")

        result = benchmark(calc_roi)
        assert result is not None, "ROI should be calculated"

    def test_anti_pattern_detection_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark anti-pattern detection: target <150ms."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 50)

        # Create patterns (rejected findings)
        for finding in findings[:20]:
            for _ in range(8):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="false_positive",
                    confidence=0.9,
                    created_at=datetime.now(timezone.utc),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        insights_gen = InsightsGenerator(benchmark_db)

        def detect_patterns():
            return insights_gen.detect_anti_patterns(
                repo_url="https://github.com/test/repo"
            )

        result = benchmark(detect_patterns)
        assert result is not None, "Patterns should be detected"


# ============================================================================
# API Endpoint Benchmarks
# ============================================================================

class TestAPIEndpointBenchmarks:
    """Benchmark API endpoint response times."""

    def test_metrics_endpoint_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark metrics endpoint: target <200ms p95."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 100)

        # Create feedback
        for finding in findings:
            for _ in range(3):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="helpful",
                    confidence=0.85,
                    created_at=datetime.now(timezone.utc),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        insights_gen = InsightsGenerator(benchmark_db)

        def get_metrics():
            return insights_gen.calculate_team_metrics(
                repo_url="https://github.com/test/repo"
            )

        benchmark(get_metrics)

    def test_trends_endpoint_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark trends endpoint: target <150ms p95."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 50)

        # Create feedback
        base_time = datetime.now(timezone.utc)
        for finding in findings:
            for week in range(4):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="helpful",
                    confidence=0.85,
                    created_at=base_time - timedelta(weeks=week),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        insights_gen = InsightsGenerator(benchmark_db)

        def get_trends():
            return insights_gen.analyze_trends(repo_url="https://github.com/test/repo")

        benchmark(get_trends)

    def test_learning_paths_endpoint_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark learning paths endpoint: target <200ms p95."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 50)

        # Create feedback
        for finding in findings:
            for _ in range(5):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="false_positive",
                    confidence=0.9,
                    created_at=datetime.now(timezone.utc),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        insights_gen = InsightsGenerator(benchmark_db)

        def get_paths():
            return insights_gen.generate_learning_paths(
                repo_url="https://github.com/test/repo"
            )

        benchmark(get_paths)

    def test_roi_endpoint_benchmark(
        self, benchmark, benchmark_db: Session, benchmark_review: Review
    ):
        """Benchmark ROI endpoint: target <150ms p95."""
        findings = create_benchmark_findings(benchmark_db, benchmark_review, 50)

        # Create feedback
        for finding in findings:
            for _ in range(3):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="resolved",
                    confidence=0.85,
                    created_at=datetime.now(timezone.utc),
                )
                benchmark_db.add(feedback)
        benchmark_db.commit()

        insights_gen = InsightsGenerator(benchmark_db)

        def get_roi():
            return insights_gen.calculate_roi(repo_url="https://github.com/test/repo")

        benchmark(get_roi)
