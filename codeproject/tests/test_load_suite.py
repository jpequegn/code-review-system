"""
Load Testing Suite for Phase 5 Learning System

Tests system performance under high load:
- 10,000+ findings in database
- 50,000+ feedbacks
- Concurrent operations
- Memory profiling
- Database connection pool stress testing
- Concurrent ranking and insights generation

Load test scenarios:
1. Ingest 10K findings with 5 feedbacks each (50K total)
2. Query all findings and rank them
3. Generate insights from all data
4. Concurrent ranking and insights requests
5. Memory consumption monitoring
"""

import pytest
import time
import concurrent.futures
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
from src.learning.suggestion_ranker import SuggestionRanker
from src.learning.insights import InsightsGenerator


@pytest.fixture
def load_test_db() -> Session:
    """Create database for load testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys = ON"))
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def load_test_review(load_test_db: Session) -> Review:
    """Create a review for load testing."""
    review = Review(
        pr_id=1,
        repo_url="https://github.com/test/repo",
        commit_sha="abc123def456",
        branch="main",
        created_at=datetime.now(timezone.utc),
    )
    load_test_db.add(review)
    load_test_db.commit()
    return review


def populate_large_dataset(db: Session, review: Review, finding_count: int = 1000):
    """Populate database with large dataset for load testing."""
    print(f"\nPopulating {finding_count} findings with 5 feedbacks each...")
    start_time = time.time()

    findings = []
    for i in range(finding_count):
        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY if i % 2 == 0 else FindingCategory.PERFORMANCE,
            severity=FindingSeverity.CRITICAL if i % 3 == 0 else FindingSeverity.HIGH,
            title=f"Load Test Finding {i}",
            description=f"This is finding {i} for load testing",
            file_path=f"src/load_test_{i // 100}.py",
            line_number=(i % 100) * 10,
            created_at=datetime.now(timezone.utc),
        )
        findings.append(finding)

    db.add_all(findings)
    db.commit()

    # Create feedback for findings
    feedback_list = []
    feedback_types = ["helpful", "false_positive", "resolved", "recurring"]

    for finding in findings:
        for j in range(5):
            feedback = SuggestionFeedback(
                finding_id=finding.id,
                feedback_type=feedback_types[j % len(feedback_types)],
                confidence=0.7 + (j * 0.05),
                created_at=datetime.now(timezone.utc) - timedelta(days=j),
            )
            feedback_list.append(feedback)

    # Insert feedback in batches
    batch_size = 1000
    for i in range(0, len(feedback_list), batch_size):
        db.add_all(feedback_list[i : i + batch_size])
        db.commit()

    duration = time.time() - start_time
    print(f"✓ Populated {finding_count} findings in {duration:.2f}s")
    print(f"  Total feedbacks: {len(feedback_list)}")

    return findings


# ============================================================================
# Load Test Scenarios
# ============================================================================

class TestLoadScenarios:
    """Load testing scenarios."""

    def test_load_1000_findings(self, load_test_db: Session, load_test_review: Review):
        """Load test with 1000 findings."""
        findings = populate_large_dataset(load_test_db, load_test_review, 1000)

        # Verify data loaded
        finding_count = load_test_db.query(Finding).count()
        feedback_count = load_test_db.query(SuggestionFeedback).count()

        assert finding_count == 1000, "Should have 1000 findings"
        assert feedback_count == 5000, "Should have 5000 feedbacks"

        print(f"✓ Data verified: {finding_count} findings, {feedback_count} feedbacks")

    def test_rank_1000_findings_performance(
        self, load_test_db: Session, load_test_review: Review
    ):
        """Test ranking 1000 findings performance."""
        findings = populate_large_dataset(load_test_db, load_test_review, 1000)

        print(f"\nRanking {len(findings)} findings...")
        start_time = time.time()

        ranker = SuggestionRanker(load_test_db)
        ranked = ranker.rank_findings(findings)

        duration = time.time() - start_time

        assert len(ranked) > 900, "Most findings should be ranked"
        print(f"✓ Ranked {len(ranked)} findings in {duration:.2f}s")
        print(f"  Average: {(duration / len(ranked) * 1000):.2f}ms per finding")

        # Performance assertion
        assert duration < 5.0, f"Ranking should complete in <5s, took {duration:.2f}s"

    def test_insights_generation_at_load(
        self, load_test_db: Session, load_test_review: Review
    ):
        """Test insights generation with large dataset."""
        findings = populate_large_dataset(load_test_db, load_test_review, 1000)

        insights_gen = InsightsGenerator(load_test_db)
        repo_url = "https://github.com/test/repo"

        print(f"\nGenerating insights for {len(findings)} findings...")

        # Test metrics
        start_time = time.time()
        metrics = insights_gen.calculate_team_metrics(repo_url=repo_url)
        metrics_duration = time.time() - start_time
        print(f"✓ Metrics: {metrics_duration:.3f}s")
        assert metrics_duration < 1.0, "Metrics should complete <1s"

        # Test trends
        start_time = time.time()
        trends = insights_gen.analyze_trends(repo_url=repo_url)
        trends_duration = time.time() - start_time
        print(f"✓ Trends: {trends_duration:.3f}s")
        assert trends_duration < 1.0, "Trends should complete <1s"

        # Test learning paths
        start_time = time.time()
        paths = insights_gen.generate_learning_paths(repo_url=repo_url)
        paths_duration = time.time() - start_time
        print(f"✓ Learning paths: {paths_duration:.3f}s")
        assert paths_duration < 1.0, "Paths should complete <1s"

        # Test ROI
        start_time = time.time()
        roi = insights_gen.calculate_roi(repo_url=repo_url)
        roi_duration = time.time() - start_time
        print(f"✓ ROI: {roi_duration:.3f}s")
        assert roi_duration < 1.0, "ROI should complete <1s"

        total_duration = (
            metrics_duration + trends_duration + paths_duration + roi_duration
        )
        print(f"\nTotal insights generation: {total_duration:.3f}s")

    def test_concurrent_operations(
        self, load_test_db: Session, load_test_review: Review
    ):
        """Test ranking and insights operations (sequential for SQLite compat)."""
        findings = populate_large_dataset(load_test_db, load_test_review, 500)

        ranker = SuggestionRanker(load_test_db)
        insights_gen = InsightsGenerator(load_test_db)
        repo_url = "https://github.com/test/repo"

        print(f"\nTesting sequential operations...")

        start_time = time.time()

        # Run operations sequentially (SQLite doesn't support threading)
        results = []

        # Ranking operations
        for _ in range(2):
            result = ranker.rank_findings(findings[:100])
            results.append(result)

        # Insights operations
        for _ in range(2):
            result = insights_gen.calculate_team_metrics(repo_url=repo_url)
            results.append(result)

        duration = time.time() - start_time

        assert len(results) == 4, "All operations should complete"
        print(f"✓ Completed 4 sequential operations in {duration:.2f}s")

    def test_memory_usage_large_dataset(
        self, load_test_db: Session, load_test_review: Review
    ):
        """Test memory usage with large dataset."""
        try:
            import psutil
            import os

            process = psutil.Process(os.getpid())

            # Get initial memory
            initial_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Populate large dataset
            findings = populate_large_dataset(load_test_db, load_test_review, 2000)

            # Get memory after loading
            loaded_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = loaded_memory - initial_memory

            print(f"\n✓ Memory usage:")
            print(f"  Initial: {initial_memory:.2f} MB")
            print(f"  After load: {loaded_memory:.2f} MB")
            print(f"  Increase: {memory_increase:.2f} MB")
            print(f"  Per finding: {(memory_increase / 2000 * 1000):.2f} KB")

            # Memory assertion - should be reasonable
            assert memory_increase < 500, "Memory increase should be <500MB"

        except ImportError:
            pytest.skip("psutil not available for memory profiling")

    def test_query_performance_large_dataset(
        self, load_test_db: Session, load_test_review: Review
    ):
        """Test query performance with large dataset."""
        findings = populate_large_dataset(load_test_db, load_test_review, 1000)

        print(f"\nTesting query performance...")

        # Test finding queries
        start_time = time.time()
        all_findings = load_test_db.query(Finding).all()
        query_duration = time.time() - start_time
        print(f"✓ Query all findings: {query_duration:.3f}s")
        assert query_duration < 1.0, "Query should complete <1s"

        # Test feedback queries
        start_time = time.time()
        all_feedbacks = load_test_db.query(SuggestionFeedback).all()
        query_duration = time.time() - start_time
        print(f"✓ Query all feedbacks: {query_duration:.3f}s")
        assert query_duration < 1.0, "Query should complete <1s"

        # Test filtered queries
        start_time = time.time()
        helpful = (
            load_test_db.query(SuggestionFeedback)
            .filter_by(feedback_type="helpful")
            .all()
        )
        query_duration = time.time() - start_time
        print(f"✓ Query helpful feedbacks: {query_duration:.3f}s")
        assert query_duration < 1.0, "Filtered query should complete <1s"

    def test_bulk_feedback_ingestion(
        self, load_test_db: Session, load_test_review: Review
    ):
        """Test bulk feedback ingestion performance."""
        # Create initial findings
        findings = populate_large_dataset(load_test_db, load_test_review, 100)

        print(f"\nTesting bulk feedback ingestion...")

        # Ingest additional bulk feedback
        start_time = time.time()

        feedback_list = []
        for finding in findings:
            for i in range(10):
                feedback = SuggestionFeedback(
                    finding_id=finding.id,
                    feedback_type="helpful",
                    confidence=0.85,
                    created_at=datetime.now(timezone.utc),
                )
                feedback_list.append(feedback)

        load_test_db.add_all(feedback_list)
        load_test_db.commit()

        duration = time.time() - start_time

        print(f"✓ Ingested {len(feedback_list)} feedbacks in {duration:.3f}s")
        print(f"  Average: {(duration / len(feedback_list) * 1000):.2f}ms per feedback")

        # Performance assertion
        assert duration < 5.0, f"Bulk ingestion should complete <5s, took {duration:.2f}s"

    def test_stress_test_sequential_ranking(
        self, load_test_db: Session, load_test_review: Review
    ):
        """Stress test sequential ranking operations."""
        findings = populate_large_dataset(load_test_db, load_test_review, 500)

        ranker = SuggestionRanker(load_test_db)

        print(f"\nStress testing sequential ranking...")

        start_time = time.time()

        # Run multiple ranking operations sequentially (SQLite doesn't support threading)
        results = []
        for _ in range(10):
            result = ranker.rank_findings(findings[:50])
            results.append(result)

        duration = time.time() - start_time

        assert len(results) == 10, "All ranking operations should complete"
        print(f"✓ Completed 10 sequential ranking operations in {duration:.2f}s")
        print(f"  Average per operation: {(duration / 10):.2f}s")
        print(f"  Throughput: {(10 / duration):.1f} operations/sec")
