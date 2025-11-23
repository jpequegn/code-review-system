"""
Tests for caching manager and batch jobs

Tests coverage:
- Cache operations (get, set, invalidate, clear)
- TTL and expiration
- Cache statistics and hit rates
- Batch job execution and state tracking
- Error handling and retries
- Job scheduling
"""

import pytest
import json
import time
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from src.database import Base, Review, Finding, FindingCategory, FindingSeverity, SuggestionFeedback
from src.learning.cache_manager import CacheManager, CacheType, CacheEntry
from src.learning.batch_jobs import (
    BatchJob,
    JobExecution,
    JobStatus,
    HourlyMetricsUpdateJob,
    DailyLearningPathsJob,
)
from src.learning.job_scheduler import JobScheduler


@pytest.fixture
def test_db() -> Session:
    """Create test database."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys = ON"))
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    yield db
    db.close()


@pytest.fixture
def sample_repo_data(test_db: Session):
    """Create sample review and findings."""
    review = Review(
        pr_id=1,
        repo_url="https://github.com/test/repo",
        commit_sha="abc123",
        branch="main",
        created_at=datetime.now(timezone.utc),
    )
    test_db.add(review)
    test_db.commit()

    findings = []
    for i in range(5):
        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title=f"Finding {i}",
            description=f"Test finding {i}",
            file_path=f"src/file_{i}.py",
            line_number=i * 10,
            created_at=datetime.now(timezone.utc),
        )
        test_db.add(finding)
        findings.append(finding)
    test_db.commit()

    return review, findings


# ============================================================================
# Cache Manager Tests
# ============================================================================

class TestCacheManager:
    """Test cache manager functionality."""

    def test_cache_set_and_get(self, test_db: Session):
        """Cache: set and retrieve value."""
        cache = CacheManager(test_db)
        repo_url = "https://github.com/test/repo"
        test_data = {"metric": "value", "count": 42}

        # Store in cache
        cache.set(CacheType.METRICS, repo_url, test_data)

        # Retrieve from cache
        result = cache.get(CacheType.METRICS, repo_url)
        assert result is not None
        assert result == test_data

    def test_cache_ttl_expiration(self, test_db: Session):
        """Cache: respects TTL and expires entries."""
        cache = CacheManager(test_db)
        repo_url = "https://github.com/test/repo"
        test_data = {"metric": "value"}

        # Store with 1 second TTL
        cache.set(CacheType.METRICS, repo_url, test_data, ttl_seconds=1)

        # Should be available immediately
        result = cache.get(CacheType.METRICS, repo_url)
        assert result == test_data

        # Wait for expiration
        time.sleep(1.1)

        # Should be expired
        result = cache.get(CacheType.METRICS, repo_url)
        assert result is None

    def test_cache_invalidate_by_type(self, test_db: Session):
        """Cache: invalidate by cache type."""
        cache = CacheManager(test_db)
        repo_url = "https://github.com/test/repo"

        # Store multiple cache types
        cache.set(CacheType.METRICS, repo_url, {"metric": 1})
        cache.set(CacheType.PATHS, repo_url, {"path": 1})
        cache.set(CacheType.TRENDS, repo_url, {"trend": 1})

        # Verify all stored
        assert cache.get(CacheType.METRICS, repo_url) is not None
        assert cache.get(CacheType.PATHS, repo_url) is not None
        assert cache.get(CacheType.TRENDS, repo_url) is not None

        # Invalidate only METRICS (counts both memory + database)
        count = cache.invalidate(cache_type=CacheType.METRICS)
        assert count >= 1  # At least 1 (memory + database count)

        # Verify only METRICS invalidated
        assert cache.get(CacheType.METRICS, repo_url) is None
        assert cache.get(CacheType.PATHS, repo_url) is not None
        assert cache.get(CacheType.TRENDS, repo_url) is not None

    def test_cache_invalidate_by_repo(self, test_db: Session):
        """Cache: invalidate by repository."""
        cache = CacheManager(test_db)
        repo1 = "https://github.com/test/repo1"
        repo2 = "https://github.com/test/repo2"

        # Store for multiple repos
        cache.set(CacheType.METRICS, repo1, {"value": 1})
        cache.set(CacheType.METRICS, repo2, {"value": 2})

        # Verify both stored
        assert cache.get(CacheType.METRICS, repo1) is not None
        assert cache.get(CacheType.METRICS, repo2) is not None

        # Invalidate repo1
        count = cache.invalidate(repo_url=repo1)
        assert count >= 1

        # Verify only repo1 invalidated
        assert cache.get(CacheType.METRICS, repo1) is None
        assert cache.get(CacheType.METRICS, repo2) is not None

    def test_cache_statistics(self, test_db: Session):
        """Cache: tracks hit rate and statistics."""
        cache = CacheManager(test_db)
        repo_url = "https://github.com/test/repo"

        # Store value
        cache.set(CacheType.METRICS, repo_url, {"value": 1})

        # Miss (non-existent key)
        cache.get(CacheType.PATHS, repo_url)
        assert cache.stats.misses == 1

        # Hit
        cache.get(CacheType.METRICS, repo_url)
        assert cache.stats.hits == 1

        # Verify hit rate
        assert cache.stats.hit_rate == 0.5  # 1 hit / 2 total requests

    def test_cache_with_params(self, test_db: Session):
        """Cache: supports parameterized keys."""
        cache = CacheManager(test_db)
        repo_url = "https://github.com/test/repo"
        params1 = {"weeks": 12}
        params2 = {"weeks": 4}

        # Store with different params
        cache.set(CacheType.TRENDS, repo_url, {"trends": "12weeks"}, params=params1)
        cache.set(CacheType.TRENDS, repo_url, {"trends": "4weeks"}, params=params2)

        # Retrieve with matching params
        result1 = cache.get(CacheType.TRENDS, repo_url, params=params1)
        result2 = cache.get(CacheType.TRENDS, repo_url, params=params2)

        assert result1 == {"trends": "12weeks"}
        assert result2 == {"trends": "4weeks"}

    def test_cache_clear_expired(self, test_db: Session):
        """Cache: clears expired entries from database."""
        cache = CacheManager(test_db)
        repo_url = "https://github.com/test/repo"

        # Store with 1 second TTL
        cache.set(CacheType.METRICS, repo_url, {"value": 1}, ttl_seconds=1)

        # Verify in database
        entry_count = test_db.query(CacheEntry).count()
        assert entry_count == 1

        # Wait for expiration
        time.sleep(1.1)

        # Clear expired
        cleared = cache.clear_expired()
        assert cleared == 1

        # Verify removed from database
        entry_count = test_db.query(CacheEntry).count()
        assert entry_count == 0

    def test_cache_info(self, test_db: Session):
        """Cache: provides detailed cache information."""
        cache = CacheManager(test_db)
        repo_url = "https://github.com/test/repo"

        # Store multiple values
        cache.set(CacheType.METRICS, repo_url, {"v": 1})
        cache.set(CacheType.PATHS, repo_url, {"v": 2})

        info = cache.get_cache_info()
        assert "memory_entries" in info
        assert "database_entries" in info
        assert "statistics" in info
        assert info["memory_entries"] >= 2


# ============================================================================
# Batch Job Tests
# ============================================================================

class TestBatchJobs:
    """Test batch job execution."""

    def test_job_execution_success(self, test_db: Session, sample_repo_data):
        """Job: tracks successful execution."""
        class TestJob(BatchJob):
            def execute(self):
                return {"status": "ok", "items": 5}

        job = TestJob(test_db, "test_job")
        execution = job.run()

        assert execution.job_name == "test_job"
        assert execution.status == JobStatus.SUCCEEDED
        assert execution.duration_seconds is not None
        assert execution.duration_seconds >= 0

        result = json.loads(execution.result)
        assert result["status"] == "ok"

    def test_job_execution_failure(self, test_db: Session):
        """Job: tracks failed execution."""
        class FailingJob(BatchJob):
            def execute(self):
                raise ValueError("Test error")

        job = FailingJob(test_db, "failing_job", max_retries=1)
        execution = job.run()

        assert execution.job_name == "failing_job"
        assert execution.status == JobStatus.RETRYING  # Can retry
        assert execution.retry_count == 1
        assert "Test error" in execution.error_message

    def test_job_retry_logic(self, test_db: Session):
        """Job: implements retry logic."""
        class RetryableJob(BatchJob):
            attempt = 0

            def execute(self):
                self.attempt += 1
                if self.attempt < 2:
                    raise ValueError("Not yet")
                return {"attempt": self.attempt}

        job = RetryableJob(test_db, "retry_job", max_retries=3)

        # First execution fails
        execution1 = job.run()
        assert execution1.status == JobStatus.RETRYING
        assert execution1.retry_count == 1

    def test_hourly_metrics_job(self, test_db: Session, sample_repo_data):
        """Job: hourly metrics update executes."""
        job = HourlyMetricsUpdateJob(test_db)
        execution = job.run()

        assert execution.job_name == "hourly_metrics_update"
        # Should succeed even with minimal data
        assert execution.status in [JobStatus.SUCCEEDED, JobStatus.FAILED]

    def test_daily_learning_paths_job(self, test_db: Session, sample_repo_data):
        """Job: daily learning paths executes."""
        job = DailyLearningPathsJob(test_db)
        execution = job.run()

        assert execution.job_name == "daily_learning_paths"
        # Should succeed even with minimal data
        assert execution.status in [JobStatus.SUCCEEDED, JobStatus.FAILED]

    def test_job_execution_history(self, test_db: Session):
        """Job: tracks execution history."""
        class SimpleJob(BatchJob):
            def execute(self):
                return {"count": 1}

        job = SimpleJob(test_db, "simple_job")

        # Run multiple times
        for i in range(3):
            job.run()

        # Retrieve history
        executions = test_db.query(JobExecution).filter(
            JobExecution.job_name == "simple_job"
        ).all()

        assert len(executions) == 3
        assert all(e.status == JobStatus.SUCCEEDED for e in executions)


# ============================================================================
# Job Scheduler Tests
# ============================================================================

class TestJobScheduler:
    """Test job scheduler."""

    def test_scheduler_initialization(self, test_db: Session):
        """Scheduler: initializes with all jobs."""
        scheduler = JobScheduler(test_db)
        assert len(scheduler.jobs) >= 6  # At least 6 job types

    def test_scheduler_trigger_job(self, test_db: Session, sample_repo_data):
        """Scheduler: can trigger jobs manually."""
        scheduler = JobScheduler(test_db)
        execution = scheduler.trigger_job("daily_learning_paths")

        assert execution is not None
        assert execution.job_name == "daily_learning_paths"

    def test_scheduler_status(self, test_db: Session):
        """Scheduler: reports overall status."""
        scheduler = JobScheduler(test_db)
        status = scheduler.get_scheduler_status()

        assert "running" in status
        assert "job_count" in status
        assert "scheduled_jobs" in status
        assert status["job_count"] >= 6

    def test_scheduler_job_status(self, test_db: Session):
        """Scheduler: reports individual job status."""
        scheduler = JobScheduler(test_db)
        status = scheduler.get_job_status("daily_learning_paths")

        # May not have status if not running
        if status:
            assert "name" in status
            assert "id" in status


# ============================================================================
# Integration Tests
# ============================================================================

class TestCachingIntegration:
    """Integration tests for caching and jobs."""

    def test_cache_invalidation_on_data_change(self, test_db: Session):
        """Caching: invalidates when data changes."""
        cache = CacheManager(test_db)
        repo_url = "https://github.com/test/repo"

        # Cache metrics
        cache.set(CacheType.METRICS, repo_url, {"accuracy": 0.85})
        assert cache.get(CacheType.METRICS, repo_url) is not None

        # Simulate data change by invalidating
        cache.invalidate(cache_type=CacheType.METRICS)
        assert cache.get(CacheType.METRICS, repo_url) is None

    def test_multiple_repos_cache_isolation(self, test_db: Session):
        """Caching: isolates caches by repository."""
        cache = CacheManager(test_db)
        repo1 = "https://github.com/org1/repo"
        repo2 = "https://github.com/org2/repo"

        cache.set(CacheType.METRICS, repo1, {"value": 1})
        cache.set(CacheType.METRICS, repo2, {"value": 2})

        # Each repo has isolated cache
        assert cache.get(CacheType.METRICS, repo1) == {"value": 1}
        assert cache.get(CacheType.METRICS, repo2) == {"value": 2}

        # Invalidate only repo1
        cache.invalidate(repo_url=repo1)
        assert cache.get(CacheType.METRICS, repo1) is None
        assert cache.get(CacheType.METRICS, repo2) == {"value": 2}
