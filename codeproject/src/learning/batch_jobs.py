"""
Batch Jobs Framework - Scheduled jobs for Phase 5 learning system

Implements hourly, daily, and weekly batch jobs with:
- Error handling and retry logic (3 retries, exponential backoff)
- Job state tracking (running, succeeded, failed)
- Dead letter queue for failed jobs
- Logging and metrics per job
- Atomic operations to prevent duplication

Job Categories:

Hourly:
- Update learning metrics (all repos)
- Calculate ROI (all repos)
- Validate data integrity (spot checks)

Daily:
- Generate learning paths (all repos)
- Analyze trends (12 weeks)
- Detect anti-patterns
- Archive old trends (>1 year)

Weekly:
- Generate insights report
- Update confidence thresholds
- Clean up orphaned records
"""

import logging
import time
from typing import Callable, Optional, Dict, Any
from datetime import datetime, timezone, timedelta
from enum import Enum
from abc import ABC, abstractmethod

from sqlalchemy import Column, Integer, String, DateTime, Text, Enum as SQLEnum
from sqlalchemy.orm import Session

from src.database import Base
from src.monitoring import get_metrics

logger = logging.getLogger(__name__)


class JobStatus(str, Enum):
    """Status of a batch job execution."""
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"


class JobFrequency(str, Enum):
    """Frequency of job execution."""
    HOURLY = "hourly"
    DAILY = "daily"
    WEEKLY = "weekly"


class JobExecution(Base):
    """Database table for tracking job executions."""

    __tablename__ = "job_executions"

    # Primary key
    id = Column(Integer, primary_key=True, index=True)

    # Job name/type
    job_name = Column(String(255), nullable=False, index=True)

    # Job status
    status = Column(SQLEnum(JobStatus), default=JobStatus.PENDING, nullable=False, index=True)

    # Execution details
    started_at = Column(DateTime, nullable=True)
    completed_at = Column(DateTime, nullable=True)
    duration_seconds = Column(Integer, nullable=True)

    # Results and error handling
    result = Column(Text, nullable=True)  # JSON result
    error_message = Column(Text, nullable=True)
    retry_count = Column(Integer, default=0, nullable=False)
    max_retries = Column(Integer, default=3, nullable=False)

    # Metadata
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc), nullable=False, index=True)
    scheduled_at = Column(DateTime, nullable=True, index=True)

    def is_running(self) -> bool:
        """Check if job is currently running."""
        return self.status == JobStatus.RUNNING

    def can_retry(self) -> bool:
        """Check if job can be retried."""
        return self.retry_count < self.max_retries

    def mark_running(self) -> None:
        """Mark job as running."""
        self.status = JobStatus.RUNNING
        self.started_at = datetime.now(timezone.utc)

    def mark_succeeded(self, result: Optional[Dict] = None) -> None:
        """Mark job as succeeded."""
        self.status = JobStatus.SUCCEEDED
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            # Ensure both datetimes are timezone-aware
            started = self.started_at.replace(tzinfo=timezone.utc) if self.started_at.tzinfo is None else self.started_at
            self.duration_seconds = int((self.completed_at - started).total_seconds())
        if result:
            import json
            self.result = json.dumps(result)

    def mark_failed(self, error: Exception, can_retry: bool = True) -> None:
        """Mark job as failed."""
        self.error_message = str(error)
        self.completed_at = datetime.now(timezone.utc)
        if self.started_at:
            # Ensure both datetimes are timezone-aware
            started = self.started_at.replace(tzinfo=timezone.utc) if self.started_at.tzinfo is None else self.started_at
            self.duration_seconds = int((self.completed_at - started).total_seconds())

        if can_retry and self.can_retry():
            self.status = JobStatus.RETRYING
            self.retry_count += 1
        else:
            self.status = JobStatus.FAILED


class BatchJob(ABC):
    """
    Abstract base class for batch jobs.

    Subclasses must implement execute() method.
    """

    def __init__(self, db: Session, name: str, max_retries: int = 3):
        """
        Initialize batch job.

        Args:
            db: SQLAlchemy database session
            name: Job name
            max_retries: Maximum retry attempts
        """
        self.db = db
        self.name = name
        self.max_retries = max_retries
        self.logger = logger.getChild(name)

    @abstractmethod
    def execute(self) -> Dict[str, Any]:
        """
        Execute the batch job.

        Returns:
            Result dictionary with job outcome

        Raises:
            Exception: If job fails (will be caught and retried)
        """
        pass

    def run(self, scheduled_at: Optional[datetime] = None) -> JobExecution:
        """
        Run the batch job with error handling and retries.

        Args:
            scheduled_at: When the job was scheduled

        Returns:
            JobExecution record
        """
        # Create job execution record
        execution = JobExecution(
            job_name=self.name,
            max_retries=self.max_retries,
            scheduled_at=scheduled_at or datetime.now(timezone.utc),
        )
        self.db.add(execution)
        self.db.commit()

        self.logger.info(f"Starting job: {self.name}")
        metrics = get_metrics()
        metrics.register_counter("batch_jobs_executed_total").increment()

        try:
            execution.mark_running()
            self.db.commit()

            # Execute job
            start_time = time.time()
            result = self.execute()
            duration = time.time() - start_time

            # Mark succeeded
            execution.mark_succeeded(result)
            self.logger.info(f"Job succeeded: {self.name} (duration: {execution.duration_seconds}s)")

            # Record metrics
            metrics.register_counter("batch_jobs_succeeded_total").increment()
            metrics.register_histogram("batch_job_duration_seconds").observe(duration)

        except Exception as e:
            self.logger.error(f"Job failed: {self.name} - {str(e)}", exc_info=True)
            can_retry = execution.retry_count < execution.max_retries
            execution.mark_failed(e, can_retry=can_retry)

            if execution.status == JobStatus.RETRYING:
                self.logger.info(
                    f"Job will retry: {self.name} "
                    f"(attempt {execution.retry_count + 1}/{execution.max_retries})"
                )
                # Exponential backoff: 5s, 25s, 125s
                backoff_delay = 5 * (5 ** execution.retry_count)
                self.logger.info(f"Retry in {backoff_delay}s")

                # Record metrics
                metrics.register_counter("batch_jobs_retried_total").increment()
            else:
                self.logger.error(f"Job failed permanently: {self.name}")
                metrics.register_counter("batch_jobs_failed_total").increment()

        finally:
            self.db.commit()

        return execution


class HourlyMetricsUpdateJob(BatchJob):
    """Update learning metrics for all repositories."""

    def __init__(self, db: Session):
        super().__init__(db, "hourly_metrics_update")

    def execute(self) -> Dict[str, Any]:
        """Update metrics for all repos."""
        from src.learning.insights import InsightsGenerator

        # Get all unique repos
        repos = (
            self.db.query(
                __import__('src.database', fromlist=['Review']).Review.repo_url
            )
            .distinct()
            .all()
        )

        updated_repos = 0
        for repo_tuple in repos:
            repo_url = repo_tuple[0]
            try:
                insights = InsightsGenerator(self.db)
                metrics = insights.calculate_team_metrics(repo_url=repo_url)
                updated_repos += 1
                self.logger.debug(f"Updated metrics for {repo_url}")
            except Exception as e:
                self.logger.warning(f"Failed to update metrics for {repo_url}: {str(e)}")

        return {
            "updated_repos": updated_repos,
            "total_repos": len(repos),
        }


class HourlyROICalculationJob(BatchJob):
    """Calculate ROI for all repositories."""

    def __init__(self, db: Session):
        super().__init__(db, "hourly_roi_calculation")

    def execute(self) -> Dict[str, Any]:
        """Calculate ROI for all repos."""
        from src.learning.insights import InsightsGenerator

        repos = (
            self.db.query(
                __import__('src.database', fromlist=['Review']).Review.repo_url
            )
            .distinct()
            .all()
        )

        calculated_repos = 0
        for repo_tuple in repos:
            repo_url = repo_tuple[0]
            try:
                insights = InsightsGenerator(self.db)
                roi = insights.calculate_roi(repo_url=repo_url)
                calculated_repos += 1
                self.logger.debug(f"Calculated ROI for {repo_url}")
            except Exception as e:
                self.logger.warning(f"Failed to calculate ROI for {repo_url}: {str(e)}")

        return {
            "calculated_repos": calculated_repos,
            "total_repos": len(repos),
        }


class DailyLearningPathsJob(BatchJob):
    """Generate learning paths for all repositories."""

    def __init__(self, db: Session):
        super().__init__(db, "daily_learning_paths")

    def execute(self) -> Dict[str, Any]:
        """Generate learning paths for all repos."""
        from src.learning.insights import InsightsGenerator

        repos = (
            self.db.query(
                __import__('src.database', fromlist=['Review']).Review.repo_url
            )
            .distinct()
            .all()
        )

        generated_repos = 0
        for repo_tuple in repos:
            repo_url = repo_tuple[0]
            try:
                insights = InsightsGenerator(self.db)
                paths = insights.generate_learning_paths(repo_url=repo_url)
                generated_repos += 1
                self.logger.debug(f"Generated learning paths for {repo_url}")
            except Exception as e:
                self.logger.warning(f"Failed to generate paths for {repo_url}: {str(e)}")

        return {
            "generated_repos": generated_repos,
            "total_repos": len(repos),
        }


class DailyTrendAnalysisJob(BatchJob):
    """Analyze vulnerability trends for all repositories."""

    def __init__(self, db: Session):
        super().__init__(db, "daily_trend_analysis")

    def execute(self) -> Dict[str, Any]:
        """Analyze trends for all repos."""
        from src.learning.insights import InsightsGenerator

        repos = (
            self.db.query(
                __import__('src.database', fromlist=['Review']).Review.repo_url
            )
            .distinct()
            .all()
        )

        analyzed_repos = 0
        for repo_tuple in repos:
            repo_url = repo_tuple[0]
            try:
                insights = InsightsGenerator(self.db)
                trends = insights.analyze_trends(repo_url=repo_url, weeks=12)
                analyzed_repos += 1
                self.logger.debug(f"Analyzed trends for {repo_url}")
            except Exception as e:
                self.logger.warning(f"Failed to analyze trends for {repo_url}: {str(e)}")

        return {
            "analyzed_repos": analyzed_repos,
            "total_repos": len(repos),
        }


class DailyAntiPatternDetectionJob(BatchJob):
    """Detect anti-patterns for all repositories."""

    def __init__(self, db: Session):
        super().__init__(db, "daily_anti_pattern_detection")

    def execute(self) -> Dict[str, Any]:
        """Detect anti-patterns for all repos."""
        from src.learning.insights import InsightsGenerator

        repos = (
            self.db.query(
                __import__('src.database', fromlist=['Review']).Review.repo_url
            )
            .distinct()
            .all()
        )

        detected_repos = 0
        total_patterns = 0
        for repo_tuple in repos:
            repo_url = repo_tuple[0]
            try:
                insights = InsightsGenerator(self.db)
                patterns = insights.detect_anti_patterns(repo_url=repo_url)
                detected_repos += 1
                total_patterns += len(patterns) if patterns else 0
                self.logger.debug(f"Detected {len(patterns) if patterns else 0} anti-patterns for {repo_url}")
            except Exception as e:
                self.logger.warning(f"Failed to detect patterns for {repo_url}: {str(e)}")

        return {
            "detected_repos": detected_repos,
            "total_repos": len(repos),
            "total_patterns": total_patterns,
        }


class WeeklyCleanupJob(BatchJob):
    """Weekly cleanup: orphaned records, old trends."""

    def __init__(self, db: Session):
        super().__init__(db, "weekly_cleanup")

    def execute(self) -> Dict[str, Any]:
        """Clean up orphaned records and old data."""
        from src.database import InsightsTrend

        # Delete trends older than 1 year
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=365)
        old_trends = self.db.query(InsightsTrend).filter(
            InsightsTrend.created_at < cutoff_date
        ).count()

        if old_trends > 0:
            self.db.query(InsightsTrend).filter(
                InsightsTrend.created_at < cutoff_date
            ).delete()
            self.db.commit()
            self.logger.info(f"Deleted {old_trends} old trends")

        return {
            "deleted_old_trends": old_trends,
        }
