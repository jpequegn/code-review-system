"""
Job Scheduler - APScheduler-based background job execution

Schedules and executes batch jobs on a defined schedule:
- Hourly: metrics update, ROI calculation, integrity checks
- Daily: learning paths, trend analysis, anti-pattern detection, archiving
- Weekly: insights reports, confidence thresholds, cleanup

Features:
- Thread-safe execution
- Persistent job tracking
- Automatic retry with exponential backoff
- Error logging and alerting
- Job state monitoring
"""

import logging
from typing import List, Dict, Optional
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger
import pytz

from src.learning.batch_jobs import (
    BatchJob,
    HourlyMetricsUpdateJob,
    HourlyROICalculationJob,
    DailyLearningPathsJob,
    DailyTrendAnalysisJob,
    DailyAntiPatternDetectionJob,
    WeeklyCleanupJob,
    JobExecution,
    JobStatus,
)

logger = logging.getLogger(__name__)


class JobScheduler:
    """
    Background job scheduler for Phase 5 batch processing.

    Manages scheduling, execution, and monitoring of:
    - Hourly jobs (metrics, ROI, integrity)
    - Daily jobs (learning paths, trends, patterns)
    - Weekly jobs (reports, cleanup)
    """

    def __init__(self, db: Session, timezone_str: str = "UTC"):
        """
        Initialize job scheduler.

        Args:
            db: SQLAlchemy database session
            timezone_str: Timezone for scheduling (default: UTC)
        """
        self.db = db
        self.scheduler = BackgroundScheduler(timezone=pytz.timezone(timezone_str))
        self.jobs: Dict[str, BatchJob] = {}
        self._setup_jobs()

    def _setup_jobs(self) -> None:
        """Set up all batch jobs."""
        # Hourly jobs
        self.jobs["hourly_metrics_update"] = HourlyMetricsUpdateJob(self.db)
        self.jobs["hourly_roi_calculation"] = HourlyROICalculationJob(self.db)

        # Daily jobs
        self.jobs["daily_learning_paths"] = DailyLearningPathsJob(self.db)
        self.jobs["daily_trend_analysis"] = DailyTrendAnalysisJob(self.db)
        self.jobs["daily_anti_pattern_detection"] = DailyAntiPatternDetectionJob(self.db)

        # Weekly jobs
        self.jobs["weekly_cleanup"] = WeeklyCleanupJob(self.db)

    def _create_job_wrapper(self, job: BatchJob):
        """Create a wrapper function for job execution."""
        def wrapper():
            try:
                execution = job.run(scheduled_at=datetime.now(timezone.utc))
                logger.info(
                    f"Job {job.name} completed with status {execution.status} "
                    f"(duration: {execution.duration_seconds}s)"
                )
            except Exception as e:
                logger.error(f"Unexpected error in job {job.name}: {str(e)}", exc_info=True)

        return wrapper

    def start(self) -> None:
        """Start the job scheduler."""
        if self.scheduler.running:
            logger.warning("Scheduler already running")
            return

        # Schedule hourly jobs (run at minute 0 of each hour)
        self.scheduler.add_job(
            self._create_job_wrapper(self.jobs["hourly_metrics_update"]),
            CronTrigger(minute=0),
            id="hourly_metrics_update",
            name="Hourly Metrics Update",
            max_instances=1,
        )
        self.scheduler.add_job(
            self._create_job_wrapper(self.jobs["hourly_roi_calculation"]),
            CronTrigger(minute=15),
            id="hourly_roi_calculation",
            name="Hourly ROI Calculation",
            max_instances=1,
        )

        # Schedule daily jobs (run at 2 AM UTC)
        self.scheduler.add_job(
            self._create_job_wrapper(self.jobs["daily_learning_paths"]),
            CronTrigger(hour=2, minute=0),
            id="daily_learning_paths",
            name="Daily Learning Paths",
            max_instances=1,
        )
        self.scheduler.add_job(
            self._create_job_wrapper(self.jobs["daily_trend_analysis"]),
            CronTrigger(hour=2, minute=15),
            id="daily_trend_analysis",
            name="Daily Trend Analysis",
            max_instances=1,
        )
        self.scheduler.add_job(
            self._create_job_wrapper(self.jobs["daily_anti_pattern_detection"]),
            CronTrigger(hour=2, minute=30),
            id="daily_anti_pattern_detection",
            name="Daily Anti-Pattern Detection",
            max_instances=1,
        )

        # Schedule weekly jobs (run every Monday at 3 AM UTC)
        self.scheduler.add_job(
            self._create_job_wrapper(self.jobs["weekly_cleanup"]),
            CronTrigger(day_of_week=0, hour=3, minute=0),
            id="weekly_cleanup",
            name="Weekly Cleanup",
            max_instances=1,
        )

        self.scheduler.start()
        logger.info("Job scheduler started")

    def stop(self) -> None:
        """Stop the job scheduler."""
        if not self.scheduler.running:
            logger.warning("Scheduler not running")
            return

        self.scheduler.shutdown(wait=True)
        logger.info("Job scheduler stopped")

    def trigger_job(self, job_name: str) -> Optional[JobExecution]:
        """
        Manually trigger a job immediately.

        Args:
            job_name: Name of the job to trigger

        Returns:
            JobExecution record or None if job not found
        """
        if job_name not in self.jobs:
            logger.error(f"Job {job_name} not found")
            return None

        logger.info(f"Manually triggering job: {job_name}")
        job = self.jobs[job_name]
        execution = job.run(scheduled_at=datetime.now(timezone.utc))
        return execution

    def get_job_status(self, job_name: str) -> Optional[Dict]:
        """
        Get status of a scheduled job.

        Args:
            job_name: Name of the job

        Returns:
            Job status dict or None if job not found
        """
        job = self.scheduler.get_job(job_name)
        if not job:
            return None

        return {
            "name": job.name,
            "id": job.id,
            "trigger": str(job.trigger),
            "next_run_time": job.next_run_time.isoformat() if job.next_run_time else None,
        }

    def get_all_job_statuses(self) -> Dict[str, Dict]:
        """Get status of all scheduled jobs."""
        statuses = {}
        for job_name in self.jobs:
            status = self.get_job_status(job_name)
            if status:
                statuses[job_name] = status
        return statuses

    def get_job_history(self, job_name: str, limit: int = 10) -> List[JobExecution]:
        """
        Get execution history for a job.

        Args:
            job_name: Name of the job
            limit: Maximum number of records to return

        Returns:
            List of JobExecution records
        """
        executions = (
            self.db.query(JobExecution)
            .filter(JobExecution.job_name == job_name)
            .order_by(JobExecution.created_at.desc())
            .limit(limit)
            .all()
        )
        return executions

    def get_failed_jobs(self) -> List[JobExecution]:
        """Get all failed jobs that haven't been resolved."""
        failed = (
            self.db.query(JobExecution)
            .filter(JobExecution.status == JobStatus.FAILED)
            .order_by(JobExecution.created_at.desc())
            .all()
        )
        return failed

    def get_retrying_jobs(self) -> List[JobExecution]:
        """Get all jobs currently retrying."""
        retrying = (
            self.db.query(JobExecution)
            .filter(JobExecution.status == JobStatus.RETRYING)
            .order_by(JobExecution.created_at.desc())
            .all()
        )
        return retrying

    def get_scheduler_status(self) -> Dict:
        """Get overall scheduler status."""
        failed_jobs = self.get_failed_jobs()
        retrying_jobs = self.get_retrying_jobs()

        return {
            "running": self.scheduler.running,
            "job_count": len(self.jobs),
            "scheduled_jobs": self.get_all_job_statuses(),
            "failed_jobs_count": len(failed_jobs),
            "retrying_jobs_count": len(retrying_jobs),
        }


# Global scheduler instance (initialized on startup)
_scheduler_instance: Optional[JobScheduler] = None


def get_scheduler(db: Session) -> JobScheduler:
    """Get or create global scheduler instance."""
    global _scheduler_instance
    if _scheduler_instance is None:
        _scheduler_instance = JobScheduler(db)
    return _scheduler_instance


def initialize_scheduler(db: Session) -> JobScheduler:
    """Initialize and start the global scheduler."""
    global _scheduler_instance
    _scheduler_instance = JobScheduler(db)
    _scheduler_instance.start()
    return _scheduler_instance
