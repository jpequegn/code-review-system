"""
Feedback Collection System

Collects user feedback on findings to improve analysis accuracy over time.
Tracks which findings were helpful, false positives, and which became production bugs.
"""

from datetime import datetime, timezone
from typing import Optional, List

from sqlalchemy.orm import Session

from src.database import FindingFeedback, FeedbackType, IssueValidation, ProductionIssue, Finding, FindingSeverity


class FeedbackCollector:
    """
    Collects and manages user feedback on code findings.

    Enables the system to learn from user decisions and improve accuracy.
    """

    def __init__(self, db: Session):
        """
        Initialize feedback collector.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    def record_finding_feedback(
        self,
        finding_id: int,
        feedback_type: FeedbackType,
        validation: IssueValidation = IssueValidation.UNVALIDATED,
        user_notes: Optional[str] = None,
        severity_adjustment: int = 0,
        helpful: Optional[bool] = None,
    ) -> FindingFeedback:
        """
        Record user feedback on a finding.

        Args:
            finding_id: ID of the finding
            feedback_type: Type of feedback (HELPFUL, FALSE_POSITIVE, etc.)
            validation: Validation outcome of the finding
            user_notes: Optional notes from user
            severity_adjustment: Adjustment to severity (-2 to +2)
            helpful: Was this finding helpful?

        Returns:
            FindingFeedback record created
        """
        feedback = FindingFeedback(
            finding_id=finding_id,
            feedback_type=feedback_type,
            validation=validation,
            user_notes=user_notes,
            severity_adjustment=severity_adjustment,
            helpful=helpful,
        )
        self.db.add(feedback)
        self.db.commit()
        self.db.refresh(feedback)
        return feedback

    def record_issue_validation(
        self,
        finding_id: int,
        validation: IssueValidation,
        helpful: Optional[bool] = None,
    ) -> FindingFeedback:
        """
        Record validation status of a finding after user investigation.

        Args:
            finding_id: ID of the finding
            validation: Outcome (CONFIRMED, FALSE, FIXED, etc.)
            helpful: Was this finding helpful?

        Returns:
            FindingFeedback record
        """
        # Check if feedback already exists for this finding
        existing = self.db.query(FindingFeedback).filter(
            FindingFeedback.finding_id == finding_id
        ).first()

        if existing:
            existing.validation = validation
            if helpful is not None:
                existing.helpful = helpful
            self.db.commit()
            self.db.refresh(existing)
            return existing
        else:
            return self.record_finding_feedback(
                finding_id=finding_id,
                feedback_type=FeedbackType.HELPFUL if validation == IssueValidation.CONFIRMED else FeedbackType.FALSE_POSITIVE,
                validation=validation,
                helpful=helpful,
            )

    def record_production_bug(
        self,
        repo_url: str,
        description: str,
        severity: FindingSeverity,
        date_discovered: datetime,
        file_path: Optional[str] = None,
        time_to_fix_minutes: Optional[int] = None,
        related_finding_ids: Optional[List[int]] = None,
    ) -> ProductionIssue:
        """
        Record a bug that occurred in production.

        Used to track false negatives and learn what we missed.

        Args:
            repo_url: Repository URL
            description: Description of the bug
            severity: Severity of the bug
            date_discovered: When the bug was discovered
            file_path: File path affected (optional)
            time_to_fix_minutes: Time spent fixing (optional)
            related_finding_ids: IDs of findings that could have caught this

        Returns:
            ProductionIssue record created
        """
        # Convert related_finding_ids to string (comma-separated)
        related_ids_str = None
        if related_finding_ids:
            related_ids_str = ",".join(str(fid) for fid in related_finding_ids)

        issue = ProductionIssue(
            repo_url=repo_url,
            description=description,
            severity=severity,
            date_discovered=date_discovered,
            file_path=file_path,
            time_to_fix_minutes=time_to_fix_minutes,
            related_finding_ids=related_ids_str,
        )
        self.db.add(issue)
        self.db.commit()
        self.db.refresh(issue)
        return issue

    def get_feedback_for_finding(self, finding_id: int) -> Optional[FindingFeedback]:
        """
        Get feedback for a specific finding.

        Args:
            finding_id: ID of the finding

        Returns:
            FindingFeedback or None if no feedback exists
        """
        return self.db.query(FindingFeedback).filter(
            FindingFeedback.finding_id == finding_id
        ).first()

    def get_all_feedback(self) -> List[FindingFeedback]:
        """
        Get all feedback records.

        Returns:
            List of all FindingFeedback records
        """
        return self.db.query(FindingFeedback).all()

    def get_feedback_by_type(self, feedback_type: FeedbackType) -> List[FindingFeedback]:
        """
        Get feedback records of a specific type.

        Args:
            feedback_type: Type of feedback to filter by

        Returns:
            List of matching FindingFeedback records
        """
        return self.db.query(FindingFeedback).filter(
            FindingFeedback.feedback_type == feedback_type
        ).all()

    def get_helpful_findings(self) -> List[FindingFeedback]:
        """
        Get findings that users marked as helpful.

        Returns:
            List of helpful FindingFeedback records
        """
        return self.db.query(FindingFeedback).filter(
            FindingFeedback.helpful == True
        ).all()

    def get_false_positives(self) -> List[FindingFeedback]:
        """
        Get findings that were false positives.

        Returns:
            List of false positive FindingFeedback records
        """
        return self.db.query(FindingFeedback).filter(
            FindingFeedback.feedback_type == FeedbackType.FALSE_POSITIVE
        ).all()

    def get_missed_issues(self) -> List[FindingFeedback]:
        """
        Get issues we should have caught but didn't.

        Returns:
            List of missed FindingFeedback records
        """
        return self.db.query(FindingFeedback).filter(
            FindingFeedback.feedback_type == FeedbackType.MISSED
        ).all()

    def get_production_issues(self, repo_url: Optional[str] = None) -> List[ProductionIssue]:
        """
        Get production issues, optionally filtered by repo.

        Args:
            repo_url: Repository URL to filter by (optional)

        Returns:
            List of ProductionIssue records
        """
        query = self.db.query(ProductionIssue)
        if repo_url:
            query = query.filter(ProductionIssue.repo_url == repo_url)
        return query.all()

    def get_recent_feedback(self, days: int = 7) -> List[FindingFeedback]:
        """
        Get feedback from the last N days.

        Args:
            days: Number of days to look back

        Returns:
            List of recent FindingFeedback records
        """
        cutoff = datetime.now(timezone.utc) - __import__('datetime').timedelta(days=days)
        return self.db.query(FindingFeedback).filter(
            FindingFeedback.created_at >= cutoff
        ).all()

    def get_feedback_count(self) -> int:
        """
        Get total number of feedback records.

        Returns:
            Count of feedback records
        """
        return self.db.query(FindingFeedback).count()

    def get_false_positive_count(self) -> int:
        """
        Get count of false positive findings.

        Returns:
            Count of false positives
        """
        return self.db.query(FindingFeedback).filter(
            FindingFeedback.feedback_type == FeedbackType.FALSE_POSITIVE
        ).count()
