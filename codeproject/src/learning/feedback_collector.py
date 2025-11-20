"""
Feedback Collector - Collects and stores suggestion feedback in the database.

Links parsed feedback signals to specific findings and persists them for
learning engine analysis.
"""

from typing import Optional, List
from datetime import datetime, timezone

from sqlalchemy.orm import Session
from sqlalchemy import and_

from src.database import SuggestionFeedback, Finding
from src.learning.feedback_parser import ParsedFeedback


class FeedbackCollector:
    """
    Collects parsed feedback signals and stores them in the database.

    Handles:
    - Linking feedback to findings
    - Preventing duplicate entries
    - Handling multiple feedbacks per finding
    - Data integrity and consistency
    """

    @staticmethod
    def collect_feedback(
        db: Session,
        finding_id: int,
        parsed_feedback: ParsedFeedback,
        pr_number: Optional[int] = None,
        confidence_override: Optional[float] = None,
    ) -> Optional[SuggestionFeedback]:
        """
        Store parsed feedback in the database.

        Args:
            db: Database session
            finding_id: ID of the finding being referenced
            parsed_feedback: ParsedFeedback object from FeedbackParser
            pr_number: Optional PR number for tracking
            confidence_override: Optional override for confidence score

        Returns:
            Created SuggestionFeedback object, or None if error
        """
        try:
            # Verify finding exists
            finding = db.query(Finding).filter(Finding.id == finding_id).first()
            if not finding:
                return None

            # Check for duplicate feedback (same type, same finding, recent)
            existing = FeedbackCollector._check_duplicate_feedback(
                db, finding_id, parsed_feedback
            )
            if existing:
                return existing

            # Create new feedback record
            feedback = SuggestionFeedback(
                finding_id=finding_id,
                feedback_type=parsed_feedback.feedback_type,
                confidence=confidence_override or parsed_feedback.confidence,
                developer_id=parsed_feedback.developer_id,
                developer_comment=parsed_feedback.raw_text,
                commit_hash=parsed_feedback.commit_hash,
                pr_number=pr_number,
                created_at=parsed_feedback.timestamp or datetime.now(timezone.utc),
                updated_at=datetime.now(timezone.utc),
            )

            db.add(feedback)
            db.commit()
            db.refresh(feedback)
            return feedback

        except Exception as e:
            db.rollback()
            raise

    @staticmethod
    def collect_multiple_feedbacks(
        db: Session,
        finding_id: int,
        parsed_feedbacks: List[ParsedFeedback],
        pr_number: Optional[int] = None,
    ) -> List[SuggestionFeedback]:
        """
        Store multiple parsed feedbacks for a finding.

        Args:
            db: Database session
            finding_id: ID of the finding
            parsed_feedbacks: List of ParsedFeedback objects
            pr_number: Optional PR number

        Returns:
            List of created SuggestionFeedback objects
        """
        results = []
        for feedback in parsed_feedbacks:
            result = FeedbackCollector.collect_feedback(
                db, finding_id, feedback, pr_number
            )
            if result:
                results.append(result)
        return results

    @staticmethod
    def get_feedback_for_finding(
        db: Session, finding_id: int
    ) -> List[SuggestionFeedback]:
        """
        Retrieve all feedback for a specific finding.

        Args:
            db: Database session
            finding_id: ID of the finding

        Returns:
            List of SuggestionFeedback records
        """
        return (
            db.query(SuggestionFeedback)
            .filter(SuggestionFeedback.finding_id == finding_id)
            .order_by(SuggestionFeedback.created_at.desc())
            .all()
        )

    @staticmethod
    def get_feedback_by_type(
        db: Session, feedback_type: str, limit: int = 100
    ) -> List[SuggestionFeedback]:
        """
        Retrieve feedback of a specific type (accepted, rejected, ignored).

        Args:
            db: Database session
            feedback_type: Type of feedback ("accepted", "rejected", "ignored")
            limit: Maximum number of results

        Returns:
            List of SuggestionFeedback records
        """
        return (
            db.query(SuggestionFeedback)
            .filter(SuggestionFeedback.feedback_type == feedback_type)
            .order_by(SuggestionFeedback.created_at.desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def get_recent_feedback(
        db: Session, days: int = 7, limit: int = 100
    ) -> List[SuggestionFeedback]:
        """
        Retrieve feedback from the last N days.

        Args:
            db: Database session
            days: Number of days to look back
            limit: Maximum number of results

        Returns:
            List of recent SuggestionFeedback records
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        return (
            db.query(SuggestionFeedback)
            .filter(SuggestionFeedback.created_at >= cutoff)
            .order_by(SuggestionFeedback.created_at.desc())
            .limit(limit)
            .all()
        )

    @staticmethod
    def calculate_acceptance_rate(
        db: Session, finding_id: int
    ) -> tuple[float, int]:
        """
        Calculate acceptance rate for a finding.

        Args:
            db: Database session
            finding_id: ID of the finding

        Returns:
            Tuple of (acceptance_rate, total_feedback_count)
        """
        feedbacks = FeedbackCollector.get_feedback_for_finding(db, finding_id)

        if not feedbacks:
            return 0.0, 0

        accepted = sum(1 for f in feedbacks if f.feedback_type == "accepted")
        return accepted / len(feedbacks), len(feedbacks)

    @staticmethod
    def calculate_confidence_average(
        db: Session, finding_id: int
    ) -> Optional[float]:
        """
        Calculate average confidence for feedbacks on a finding.

        Args:
            db: Database session
            finding_id: ID of the finding

        Returns:
            Average confidence score (0.0-1.0), or None if no feedback
        """
        feedbacks = (
            db.query(SuggestionFeedback)
            .filter(
                and_(
                    SuggestionFeedback.finding_id == finding_id,
                    SuggestionFeedback.confidence.isnot(None),
                )
            )
            .all()
        )

        if not feedbacks:
            return None

        return sum(f.confidence for f in feedbacks) / len(feedbacks)

    @staticmethod
    def delete_feedback(db: Session, feedback_id: int) -> bool:
        """
        Delete a feedback record.

        Args:
            db: Database session
            feedback_id: ID of the feedback to delete

        Returns:
            True if deleted, False if not found
        """
        try:
            feedback = (
                db.query(SuggestionFeedback)
                .filter(SuggestionFeedback.id == feedback_id)
                .first()
            )
            if not feedback:
                return False

            db.delete(feedback)
            db.commit()
            return True
        except Exception as e:
            db.rollback()
            raise

    @staticmethod
    def _check_duplicate_feedback(
        db: Session, finding_id: int, parsed_feedback: ParsedFeedback
    ) -> Optional[SuggestionFeedback]:
        """
        Check if similar feedback already exists (within 1 hour).

        Args:
            db: Database session
            finding_id: ID of the finding
            parsed_feedback: Parsed feedback to check

        Returns:
            Existing feedback if found, None otherwise
        """
        from datetime import timedelta

        cutoff = datetime.now(timezone.utc) - timedelta(hours=1)

        # Look for exact type + developer match
        existing = (
            db.query(SuggestionFeedback)
            .filter(
                and_(
                    SuggestionFeedback.finding_id == finding_id,
                    SuggestionFeedback.feedback_type == parsed_feedback.feedback_type,
                    SuggestionFeedback.developer_id == parsed_feedback.developer_id,
                    SuggestionFeedback.created_at >= cutoff,
                )
            )
            .first()
        )

        return existing


class FeedbackStats:
    """Utility class for computing feedback statistics."""

    @staticmethod
    def get_stats_for_finding(
        db: Session, finding_id: int
    ) -> dict:
        """
        Get comprehensive feedback stats for a finding.

        Args:
            db: Database session
            finding_id: ID of the finding

        Returns:
            Dictionary with feedback statistics
        """
        feedbacks = FeedbackCollector.get_feedback_for_finding(db, finding_id)

        if not feedbacks:
            return {
                "total": 0,
                "accepted": 0,
                "rejected": 0,
                "ignored": 0,
                "acceptance_rate": 0.0,
                "avg_confidence": None,
            }

        total = len(feedbacks)
        accepted = sum(1 for f in feedbacks if f.feedback_type == "accepted")
        rejected = sum(1 for f in feedbacks if f.feedback_type == "rejected")
        ignored = total - accepted - rejected

        confidences = [f.confidence for f in feedbacks if f.confidence is not None]
        avg_confidence = sum(confidences) / len(confidences) if confidences else None

        return {
            "total": total,
            "accepted": accepted,
            "rejected": rejected,
            "ignored": ignored,
            "acceptance_rate": accepted / total if total > 0 else 0.0,
            "avg_confidence": avg_confidence,
        }

    @staticmethod
    def get_stats_by_type(
        db: Session,
    ) -> dict:
        """
        Get feedback statistics aggregated by feedback type.

        Args:
            db: Database session

        Returns:
            Dictionary with stats by feedback type
        """
        from sqlalchemy import func

        result = (
            db.query(
                SuggestionFeedback.feedback_type,
                func.count(SuggestionFeedback.id).label("count"),
                func.avg(SuggestionFeedback.confidence).label("avg_confidence"),
            )
            .group_by(SuggestionFeedback.feedback_type)
            .all()
        )

        stats = {}
        for feedback_type, count, avg_conf in result:
            stats[feedback_type] = {
                "count": count,
                "avg_confidence": avg_conf,
            }

        return stats
