"""
Acceptance Rate Analytics - Analyze feedback to extract acceptance patterns.

Calculates acceptance rates by category, severity, and type to understand
which types of findings are most/least accepted by developers.

This data enables:
- Confidence tuning (5.2.3)
- Suggestion ranking (5.2.4)
- Pattern learning (5.2.5)
"""

from typing import Optional
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database import (
    Finding,
    SuggestionFeedback,
    LearningMetrics,
    FindingCategory,
    FindingSeverity,
)
from src.learning.metrics import AcceptanceMetrics


class AcceptanceAnalyzer:
    """
    Analyzes feedback to extract acceptance patterns.

    Calculates acceptance rates by:
    - Finding category (SQL Injection, Resource Leak, etc)
    - Severity level (CRITICAL, HIGH, MEDIUM, LOW)
    - Issue type (SECURITY vs PERFORMANCE)
    - Custom filters (category + severity + type combinations)

    Example:
        analyzer = AcceptanceAnalyzer(db_session)

        # Get acceptance rates by category
        rates = analyzer.calculate_acceptance_by_category()
        for metric in rates:
            print(f"{metric.finding_category}: {metric.acceptance_rate:.1%}")

        # Get specific combination
        sql_critical = analyzer.calculate_composite_acceptance(
            category="SQL Injection",
            severity="CRITICAL"
        )
        print(f"SQL Injection (CRITICAL): {sql_critical.acceptance_rate:.1%}")

        # Persist to database
        analyzer.persist_all_metrics()
    """

    def __init__(self, db: Session):
        """
        Initialize analyzer with database session.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    # ============================================================================
    # Core Analytics Methods
    # ============================================================================

    def calculate_acceptance_by_category(
        self, days: int = 30
    ) -> list[AcceptanceMetrics]:
        """
        Acceptance rates for each finding category/type.

        Returns acceptance metrics grouped by finding category (e.g., "SQL Injection",
        "Resource Leak"). For each category, calculates:
        - Total findings
        - Count accepted, rejected, ignored
        - Acceptance rate (accepted / (accepted + rejected))
        - Average confidence scores
        - Fix rate (actually applied / total)

        Args:
            days: Only include feedback from last N days (default: 30)

        Returns:
            List of AcceptanceMetrics sorted by acceptance_rate DESC
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        # Get all unique finding titles (categories)
        categories = (
            self.db.query(Finding.title)
            .filter(Finding.created_at >= cutoff_date)
            .distinct()
            .all()
        )

        results = []
        for (category,) in categories:
            if category:  # Skip null categories
                metrics = self.calculate_composite_acceptance(
                    category=category, days=days
                )
                results.append(metrics)

        # Sort by acceptance rate DESC
        return sorted(results, key=lambda m: m.acceptance_rate, reverse=True)

    def calculate_acceptance_by_severity(
        self, days: int = 30
    ) -> list[AcceptanceMetrics]:
        """
        Acceptance rates by severity level.

        Returns acceptance metrics grouped by severity:
        - CRITICAL
        - HIGH
        - MEDIUM
        - LOW

        Args:
            days: Only include feedback from last N days (default: 30)

        Returns:
            List of AcceptanceMetrics sorted by severity level
            (CRITICAL, HIGH, MEDIUM, LOW)
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        severity_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
        results = []

        for severity in severity_order:
            metrics = self.calculate_composite_acceptance(
                severity=severity, days=days
            )
            results.append(metrics)

        return results

    def calculate_acceptance_by_issue_type(
        self, days: int = 30
    ) -> list[AcceptanceMetrics]:
        """
        Acceptance rates by issue type (security vs performance).

        Returns acceptance metrics grouped by:
        - SECURITY
        - PERFORMANCE
        - BEST_PRACTICE

        Args:
            days: Only include feedback from last N days (default: 30)

        Returns:
            List of AcceptanceMetrics sorted by acceptance_rate DESC
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        issue_types = ["SECURITY", "PERFORMANCE", "BEST_PRACTICE"]
        results = []

        for issue_type in issue_types:
            # Map string to FindingCategory enum
            category_enum = None
            if issue_type == "SECURITY":
                category_enum = FindingCategory.SECURITY
            elif issue_type == "PERFORMANCE":
                category_enum = FindingCategory.PERFORMANCE
            elif issue_type == "BEST_PRACTICE":
                category_enum = FindingCategory.BEST_PRACTICE

            if category_enum:
                # Get findings of this type
                findings = (
                    self.db.query(Finding)
                    .filter(
                        Finding.category == category_enum,
                        Finding.created_at >= cutoff_date,
                    )
                    .all()
                )

                if findings:
                    accepted = 0
                    rejected = 0
                    ignored = len(findings)
                    total_confidence = 0.0
                    total_feedback_count = 0
                    fix_count = 0

                    for finding in findings:
                        # Check feedback for this finding
                        feedback = (
                            self.db.query(SuggestionFeedback)
                            .filter(SuggestionFeedback.finding_id == finding.id)
                            .all()
                        )

                        if feedback:
                            ignored -= 1
                            for f in feedback:
                                if f.feedback_type == "ACCEPTED":
                                    accepted += 1
                                    if f.commit_hash:
                                        fix_count += 1
                                elif f.feedback_type == "REJECTED":
                                    rejected += 1

                                # Track confidence from feedback (count all non-null, sum non-zero)
                                if f.confidence is not None:
                                    total_feedback_count += 1
                                    if f.confidence > 0:
                                        total_confidence += f.confidence

                    total = len(findings)
                    acceptance_rate = (
                        accepted / (accepted + rejected)
                        if (accepted + rejected) > 0
                        else 0.0
                    )
                    confidence_avg = (
                        total_confidence / total_feedback_count if total_feedback_count > 0 else 0.0
                    )
                    fix_rate = accepted / total if total > 0 else 0.0

                    metrics = AcceptanceMetrics(
                        finding_category=issue_type,
                        severity="ALL",
                        total=total,
                        accepted=accepted,
                        rejected=rejected,
                        ignored=ignored,
                        acceptance_rate=acceptance_rate,
                        confidence_avg=confidence_avg,
                        fix_rate=fix_rate,
                    )
                    results.append(metrics)

        return sorted(results, key=lambda m: m.acceptance_rate, reverse=True)

    def calculate_composite_acceptance(
        self,
        category: Optional[str] = None,
        severity: Optional[str] = None,
        issue_type: Optional[str] = None,
        days: int = 30,
    ) -> AcceptanceMetrics:
        """
        Calculate acceptance for specific filter combination.

        Filters findings by provided criteria and calculates:
        - Total count
        - Accepted count (feedback_type == ACCEPTED)
        - Rejected count (feedback_type == REJECTED)
        - Ignored count (no feedback)
        - Acceptance rate = accepted / (accepted + rejected)
        - Average confidence
        - Fix rate = actually applied / total

        Args:
            category: Finding category (e.g., "SQL Injection")
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
            issue_type: FindingCategory (SECURITY, PERFORMANCE, BEST_PRACTICE)
            days: Only include feedback from last N days (default: 30)

        Returns:
            AcceptanceMetrics for the filtered findings
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)

        # Start with base query
        query = self.db.query(Finding).filter(Finding.created_at >= cutoff_date)

        # Apply filters
        if category:
            query = query.filter(Finding.title == category)

        if severity:
            # Convert string to enum (handle both uppercase and lowercase)
            severity_enum = None
            severity_upper = severity.upper() if isinstance(severity, str) else ""
            if severity_upper == "CRITICAL":
                severity_enum = FindingSeverity.CRITICAL
            elif severity_upper == "HIGH":
                severity_enum = FindingSeverity.HIGH
            elif severity_upper == "MEDIUM":
                severity_enum = FindingSeverity.MEDIUM
            elif severity_upper == "LOW":
                severity_enum = FindingSeverity.LOW

            if severity_enum:
                query = query.filter(Finding.severity == severity_enum)

        if issue_type:
            # Convert string to enum
            issue_enum = None
            if issue_type == "SECURITY":
                issue_enum = FindingCategory.SECURITY
            elif issue_type == "PERFORMANCE":
                issue_enum = FindingCategory.PERFORMANCE
            elif issue_type == "BEST_PRACTICE":
                issue_enum = FindingCategory.BEST_PRACTICE

            if issue_enum:
                query = query.filter(Finding.category == issue_enum)

        findings = query.all()

        if not findings:
            # Return empty metrics
            return AcceptanceMetrics(
                finding_category=category or "UNKNOWN",
                severity=severity or "ALL",
                total=0,
                accepted=0,
                rejected=0,
                ignored=0,
                acceptance_rate=0.0,
                confidence_avg=0.0,
                fix_rate=0.0,
            )

        # Calculate metrics from findings and their feedback
        accepted = 0
        rejected = 0
        ignored = len(findings)
        total_confidence = 0.0
        total_feedback_count = 0
        fix_count = 0

        for finding in findings:
            # Query feedback for this finding
            feedback_list = (
                self.db.query(SuggestionFeedback)
                .filter(SuggestionFeedback.finding_id == finding.id)
                .all()
            )

            if feedback_list:
                ignored -= 1
                for feedback in feedback_list:
                    if feedback.feedback_type == "ACCEPTED":
                        accepted += 1
                        # Count as "fixed" if it has a commit hash
                        if feedback.commit_hash:
                            fix_count += 1
                    elif feedback.feedback_type == "REJECTED":
                        rejected += 1

                    # Accumulate confidence scores from feedback (count all non-null, sum non-zero)
                    if feedback.confidence is not None:
                        total_feedback_count += 1
                        if feedback.confidence > 0:
                            total_confidence += feedback.confidence

        total = len(findings)
        acceptance_rate = (
            accepted / (accepted + rejected) if (accepted + rejected) > 0 else 0.0
        )
        confidence_avg = (
            total_confidence / total_feedback_count if total_feedback_count > 0 else 0.0
        )
        fix_rate = fix_count / total if total > 0 else 0.0

        return AcceptanceMetrics(
            finding_category=category or "ALL",
            severity=severity or "ALL",
            total=total,
            accepted=accepted,
            rejected=rejected,
            ignored=ignored,
            acceptance_rate=acceptance_rate,
            confidence_avg=confidence_avg,
            fix_rate=fix_rate,
        )

    def get_fixing_timeline(
        self, category: Optional[str] = None, limit: int = 100
    ) -> list[dict]:
        """
        Time from suggestion to actual fix (commit).

        Analyzes how long it takes from when a suggestion is posted to when
        the developer actually fixes the issue (indicated by commit_hash in feedback).

        Args:
            category: Optional category filter (e.g., "SQL Injection")
            limit: Maximum number of records to return

        Returns:
            List of dicts:
            {
                'finding_id': 42,
                'title': 'SQL Injection',
                'severity': 'CRITICAL',
                'suggested_at': datetime,
                'fixed_at': datetime,  # Estimated from commit
                'time_to_fix_hours': float,
                'accepted': bool,
                'confidence': float
            }
            Sorted by time_to_fix DESC (slowest first)
        """
        query = (
            self.db.query(Finding, SuggestionFeedback)
            .outerjoin(SuggestionFeedback)
            .filter(SuggestionFeedback.commit_hash.isnot(None))
        )

        if category:
            query = query.filter(Finding.title == category)

        results = []
        for finding, feedback in query.limit(limit).all():
            if not feedback or not feedback.commit_hash:
                continue

            # Calculate time delta
            suggested_time = finding.created_at
            fixed_time = feedback.created_at  # When feedback was recorded (fix detected)

            if suggested_time and fixed_time:
                time_delta = fixed_time - suggested_time
                hours = time_delta.total_seconds() / 3600

                result = {
                    "finding_id": finding.id,
                    "title": finding.title,
                    "severity": finding.severity.value if finding.severity else None,
                    "suggested_at": suggested_time,
                    "fixed_at": fixed_time,
                    "time_to_fix_hours": hours,
                    "accepted": feedback.feedback_type == "ACCEPTED",
                    "confidence": feedback.confidence or 0.0,
                    "file_path": finding.file_path,
                }
                results.append(result)

        # Sort by time_to_fix DESC (slowest first)
        return sorted(results, key=lambda x: x["time_to_fix_hours"], reverse=True)

    # ============================================================================
    # Aggregation & Persistence
    # ============================================================================

    def persist_metrics_for_category(
        self, category: str, severity: str, issue_type: str, repo_url: str = "default"
    ) -> LearningMetrics:
        """
        Calculate and persist metrics for specific category+severity+type combo.

        Creates or updates a LearningMetrics record in the database with
        acceptance statistics for this category/severity combination.

        Args:
            category: Finding category (e.g., "SQL Injection")
            severity: Severity level (CRITICAL, HIGH, MEDIUM, LOW)
            issue_type: Issue type (SECURITY, PERFORMANCE, BEST_PRACTICE)
            repo_url: Repository URL for tracking project-specific metrics (default: "default")

        Returns:
            LearningMetrics record (new or updated)
        """
        # Calculate acceptance metrics
        metrics = self.calculate_composite_acceptance(
            category=category, severity=severity, issue_type=issue_type
        )

        # Map severity string to enum (handle both uppercase and lowercase)
        severity_enum = None
        severity_upper = severity.upper() if isinstance(severity, str) else ""
        if severity_upper == "CRITICAL":
            severity_enum = FindingSeverity.CRITICAL
        elif severity_upper == "HIGH":
            severity_enum = FindingSeverity.HIGH
        elif severity_upper == "MEDIUM":
            severity_enum = FindingSeverity.MEDIUM
        elif severity_upper == "LOW":
            severity_enum = FindingSeverity.LOW

        # Map issue_type string to FindingCategory enum
        category_enum = None
        if issue_type == "SECURITY":
            category_enum = FindingCategory.SECURITY
        elif issue_type == "PERFORMANCE":
            category_enum = FindingCategory.PERFORMANCE
        elif issue_type == "BEST_PRACTICE":
            category_enum = FindingCategory.BEST_PRACTICE

        # Get or create LearningMetrics record
        record = (
            self.db.query(LearningMetrics)
            .filter(
                LearningMetrics.repo_url == repo_url,
                LearningMetrics.category == category_enum,
                LearningMetrics.severity == severity_enum,
            )
            .first()
        )

        if not record:
            record = LearningMetrics(
                repo_url=repo_url,
                category=category_enum,
                severity=severity_enum,
                total_findings=metrics.total,
                confirmed_findings=metrics.accepted,
                false_positives=metrics.rejected,
                false_negatives=metrics.ignored,
            )
            self.db.add(record)

        # Update metrics
        record.total_findings = metrics.total
        record.confirmed_findings = metrics.accepted
        record.false_positives = metrics.rejected
        record.false_negatives = metrics.ignored

        # Calculate accuracy metrics
        total = metrics.total
        if total > 0:
            confirmed = metrics.accepted
            record.accuracy = (confirmed / total) * 100
            record.precision = (
                confirmed / (confirmed + metrics.rejected)
                if (confirmed + metrics.rejected) > 0
                else 0.0
            )
            record.recall = (
                confirmed / (confirmed + metrics.ignored)
                if (confirmed + metrics.ignored) > 0
                else 0.0
            )

        record.updated_at = datetime.now(timezone.utc)

        self.db.commit()
        return record

    def persist_all_metrics(self) -> int:
        """
        Recalculate and persist metrics for all category+severity combos.

        Automatically discovers all unique category/severity combinations
        in the database and calculates metrics for each.

        Returns:
            Number of metrics records updated/created
        """
        # Get all unique (category, severity) combinations
        combos = (
            self.db.query(Finding.title, Finding.severity)
            .distinct()
            .all()
        )

        count = 0
        for title, severity in combos:
            if title and severity:
                severity_str = severity.value
                self.persist_metrics_for_category(
                    category=title,
                    severity=severity_str,
                    issue_type="SECURITY",  # Simplified for now
                )
                count += 1

        return count

    def get_metrics_by_category(self, category: str) -> list[LearningMetrics]:
        """
        Retrieve persisted metrics for a category.

        Returns all LearningMetrics records for the specified finding category.

        Args:
            category: Finding category to query

        Returns:
            List of LearningMetrics sorted by severity
            (CRITICAL, HIGH, MEDIUM, LOW)
        """
        records = (
            self.db.query(LearningMetrics)
            .filter(LearningMetrics.category == FindingCategory.SECURITY)
            .all()
        )

        # Sort by severity
        severity_order = {
            FindingSeverity.CRITICAL: 0,
            FindingSeverity.HIGH: 1,
            FindingSeverity.MEDIUM: 2,
            FindingSeverity.LOW: 3,
        }
        return sorted(
            records, key=lambda r: severity_order.get(r.severity, 999)
        )

    def get_top_categories_by_acceptance(
        self, limit: int = 10, ascending: bool = False
    ) -> list[tuple[str, float]]:
        """
        Get categories ranked by acceptance rate.

        Returns the top N finding categories sorted by their historical
        acceptance rate.

        Args:
            limit: Number of categories to return
            ascending: If True, return lowest acceptance rates first

        Returns:
            List of (category_name, acceptance_rate) tuples
        """
        metrics = self.calculate_acceptance_by_category()

        if ascending:
            metrics = sorted(
                metrics, key=lambda m: m.acceptance_rate
            )
        else:
            metrics = sorted(
                metrics, key=lambda m: m.acceptance_rate, reverse=True
            )

        return [(m.finding_category, m.acceptance_rate) for m in metrics[:limit]]
