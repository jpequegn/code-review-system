"""
Pattern Learning & Detection - Learn recurring patterns and anti-patterns

Identifies team-specific patterns, anti-patterns, and best practices
from code review feedback to improve suggestion ranking and quality.
"""

import json
from typing import Optional, List, Dict
from datetime import datetime, timezone
from collections import defaultdict
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database import (
    Finding,
    SuggestionFeedback,
    PatternMetrics,
)


class PatternLearner:
    """
    Learn and detect patterns from code review findings and feedback.

    Problem: Not all code issues are equally important. Some patterns appear
    frequently and are consistently accepted (best practices), others are
    frequently rejected (anti-patterns), and some are new or context-specific.

    Solution: Track pattern occurrence, acceptance/rejection rates, and
    affected files to identify team practices and improve prioritization.

    Example:
        learner = PatternLearner(db_session)

        # Detect all patterns from feedback history
        patterns = learner.detect_patterns(min_occurrences=2)

        # Identify anti-patterns (mostly rejected)
        anti_patterns = learner.detect_anti_patterns(rejection_threshold=0.6)

        # Find best practices (mostly accepted)
        best_practices = learner.identify_best_practices(acceptance_threshold=0.8)

        # Persist patterns to database for ranking and analysis
        learner.persist_all_patterns(patterns)
    """

    def __init__(self, db: Session):
        """
        Initialize pattern learner with database session.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    # ============================================================================
    # Pattern Detection
    # ============================================================================

    def detect_patterns(self, min_occurrences: int = 2) -> List[Dict]:
        """
        Detect recurring patterns from finding titles.

        Aggregates findings by title to identify frequently occurring issues.
        Returns pattern info including:
        - pattern_type: Finding title
        - occurrences: Number of findings with this title
        - acceptance_count: Number accepted
        - rejection_count: Number rejected
        - acceptance_rate: Accepted / (Accepted + Rejected)
        - files_affected: Dict of {filename: count}

        Args:
            min_occurrences: Minimum occurrences to consider a pattern (default: 2)

        Returns:
            List of pattern dicts sorted by occurrences DESC
        """
        # Group findings by title
        pattern_counts = defaultdict(lambda: {
            "findings": [],
            "files_affected": defaultdict(int),
        })

        findings = self.db.query(Finding).all()

        for finding in findings:
            title = finding.title
            pattern_counts[title]["findings"].append(finding)
            if finding.file_path:
                pattern_counts[title]["files_affected"][finding.file_path] += 1

        # Calculate statistics for each pattern
        patterns = []
        for title, data in pattern_counts.items():
            if len(data["findings"]) < min_occurrences:
                continue

            # Count feedback
            accepted = 0
            rejected = 0
            for finding in data["findings"]:
                feedbacks = self.db.query(SuggestionFeedback).filter(
                    SuggestionFeedback.finding_id == finding.id
                ).all()
                for feedback in feedbacks:
                    if feedback.feedback_type == "ACCEPTED":
                        accepted += 1
                    elif feedback.feedback_type == "REJECTED":
                        rejected += 1

            total_feedback = accepted + rejected
            acceptance_rate = accepted / total_feedback if total_feedback > 0 else 0.0

            pattern = {
                "pattern_type": title,
                "occurrences": len(data["findings"]),
                "acceptance_count": accepted,
                "rejection_count": rejected,
                "acceptance_rate": acceptance_rate,
                "files_affected": dict(data["files_affected"]),
                "is_anti_pattern": False,  # Will be set by detect_anti_patterns
            }
            patterns.append(pattern)

        # Sort by occurrences descending
        return sorted(patterns, key=lambda p: p["occurrences"], reverse=True)

    # ============================================================================
    # Anti-Pattern Detection
    # ============================================================================

    def detect_anti_patterns(self, rejection_threshold: float = 0.6) -> List[Dict]:
        """
        Detect anti-patterns (patterns developers actively reject).

        An anti-pattern has a rejection rate above the threshold, indicating
        the team consistently rejects this type of suggestion.

        Args:
            rejection_threshold: Rejection rate threshold (0.0-1.0, default: 0.6)

        Returns:
            List of anti-patterns sorted by rejection rate DESC
        """
        patterns = self.detect_patterns()
        anti_patterns = []

        for pattern in patterns:
            total_feedback = pattern["acceptance_count"] + pattern["rejection_count"]
            if total_feedback == 0:
                continue

            rejection_rate = pattern["rejection_count"] / total_feedback
            if rejection_rate >= rejection_threshold:
                pattern["is_anti_pattern"] = True
                anti_patterns.append(pattern)

        return sorted(anti_patterns, key=lambda p: (
            (p["rejection_count"] / (p["acceptance_count"] + p["rejection_count"]))
            if (p["acceptance_count"] + p["rejection_count"]) > 0 else 0
        ), reverse=True)

    # ============================================================================
    # Pattern Ranking
    # ============================================================================

    def rank_patterns(
        self,
        patterns: List[Dict],
        by: str = "occurrences",
        descending: bool = True,
    ) -> List[Dict]:
        """
        Rank patterns by specified metric.

        Args:
            patterns: List of pattern dicts
            by: Ranking metric: 'occurrences', 'acceptance_rate', 'rejection_rate'
            descending: Sort descending (True) or ascending (False)

        Returns:
            Sorted list of patterns
        """
        if not patterns:
            return []

        if by == "occurrences":
            key_func = lambda p: p["occurrences"]
        elif by == "acceptance_rate":
            key_func = lambda p: p["acceptance_rate"]
        elif by == "rejection_rate":
            key_func = lambda p: 1.0 - p["acceptance_rate"] if p["acceptance_rate"] is not None else 0.0
        else:
            key_func = lambda p: p.get(by, 0)

        return sorted(patterns, key=key_func, reverse=descending)

    # ============================================================================
    # Best Practice Identification
    # ============================================================================

    def identify_best_practices(
        self, acceptance_threshold: float = 0.8
    ) -> List[Dict]:
        """
        Identify best practices (patterns with high, consistent acceptance).

        A best practice has:
        - Acceptance rate above threshold
        - Minimum occurrences to establish as pattern
        - Consistent acceptance across reviews

        Args:
            acceptance_threshold: Minimum acceptance rate (0.0-1.0, default: 0.8)

        Returns:
            List of best practices sorted by acceptance rate DESC
        """
        patterns = self.detect_patterns()
        best_practices = []

        for pattern in patterns:
            if pattern["acceptance_rate"] >= acceptance_threshold:
                best_practices.append(pattern)

        return sorted(best_practices, key=lambda p: p["acceptance_rate"], reverse=True)

    # ============================================================================
    # Persistence
    # ============================================================================

    def persist_pattern(self, pattern: Dict) -> PatternMetrics:
        """
        Save or update a pattern in the database.

        Creates or updates PatternMetrics record with pattern information.

        Args:
            pattern: Pattern dict with pattern_type, occurrences, etc.

        Returns:
            PatternMetrics record
        """
        # Get or create record
        record = (
            self.db.query(PatternMetrics)
            .filter(PatternMetrics.pattern_type == pattern["pattern_type"])
            .first()
        )

        files_affected_json = json.dumps(pattern.get("files_affected", {}))

        if not record:
            record = PatternMetrics(
                pattern_hash=hash(pattern["pattern_type"]),
                pattern_type=pattern["pattern_type"],
                occurrences=pattern.get("occurrences", 0),
                files_affected=files_affected_json,
                avg_severity=0.5,  # Default: calculate from findings if needed
                acceptance_rate=pattern.get("acceptance_rate", 0.5),
                fix_count=pattern.get("acceptance_count", 0),
                anti_pattern=pattern.get("is_anti_pattern", False),
                team_prevalence=self._calculate_prevalence(pattern.get("occurrences", 0)),
                recommended_fix="",
                created_at=datetime.now(timezone.utc),
            )
            self.db.add(record)
        else:
            # Update existing
            record.occurrences = pattern.get("occurrences", record.occurrences)
            record.files_affected = files_affected_json
            record.acceptance_rate = pattern.get("acceptance_rate", record.acceptance_rate)
            record.fix_count = pattern.get("acceptance_count", record.fix_count)
            record.anti_pattern = pattern.get("is_anti_pattern", record.anti_pattern)
            record.team_prevalence = self._calculate_prevalence(pattern.get("occurrences", 0))
            record.last_updated = datetime.now(timezone.utc)

        self.db.commit()
        return record

    def persist_all_patterns(self, patterns: List[Dict]) -> int:
        """
        Persist all patterns to database.

        Args:
            patterns: List of pattern dicts

        Returns:
            Count of patterns persisted
        """
        count = 0
        for pattern in patterns:
            self.persist_pattern(pattern)
            count += 1
        return count

    # ============================================================================
    # Helpers
    # ============================================================================

    def _calculate_prevalence(self, occurrences: int) -> str:
        """
        Calculate prevalence level based on occurrence count.

        Thresholds:
        - rare: 1-3 occurrences
        - occasional: 4-10 occurrences
        - common: 11+ occurrences

        Args:
            occurrences: Number of occurrences

        Returns:
            Prevalence level: 'rare', 'occasional', or 'common'
        """
        if occurrences <= 3:
            return "rare"
        elif occurrences <= 10:
            return "occasional"
        else:
            return "common"

    def get_pattern_report(self) -> Dict:
        """
        Generate comprehensive pattern report.

        Includes:
        - Total patterns detected
        - Best practices and anti-patterns
        - Top patterns by occurrence
        - Pattern statistics

        Returns:
            Report dict with pattern analysis
        """
        all_patterns = self.detect_patterns(min_occurrences=1)
        anti_patterns = self.detect_anti_patterns()
        best_practices = self.identify_best_practices()

        total_findings = self.db.query(Finding).count()
        total_patterns = len(all_patterns)

        return {
            "total_findings": total_findings,
            "total_patterns": total_patterns,
            "best_practices_count": len(best_practices),
            "anti_patterns_count": len(anti_patterns),
            "top_patterns": self.rank_patterns(all_patterns)[:5],
            "best_practices": best_practices,
            "anti_patterns": anti_patterns,
            "report_generated": datetime.now(timezone.utc).isoformat(),
        }
