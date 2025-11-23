"""
Optimized Pattern Learning & Detection - Uses SQL aggregation instead of Python loops

Identifies team-specific patterns, anti-patterns, and best practices
from code review feedback to improve suggestion ranking and quality.

Key optimization: Use SQL GROUP BY + aggregation instead of Python loops
to eliminate N+1 query patterns.
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


class PatternLearnerOptimized:
    """
    Optimized pattern learner using SQL aggregation.

    Uses GROUP BY queries to calculate statistics in database instead of Python,
    reducing queries from 101 to ~2 total.
    """

    def __init__(self, db: Session):
        """
        Initialize pattern learner with database session.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    # ============================================================================
    # Optimized Pattern Detection
    # ============================================================================

    def detect_patterns(self, min_occurrences: int = 2) -> List[Dict]:
        """
        Detect recurring patterns using SQL aggregation.

        Uses GROUP BY to calculate statistics in database instead of Python loops.
        Single query instead of 101+ queries.

        Args:
            min_occurrences: Minimum occurrences to consider a pattern (default: 2)

        Returns:
            List of pattern dicts sorted by occurrences DESC
        """
        # Get all findings with feedback counts in single query
        findings = self.db.query(Finding).all()

        if not findings:
            return []

        finding_ids = [f.id for f in findings]

        # Use SQL aggregation to count feedback by type per finding
        feedback_stats = (
            self.db.query(
                SuggestionFeedback.finding_id,
                SuggestionFeedback.feedback_type,
                func.count(SuggestionFeedback.id).label("count"),
            )
            .filter(SuggestionFeedback.finding_id.in_(finding_ids))
            .group_by(SuggestionFeedback.finding_id, SuggestionFeedback.feedback_type)
            .all()
        )

        # Build feedback count map for quick lookup
        feedback_counts = defaultdict(lambda: defaultdict(int))
        for finding_id, feedback_type, count in feedback_stats:
            feedback_counts[finding_id][feedback_type] = count

        # Group findings by title (pattern type)
        pattern_counts = defaultdict(lambda: {
            "findings": [],
            "files_affected": defaultdict(int),
        })

        for finding in findings:
            title = finding.title
            pattern_counts[title]["findings"].append(finding)
            if finding.file_path:
                pattern_counts[title]["files_affected"][finding.file_path] += 1

        # Calculate statistics for each pattern using precomputed feedback counts
        patterns = []
        for title, data in pattern_counts.items():
            if len(data["findings"]) < min_occurrences:
                continue

            # Sum feedback counts (use precomputed stats)
            accepted = 0
            rejected = 0
            for finding in data["findings"]:
                # Count "helpful" and "false_positive" feedback
                accepted += feedback_counts[finding.id].get("helpful", 0)
                rejected += feedback_counts[finding.id].get("false_positive", 0)

            total_feedback = accepted + rejected
            acceptance_rate = accepted / total_feedback if total_feedback > 0 else 0.0

            pattern = {
                "pattern_type": title,
                "occurrences": len(data["findings"]),
                "acceptance_count": accepted,
                "rejection_count": rejected,
                "acceptance_rate": acceptance_rate,
                "files_affected": dict(data["files_affected"]),
                "is_anti_pattern": False,
            }
            patterns.append(pattern)

        # Sort by occurrences descending
        return sorted(patterns, key=lambda p: p["occurrences"], reverse=True)

    # ============================================================================
    # Anti-Pattern Detection
    # ============================================================================

    def detect_anti_patterns(self, rejection_threshold: float = 0.6) -> List[Dict]:
        """
        Detect anti-patterns using optimized pattern detection.

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
        Identify best practices using optimized pattern detection.

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
                avg_severity=0.5,
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
