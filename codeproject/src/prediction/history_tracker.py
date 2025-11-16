"""
Historical Tracking System

Captures code quality snapshots per commit and maintains historical database
for trend analysis and pattern learning.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class CodeQualitySnapshot:
    """Represents code quality at a point in time (per commit)."""

    commit_sha: str
    timestamp: datetime
    files_changed: List[str]  # Files modified in this commit
    metrics_per_file: Dict[str, Dict]  # File -> {complexity, lines, etc.}
    complexity_trend: Dict[str, float]  # File -> complexity score
    test_coverage_trend: Dict[str, float]  # File -> coverage %
    issues_found: int  # Issues in this commit
    production_bugs: int  # Bugs that reached production
    fixed_issues: int  # Issues fixed in this commit
    architectural_metrics: Dict[str, float]  # density, coupling, etc.
    coupling_trend: Dict[str, float]  # Module -> coupling score


class HistoryDatabase:
    """Maintains and queries historical snapshots."""

    def __init__(self, db_path: Optional[str] = None):
        """
        Initialize history database.

        Args:
            db_path: Optional path to persist history
        """
        self.snapshots: List[CodeQualitySnapshot] = []
        self.db_path = Path(db_path) if db_path else None
        self._load_history()

    def add_snapshot(self, snapshot: CodeQualitySnapshot) -> None:
        """
        Add a code quality snapshot.

        Args:
            snapshot: Snapshot to add
        """
        self.snapshots.append(snapshot)
        self.snapshots.sort(key=lambda s: s.timestamp)
        if self.db_path:
            self._save_history()

    def get_snapshots_since(self, days_ago: int) -> List[CodeQualitySnapshot]:
        """
        Get all snapshots from the past N days.

        Args:
            days_ago: Number of days to look back

        Returns:
            List of snapshots within the time range
        """
        from datetime import timedelta

        cutoff_time = datetime.now() - timedelta(days=days_ago)
        return [s for s in self.snapshots if s.timestamp >= cutoff_time]

    def get_file_history(self, file_path: str) -> List[Dict]:
        """
        Get historical metrics for a specific file.

        Args:
            file_path: Path to file

        Returns:
            List of historical data points for the file
        """
        history = []
        for snapshot in self.snapshots:
            if file_path in snapshot.metrics_per_file:
                history.append({
                    "timestamp": snapshot.timestamp,
                    "commit_sha": snapshot.commit_sha,
                    "metrics": snapshot.metrics_per_file[file_path],
                    "complexity": snapshot.complexity_trend.get(file_path, 0.0),
                    "coverage": snapshot.test_coverage_trend.get(file_path, 0.0),
                })
        return history

    def get_file_churn_rate(self, file_path: str, days: int = 30) -> float:
        """
        Calculate how often a file is modified (churn rate).

        Args:
            file_path: Path to file
            days: Look back period in days

        Returns:
            Number of commits touching this file in the period
        """
        recent = self.get_snapshots_since(days)
        return sum(1 for s in recent if file_path in s.files_changed)

    def get_complexity_trend(self, file_path: str, days: int = 90) -> List[float]:
        """
        Get complexity trend over time for a file.

        Args:
            file_path: Path to file
            days: Look back period

        Returns:
            List of complexity scores over time
        """
        history = self.get_file_history(file_path)
        recent = [h for h in history if (datetime.now() - h["timestamp"]).days <= days]
        return [h["complexity"] for h in recent]

    def get_coverage_trend(self, file_path: str, days: int = 90) -> List[float]:
        """
        Get test coverage trend for a file.

        Args:
            file_path: Path to file
            days: Look back period

        Returns:
            List of coverage percentages over time
        """
        history = self.get_file_history(file_path)
        recent = [h for h in history if (datetime.now() - h["timestamp"]).days <= days]
        return [h["coverage"] for h in recent]

    def get_bug_history(self, file_path: str, days: int = 180) -> int:
        """
        Get number of bugs found in a file historically.

        Args:
            file_path: Path to file
            days: Look back period

        Returns:
            Number of bugs found in the file
        """
        recent = self.get_snapshots_since(days)
        bug_count = 0
        for snapshot in recent:
            if file_path in snapshot.files_changed:
                # Estimate: proportion of issues/bugs in this file
                if file_path in snapshot.metrics_per_file:
                    bug_count += max(0, snapshot.production_bugs // max(1, len(snapshot.files_changed)))
        return bug_count

    def get_most_changed_files(self, days: int = 30, limit: int = 10) -> List[tuple]:
        """
        Get most frequently modified files.

        Args:
            days: Look back period
            limit: Maximum number of files to return

        Returns:
            List of (file_path, change_count) tuples
        """
        file_counts: Dict[str, int] = {}
        recent = self.get_snapshots_since(days)

        for snapshot in recent:
            for file_path in snapshot.files_changed:
                file_counts[file_path] = file_counts.get(file_path, 0) + 1

        return sorted(file_counts.items(), key=lambda x: x[1], reverse=True)[:limit]

    def get_quality_trajectory(self, days: int = 90) -> Dict[str, list]:
        """
        Get overall quality metrics trajectory.

        Args:
            days: Look back period

        Returns:
            Dict with trajectory of key metrics
        """
        recent = self.get_snapshots_since(days)

        return {
            "timestamps": [s.timestamp for s in recent],
            "issues_found": [s.issues_found for s in recent],
            "production_bugs": [s.production_bugs for s in recent],
            "avg_complexity": [
                sum(s.complexity_trend.values()) / max(1, len(s.complexity_trend))
                for s in recent
            ],
            "avg_coverage": [
                sum(s.test_coverage_trend.values()) / max(1, len(s.test_coverage_trend))
                for s in recent
            ],
        }

    def _load_history(self) -> None:
        """Load historical snapshots from disk if available."""
        if not self.db_path or not self.db_path.exists():
            return

        # Placeholder for loading from database
        # In production, this would load from a database or JSON file
        pass

    def _save_history(self) -> None:
        """Save historical snapshots to disk."""
        if not self.db_path:
            return

        # Placeholder for saving to database
        # In production, this would persist to a database or JSON file
        pass
