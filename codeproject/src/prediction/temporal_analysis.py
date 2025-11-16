"""
Temporal Pattern Analysis

Analyzes time-based patterns in code quality to understand
productivity cycles and stress-related quality changes.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List
from enum import Enum

from src.prediction.history_tracker import HistoryDatabase


class TimeOfDay(str, Enum):
    """Time of day categorization."""

    MORNING = "morning"  # 6-12
    AFTERNOON = "afternoon"  # 12-17
    EVENING = "evening"  # 17-21
    NIGHT = "night"  # 21-6


class DayOfWeek(str, Enum):
    """Days of week."""

    MONDAY = "monday"
    TUESDAY = "tuesday"
    WEDNESDAY = "wednesday"
    THURSDAY = "thursday"
    FRIDAY = "friday"
    SATURDAY = "saturday"
    SUNDAY = "sunday"


@dataclass
class TemporalPattern:
    """Represents a time-based pattern in code quality."""

    pattern_type: str  # "day_of_week", "time_of_day", "stress_period"
    time_period: str  # "monday", "morning", "crunch_week"
    issue_count: int
    quality_score: float  # 0-1, higher = better
    sample_size: int  # Number of observations
    confidence: float  # 0-1, how reliable is this pattern


class TemporalAnalyzer:
    """Analyzes temporal patterns in code quality."""

    def __init__(self, history_db: HistoryDatabase):
        """
        Initialize temporal analyzer.

        Args:
            history_db: Historical database
        """
        self.history_db = history_db

    def analyze_day_of_week_pattern(self) -> Dict[str, TemporalPattern]:
        """
        Analyze code quality by day of week.

        Returns:
            Dict of day -> TemporalPattern
        """
        patterns = {}
        snapshots = self.history_db.get_snapshots_since(180)  # 6 months

        day_stats: Dict[str, List[int]] = {day: [] for day in DayOfWeek}

        for snapshot in snapshots:
            day_name = self._get_day_name(snapshot.timestamp)
            # Quality = inverse of issues
            quality = 1.0 - min(1.0, snapshot.issues_found / 10.0)
            day_stats[day_name].append(quality)

        for day, qualities in day_stats.items():
            if qualities:
                avg_quality = sum(qualities) / len(qualities)
                patterns[day] = TemporalPattern(
                    pattern_type="day_of_week",
                    time_period=day,
                    issue_count=int((1.0 - avg_quality) * 10),
                    quality_score=avg_quality,
                    sample_size=len(qualities),
                    confidence=min(1.0, len(qualities) / 20.0),
                )

        return patterns

    def analyze_time_of_day_pattern(self) -> Dict[str, TemporalPattern]:
        """
        Analyze code quality by time of day.

        Returns:
            Dict of time_of_day -> TemporalPattern
        """
        patterns = {}
        snapshots = self.history_db.get_snapshots_since(180)

        time_stats: Dict[str, List[int]] = {time.value: [] for time in TimeOfDay}

        for snapshot in snapshots:
            hour = snapshot.timestamp.hour
            time_period = self._get_time_of_day(hour)
            quality = 1.0 - min(1.0, snapshot.issues_found / 10.0)
            time_stats[time_period.value].append(quality)

        for time_period, qualities in time_stats.items():
            if qualities:
                avg_quality = sum(qualities) / len(qualities)
                patterns[time_period] = TemporalPattern(
                    pattern_type="time_of_day",
                    time_period=time_period,
                    issue_count=int((1.0 - avg_quality) * 10),
                    quality_score=avg_quality,
                    sample_size=len(qualities),
                    confidence=min(1.0, len(qualities) / 20.0),
                )

        return patterns

    def detect_stress_periods(self) -> List[TemporalPattern]:
        """
        Detect when you're under stress (quality drops).

        Returns:
            List of detected stress periods
        """
        stress_periods = []
        snapshots = self.history_db.get_snapshots_since(180)

        # Calculate 2-week rolling average quality
        window_size = 5  # Approximately 5 working days per entry

        for i in range(len(snapshots) - window_size):
            window = snapshots[i : i + window_size]

            avg_issues = sum(s.issues_found for s in window) / window_size
            avg_quality = 1.0 - min(1.0, avg_issues / 10.0)

            # Stress period = quality drops significantly
            if avg_quality < 0.5:  # More than 5 issues per snapshot
                start_date = window[0].timestamp.strftime("%Y-%m-%d")
                end_date = window[-1].timestamp.strftime("%Y-%m-%d")

                stress_periods.append(
                    TemporalPattern(
                        pattern_type="stress_period",
                        time_period=f"{start_date} to {end_date}",
                        issue_count=int(avg_issues),
                        quality_score=avg_quality,
                        sample_size=window_size,
                        confidence=0.8,
                    )
                )

        return stress_periods

    def detect_crunch_time(self) -> List[str]:
        """
        Detect crunch periods (high churn, low quality).

        Returns:
            List of date ranges during crunch
        """
        crunch_periods = []
        snapshots = self.history_db.get_snapshots_since(180)

        for snapshot in snapshots:
            if len(snapshot.files_changed) > 20 and snapshot.issues_found > 5:
                # Many files changed + many issues = crunch
                crunch_periods.append(snapshot.timestamp.strftime("%Y-%m-%d"))

        return crunch_periods

    def get_best_coding_time(self) -> Dict[str, float]:
        """
        Identify when you produce the best code.

        Returns:
            Dict of time_period -> quality_score
        """
        day_patterns = self.analyze_day_of_week_pattern()
        time_patterns = self.analyze_time_of_day_pattern()

        best_times = {}

        # Find best days
        best_day = max(day_patterns.items(), key=lambda x: x[1].quality_score)
        best_times[f"best_day"] = best_day[1].quality_score
        best_times[f"best_day_name"] = best_day[0]

        # Find best times of day
        best_time = max(time_patterns.items(), key=lambda x: x[1].quality_score)
        best_times[f"best_time"] = best_time[1].quality_score
        best_times[f"best_time_name"] = best_time[0]

        return best_times

    def get_worst_coding_time(self) -> Dict[str, float]:
        """
        Identify when you produce the worst code.

        Returns:
            Dict of time_period -> quality_score
        """
        day_patterns = self.analyze_day_of_week_pattern()
        time_patterns = self.analyze_time_of_day_pattern()

        worst_times = {}

        # Find worst days
        worst_day = min(day_patterns.items(), key=lambda x: x[1].quality_score)
        worst_times[f"worst_day"] = worst_day[1].quality_score
        worst_times[f"worst_day_name"] = worst_day[0]

        # Find worst times of day
        worst_time = min(time_patterns.items(), key=lambda x: x[1].quality_score)
        worst_times[f"worst_time"] = worst_time[1].quality_score
        worst_times[f"worst_time_name"] = worst_time[0]

        return worst_times

    def predict_quality_for_time(self, day: DayOfWeek, time: TimeOfDay) -> float:
        """
        Predict code quality for a given time.

        Args:
            day: Day of week
            time: Time of day

        Returns:
            Expected quality score (0-1)
        """
        day_patterns = self.analyze_day_of_week_pattern()
        time_patterns = self.analyze_time_of_day_pattern()

        day_quality = day_patterns.get(day.value, TemporalPattern(
            pattern_type="day",
            time_period=day.value,
            issue_count=0,
            quality_score=0.5,
            sample_size=0,
            confidence=0.0,
        )).quality_score

        time_quality = time_patterns.get(time.value, TemporalPattern(
            pattern_type="time",
            time_period=time.value,
            issue_count=0,
            quality_score=0.5,
            sample_size=0,
            confidence=0.0,
        )).quality_score

        # Average of day and time patterns
        return (day_quality + time_quality) / 2

    def get_productivity_insights(self) -> List[str]:
        """
        Get personalized productivity insights.

        Returns:
            List of insights
        """
        insights = []

        # Day of week insights
        day_patterns = self.analyze_day_of_week_pattern()
        best_day = max(day_patterns.items(), key=lambda x: x[1].quality_score)
        worst_day = min(day_patterns.items(), key=lambda x: x[1].quality_score)

        insights.append(f"Best day: {best_day[0]} ({best_day[1].quality_score:.0%} quality)")
        insights.append(f"Worst day: {worst_day[0]} ({worst_day[1].quality_score:.0%} quality)")

        # Time of day insights
        time_patterns = self.analyze_time_of_day_pattern()
        best_time = max(time_patterns.items(), key=lambda x: x[1].quality_score)
        worst_time = min(time_patterns.items(), key=lambda x: x[1].quality_score)

        insights.append(f"Best time: {best_time[0]} ({best_time[1].quality_score:.0%} quality)")
        insights.append(f"Worst time: {worst_time[0]} ({worst_time[1].quality_score:.0%} quality)")

        # Stress periods
        stress = self.detect_stress_periods()
        if stress:
            insights.append(f"Detected {len(stress)} stress periods in past 6 months")

        return insights

    def _get_day_name(self, dt: datetime) -> str:
        """Get day of week name."""
        days = [
            "monday",
            "tuesday",
            "wednesday",
            "thursday",
            "friday",
            "saturday",
            "sunday",
        ]
        return days[dt.weekday()]

    def _get_time_of_day(self, hour: int) -> TimeOfDay:
        """Categorize hour into time of day."""
        if 6 <= hour < 12:
            return TimeOfDay.MORNING
        elif 12 <= hour < 17:
            return TimeOfDay.AFTERNOON
        elif 17 <= hour < 21:
            return TimeOfDay.EVENING
        else:
            return TimeOfDay.NIGHT
