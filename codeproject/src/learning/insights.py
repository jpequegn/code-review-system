"""
Insights Generation Engine - Team metrics, trends, and learning paths

Generates actionable insights from feedback and learning data:
- Team KPIs and acceptance rates
- Vulnerability trend analysis
- Common anti-patterns detection
- Learning paths with improvement opportunities
- ROI analysis (hours saved from auto-fixes)
"""

import json
import math
from typing import Optional, List, Dict, Tuple
from datetime import datetime, timezone, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc

from src.database import (
    Finding,
    SuggestionFeedback,
    LearningMetrics,
    PatternMetrics,
    TeamMetrics,
    LearningPath,
    InsightsTrend,
    FindingCategory,
    FindingSeverity,
)


class InsightsGenerator:
    """
    Generate team insights from learning data.

    Problem: Teams need actionable insights to understand their security posture,
    improvement trends, and learning opportunities. Management needs visibility
    into team performance and ROI from automated code review.

    Solution: Aggregate findings, feedback, and learning metrics to compute:
    - Team KPIs (acceptance rate, fix time, ROI)
    - Vulnerability trends over time
    - Common anti-patterns
    - Prioritized learning paths
    - ROI analysis showing hours saved

    Example:
        insights = InsightsGenerator(db_session)

        # Get team metrics
        metrics = insights.calculate_team_metrics(repo_url="https://github.com/org/repo")

        # Get vulnerability trends
        trends = insights.analyze_trends(repo_url=repo_url, weeks=12)

        # Get learning recommendations
        paths = insights.generate_learning_paths(repo_url=repo_url, top_n=5)

        # Get ROI analysis
        roi = insights.calculate_roi(repo_url=repo_url)
    """

    def __init__(self, db: Session):
        """
        Initialize insights generator.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db

    # ============================================================================
    # Team Metrics Calculation
    # ============================================================================

    def calculate_team_metrics(
        self, repo_url: str, period_days: int = 30
    ) -> Dict:
        """
        Calculate aggregated team metrics.

        Computes acceptance rates, average fix time, ROI, and top vulnerabilities.

        Args:
            repo_url: Repository URL to analyze
            period_days: Days back to analyze (default: 30)

        Returns:
            Dict with team metrics including:
            - total_findings, accepted_findings, rejected_findings
            - acceptance_rate, avg_fix_time, roi_hours_saved
            - top_vulnerabilities, trend_direction, trend_strength
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=period_days)

        # Get all feedbacks from the period
        feedbacks = (
            self.db.query(SuggestionFeedback)
            .join(Finding)
            .filter(
                SuggestionFeedback.created_at >= cutoff_date,
            )
            .all()
        )

        if not feedbacks:
            # Return default metrics
            return {
                "total_findings": 0,
                "accepted_findings": 0,
                "rejected_findings": 0,
                "ignored_findings": 0,
                "acceptance_rate": 0.0,
                "avg_fix_time": 0.0,
                "roi_hours_saved": 0.0,
                "roi_percentage": 0.0,
                "top_vulnerabilities": [],
                "trend_direction": "stable",
                "trend_strength": 0.0,
            }

        # Count feedback types
        total = len(feedbacks)
        accepted = len([f for f in feedbacks if f.feedback_type == "helpful"])
        rejected = len([f for f in feedbacks if f.feedback_type == "false_positive"])
        ignored = total - accepted - rejected

        acceptance_rate = (accepted / total * 100) if total > 0 else 0.0

        # Calculate average fix time from learning metrics
        learning_metrics = self.db.query(LearningMetrics).all()
        avg_fix_times = [m.avg_time_to_fix for m in learning_metrics if m.avg_time_to_fix]
        avg_fix_time = sum(avg_fix_times) / len(avg_fix_times) if avg_fix_times else 0.0

        # Calculate ROI (accepted findings × avg time saved)
        roi_hours_saved = accepted * avg_fix_time
        roi_percentage = (roi_hours_saved / (total * avg_fix_time * 100)) if total > 0 and avg_fix_time > 0 else 0.0

        # Get top vulnerabilities
        vulnerability_counts = {}
        for feedback in feedbacks:
            finding = feedback.finding
            if finding:
                key = f"{finding.title}"
                vulnerability_counts[key] = vulnerability_counts.get(key, 0) + 1

        top_vulns = sorted(
            vulnerability_counts.items(), key=lambda x: x[1], reverse=True
        )[:5]
        top_vulnerabilities = [{"type": t, "count": c} for t, c in top_vulns]

        # Calculate trend direction (comparing to previous period)
        trend_direction = self._calculate_trend_direction(
            repo_url, acceptance_rate, period_days
        )
        trend_strength = self._calculate_trend_strength(acceptance_rate, period_days)

        return {
            "total_findings": total,
            "accepted_findings": accepted,
            "rejected_findings": rejected,
            "ignored_findings": ignored,
            "acceptance_rate": acceptance_rate,
            "avg_fix_time": avg_fix_time,
            "roi_hours_saved": roi_hours_saved,
            "roi_percentage": roi_percentage,
            "top_vulnerabilities": top_vulnerabilities,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength,
        }

    def _calculate_trend_direction(
        self, repo_url: str, current_rate: float, period_days: int
    ) -> str:
        """
        Calculate trend direction by comparing current to previous period.

        Args:
            repo_url: Repository URL
            current_rate: Current acceptance rate
            period_days: Analysis period length

        Returns:
            "improving", "declining", or "stable"
        """
        # Get previous period metrics
        previous_cutoff = datetime.now(timezone.utc) - timedelta(days=period_days * 2)
        previous_metrics = (
            self.db.query(TeamMetrics)
            .filter(
                TeamMetrics.repo_url == repo_url,
                TeamMetrics.created_at >= previous_cutoff,
            )
            .order_by(desc(TeamMetrics.created_at))
            .first()
        )

        if not previous_metrics:
            return "stable"

        diff = current_rate - previous_metrics.acceptance_rate
        threshold = 5.0  # 5% change threshold

        if diff > threshold:
            return "improving"
        elif diff < -threshold:
            return "declining"
        else:
            return "stable"

    def _calculate_trend_strength(self, acceptance_rate: float, period_days: int) -> float:
        """
        Calculate trend strength (0-1) based on acceptance rate.

        Args:
            acceptance_rate: Current acceptance rate (0-100)
            period_days: Analysis period

        Returns:
            Trend strength (0.0-1.0)
        """
        # Normalize acceptance rate to 0-1, compute distance from 50%
        normalized = acceptance_rate / 100.0
        distance_from_neutral = abs(normalized - 0.5)
        strength = min(distance_from_neutral * 2, 1.0)
        return strength

    # ============================================================================
    # Trend Analysis
    # ============================================================================

    def analyze_trends(
        self, repo_url: str, weeks: int = 12
    ) -> List[Dict]:
        """
        Analyze vulnerability trends over time.

        Tracks most common vulnerability types, frequency over weeks/months,
        and identifies improvement/regression patterns.

        Args:
            repo_url: Repository URL to analyze
            weeks: Number of weeks to analyze (default: 12)

        Returns:
            List of trend data dicts:
            [
                {
                    "week": "2025-W47",
                    "findings_count": 15,
                    "acceptance_rate": 68.0,
                    "critical": 2,
                    "high": 5,
                    "medium": 6,
                    "low": 2,
                    "top_category": "security"
                },
                ...
            ]
        """
        trends = []
        now = datetime.now(timezone.utc)

        for i in range(weeks):
            # Calculate week range
            week_end = now - timedelta(weeks=i)
            week_start = week_end - timedelta(weeks=1)

            # Query findings from this week
            week_feedbacks = (
                self.db.query(SuggestionFeedback)
                .join(Finding)
                .filter(
                    SuggestionFeedback.created_at >= week_start,
                    SuggestionFeedback.created_at <= week_end,
                )
                .all()
            )

            if not week_feedbacks:
                continue

            # Calculate metrics for the week
            total = len(week_feedbacks)
            accepted = len([f for f in week_feedbacks if f.feedback_type == "helpful"])
            acceptance_rate = (accepted / total * 100) if total > 0 else 0.0

            # Count by severity
            findings = [f.finding for f in week_feedbacks]
            critical = len([f for f in findings if f and f.severity == FindingSeverity.CRITICAL])
            high = len([f for f in findings if f and f.severity == FindingSeverity.HIGH])
            medium = len([f for f in findings if f and f.severity == FindingSeverity.MEDIUM])
            low = len([f for f in findings if f and f.severity == FindingSeverity.LOW])

            # Top category
            categories = {}
            for f in findings:
                if f:
                    cat = f.category.value if f.category else "unknown"
                    categories[cat] = categories.get(cat, 0) + 1

            top_category = max(categories, key=categories.get) if categories else "unknown"

            # Week identifier (ISO format: YYYY-Www)
            week_num = week_end.isocalendar()[1]
            year = week_end.year
            week_id = f"{year}-W{week_num:02d}"

            trends.append({
                "week": week_id,
                "findings_count": total,
                "acceptance_rate": acceptance_rate,
                "critical": critical,
                "high": high,
                "medium": medium,
                "low": low,
                "top_category": top_category,
            })

        return sorted(trends, key=lambda x: x["week"])

    def detect_anti_patterns(self, repo_url: str) -> List[Dict]:
        """
        Detect common anti-patterns (frequently rejected findings).

        Args:
            repo_url: Repository URL

        Returns:
            List of anti-patterns:
            [
                {
                    "pattern": "Over-conservative suggestions",
                    "occurrences": 23,
                    "rejection_rate": 78.0,
                    "category": "security"
                },
                ...
            ]
        """
        # Query patterns
        patterns = self.db.query(PatternMetrics).filter(
            PatternMetrics.anti_pattern == True
        ).all()

        anti_patterns = []
        for pattern in patterns:
            # Calculate rejection rate: (1 - acceptance_rate)
            rejection_rate = 100.0 - pattern.acceptance_rate

            anti_patterns.append({
                "pattern": pattern.pattern_type,
                "occurrences": pattern.occurrences,
                "rejection_rate": rejection_rate,
                "prevalence": pattern.team_prevalence,
            })

        # Sort by rejection rate
        return sorted(anti_patterns, key=lambda x: x["rejection_rate"], reverse=True)

    # ============================================================================
    # Learning Path Generation
    # ============================================================================

    def generate_learning_paths(
        self, repo_url: str, top_n: int = 5
    ) -> List[Dict]:
        """
        Generate prioritized learning paths for team improvement.

        Identifies top improvement areas with:
        - Current vs potential acceptance rates
        - Estimated hours that could be saved
        - Recommended resources
        - Priority scoring

        Args:
            repo_url: Repository URL
            top_n: Number of top paths to return (default: 5)

        Returns:
            List of learning paths:
            [
                {
                    "rank": 1,
                    "vulnerability_type": "SQL Injection",
                    "category": "security",
                    "current_rate": 45.0,
                    "potential_rate": 85.0,
                    "improvement_potential": 40.0,
                    "occurrences": 23,
                    "hours_saved": 46.0,
                    "priority_score": 0.92,
                    "resources": [...]
                },
                ...
            ]
        """
        # Get all vulnerability types with feedback
        vulnerability_data = {}

        feedbacks = self.db.query(SuggestionFeedback).join(Finding).all()

        for feedback in feedbacks:
            finding = feedback.finding
            if not finding:
                continue

            key = finding.title
            if key not in vulnerability_data:
                vulnerability_data[key] = {
                    "category": finding.category,
                    "total": 0,
                    "accepted": 0,
                    "occurrences": 0,
                }

            vulnerability_data[key]["total"] += 1
            if feedback.feedback_type == "helpful":
                vulnerability_data[key]["accepted"] += 1
            vulnerability_data[key]["occurrences"] += 1

        # Calculate metrics for each vulnerability
        learning_paths = []

        for vuln_type, data in vulnerability_data.items():
            if data["total"] == 0:
                continue

            current_rate = (data["accepted"] / data["total"]) * 100

            # Estimate potential rate (80% if we improve)
            potential_rate = min(current_rate + 40, 95.0)
            improvement_potential = potential_rate - current_rate

            # Get avg fix time
            learning_metric = (
                self.db.query(LearningMetrics)
                .filter(LearningMetrics.category == data["category"])
                .first()
            )

            avg_time = learning_metric.avg_time_to_fix if learning_metric and learning_metric.avg_time_to_fix else 2.0

            # Hours that could be saved
            additional_fixed = int(data["total"] * (improvement_potential / 100))
            hours_saved = additional_fixed * avg_time

            # Priority score (0-1): combines improvement potential and impact
            impact_score = min(data["occurrences"] / 50.0, 1.0)  # Normalize to 1
            improvement_score = improvement_potential / 100.0
            priority_score = (impact_score * 0.6) + (improvement_score * 0.4)

            # Resources (placeholder)
            resources = self._get_learning_resources(vuln_type, data["category"])

            learning_paths.append({
                "rank": 0,  # Will be set after sorting
                "vulnerability_type": vuln_type,
                "category": data["category"].value,
                "current_rate": current_rate,
                "potential_rate": potential_rate,
                "improvement_potential": improvement_potential,
                "occurrences": data["occurrences"],
                "hours_saved": hours_saved,
                "priority_score": priority_score,
                "resources": resources,
            })

        # Sort by priority score and rank
        sorted_paths = sorted(
            learning_paths, key=lambda x: x["priority_score"], reverse=True
        )[:top_n]

        for rank, path in enumerate(sorted_paths, 1):
            path["rank"] = rank

        return sorted_paths

    def _get_learning_resources(
        self, vulnerability_type: str, category
    ) -> List[str]:
        """
        Get recommended learning resources for a vulnerability type.

        Args:
            vulnerability_type: Type of vulnerability
            category: Finding category

        Returns:
            List of resource URLs or names
        """
        # Placeholder resource mapping
        resources_map = {
            "SQL Injection": [
                "OWASP SQL Injection Prevention Cheat Sheet",
                "Prepared Statements Tutorial",
                "SQL Parameterization Best Practices",
            ],
            "Buffer Overflow": [
                "Memory Safety in Rust",
                "C Buffer Overflow Protection",
                "ASAN - Address Sanitizer Guide",
            ],
            "XSS": [
                "OWASP XSS Prevention",
                "Content Security Policy Guide",
                "DOM-based XSS Prevention",
            ],
        }

        return resources_map.get(vulnerability_type, [
            "OWASP Top 10 Guide",
            "CWE/SANS Top 25",
            f"Secure Coding for {category.value if hasattr(category, 'value') else category}",
        ])

    # ============================================================================
    # ROI Analysis
    # ============================================================================

    def calculate_roi(self, repo_url: str, period_days: int = 90) -> Dict:
        """
        Calculate return on investment from automated code review.

        Estimates hours saved from auto-fixes and suggestion acceptance,
        with monetary value assuming hourly rates.

        Args:
            repo_url: Repository URL
            period_days: Analysis period

        Returns:
            Dict with ROI metrics:
            {
                "period_days": 90,
                "total_findings_reviewed": 156,
                "suggestions_accepted": 98,
                "auto_fixes_applied": 45,
                "hours_saved_from_suggestions": 98.0,
                "hours_saved_from_autofix": 180.0,
                "total_hours_saved": 278.0,
                "monetary_value": 33360.0,  # assuming $120/hour
                "roi_percentage": 34.5,
            }
        """
        cutoff_date = datetime.now(timezone.utc) - timedelta(days=period_days)

        # Query findings and feedback from period
        feedbacks = (
            self.db.query(SuggestionFeedback)
            .join(Finding)
            .filter(SuggestionFeedback.created_at >= cutoff_date)
            .all()
        )

        if not feedbacks:
            return {
                "period_days": period_days,
                "total_findings_reviewed": 0,
                "suggestions_accepted": 0,
                "auto_fixes_applied": 0,
                "hours_saved_from_suggestions": 0.0,
                "hours_saved_from_autofix": 0.0,
                "total_hours_saved": 0.0,
                "monetary_value": 0.0,
                "roi_percentage": 0.0,
            }

        total_findings = len(feedbacks)
        accepted = len([f for f in feedbacks if f.feedback_type == "helpful"])

        # Estimate autofix applications (assume 50% of accepted have auto-fix)
        auto_fixes_applied = int(accepted * 0.5)

        # Get average fix time
        learning_metrics = self.db.query(LearningMetrics).all()
        avg_times = [m.avg_time_to_fix for m in learning_metrics if m.avg_time_to_fix]
        avg_fix_time = sum(avg_times) / len(avg_times) if avg_times else 1.0

        # Calculate hours saved
        # - From accepted suggestions: accepted * avg_time / 2 (they still need review)
        hours_from_suggestions = accepted * (avg_fix_time / 2.0)

        # - From auto-fixes: auto_fixes * avg_time (automatic application)
        hours_from_autofix = auto_fixes_applied * avg_fix_time

        total_hours_saved = hours_from_suggestions + hours_from_autofix

        # Monetary value (assuming $120/hour developer rate)
        hourly_rate = 120.0
        monetary_value = total_hours_saved * hourly_rate

        # ROI percentage (assuming tool costs $300/month ≈ $3600/year)
        annual_tool_cost = 3600.0
        roi_percentage = (monetary_value / annual_tool_cost) * 100 if annual_tool_cost > 0 else 0.0

        return {
            "period_days": period_days,
            "total_findings_reviewed": total_findings,
            "suggestions_accepted": accepted,
            "auto_fixes_applied": auto_fixes_applied,
            "hours_saved_from_suggestions": hours_from_suggestions,
            "hours_saved_from_autofix": hours_from_autofix,
            "total_hours_saved": total_hours_saved,
            "monetary_value": monetary_value,
            "roi_percentage": roi_percentage,
        }

    # ============================================================================
    # Persistence
    # ============================================================================

    def save_team_metrics(self, repo_url: str, metrics: Dict) -> TeamMetrics:
        """
        Save calculated team metrics to database.

        Args:
            repo_url: Repository URL
            metrics: Metrics dict from calculate_team_metrics()

        Returns:
            Saved TeamMetrics record
        """
        now = datetime.now(timezone.utc)
        period_start = now - timedelta(days=30)

        team_metric = TeamMetrics(
            team_id=repo_url,
            repo_url=repo_url,
            total_findings=metrics["total_findings"],
            accepted_findings=metrics["accepted_findings"],
            rejected_findings=metrics["rejected_findings"],
            ignored_findings=metrics["ignored_findings"],
            acceptance_rate=metrics["acceptance_rate"],
            avg_fix_time=metrics["avg_fix_time"],
            roi_hours_saved=metrics["roi_hours_saved"],
            roi_percentage=metrics["roi_percentage"],
            top_vulnerabilities=json.dumps(metrics["top_vulnerabilities"]),
            trend_direction=metrics["trend_direction"],
            trend_strength=metrics["trend_strength"],
            period_start=period_start,
            period_end=now,
        )

        self.db.add(team_metric)
        self.db.commit()
        self.db.refresh(team_metric)

        return team_metric

    def save_learning_paths(self, repo_url: str, paths: List[Dict]) -> List[LearningPath]:
        """
        Save learning paths to database.

        Args:
            repo_url: Repository URL
            paths: Learning paths list from generate_learning_paths()

        Returns:
            List of saved LearningPath records
        """
        saved_paths = []

        for path_data in paths:
            path = LearningPath(
                team_id=repo_url,
                repo_url=repo_url,
                vulnerability_type=path_data["vulnerability_type"],
                category=FindingCategory(path_data["category"]),
                current_acceptance_rate=path_data["current_rate"],
                potential_acceptance_rate=path_data["potential_rate"],
                estimated_hours_saved=path_data["hours_saved"],
                resources=json.dumps(path_data["resources"]),
                priority_score=path_data["priority_score"],
                rank=path_data["rank"],
                occurrences=path_data["occurrences"],
            )

            self.db.add(path)
            saved_paths.append(path)

        self.db.commit()

        return saved_paths

    def save_insights_trend(self, repo_url: str, period: str, trend_data: Dict) -> InsightsTrend:
        """
        Save insights trend data for historical tracking.

        Args:
            repo_url: Repository URL
            period: Period identifier (e.g., "2025-W47")
            trend_data: Trend data dict

        Returns:
            Saved InsightsTrend record
        """
        trend = InsightsTrend(
            team_id=repo_url,
            period=period,
            findings_count=trend_data.get("findings_count", 0),
            acceptance_rate=trend_data.get("acceptance_rate", 0.0),
            avg_fix_time=trend_data.get("avg_fix_time", 0.0),
            critical_findings=trend_data.get("critical", 0),
            high_findings=trend_data.get("high", 0),
            medium_findings=trend_data.get("medium", 0),
            low_findings=trend_data.get("low", 0),
            top_categories=json.dumps(trend_data.get("top_category", "")),
        )

        self.db.add(trend)
        self.db.commit()
        self.db.refresh(trend)

        return trend
