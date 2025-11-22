"""
Insights Dashboard REST API Endpoints

Provides endpoints for retrieving team metrics, trends, learning paths, and ROI analysis.
All endpoints include validation and error handling for production use.
"""

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy.orm import Session

from src.database import get_db
from src.learning.insights import InsightsGenerator

# Create router for insights endpoints
router = APIRouter(prefix="/api/insights", tags=["insights"])


@router.get("/team/{team_id}/metrics")
async def get_team_metrics(
    team_id: str,
    repo_url: str = Query(..., description="Repository URL"),
    period_days: int = Query(30, ge=1, le=365, description="Analysis period in days"),
    db: Session = Depends(get_db),
):
    """
    Get team metrics and KPIs.

    Returns:
    - total_findings: Total findings reviewed in period
    - accepted_findings: Findings accepted/helpful
    - rejected_findings: False positives
    - ignored_findings: Findings not acted upon
    - acceptance_rate: % of findings accepted
    - avg_fix_time: Average hours to fix
    - roi_hours_saved: Hours saved from automation
    - roi_percentage: % of effort saved
    - top_vulnerabilities: Most common issues
    - trend_direction: improving/declining/stable
    - trend_strength: Trend magnitude (0-1)
    """
    try:
        generator = InsightsGenerator(db)
        metrics = generator.calculate_team_metrics(
            repo_url=repo_url,
            period_days=period_days,
        )

        # Save metrics to database
        generator.save_team_metrics(repo_url, metrics)

        return {
            "team_id": team_id,
            "repo_url": repo_url,
            "period_days": period_days,
            **metrics,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating metrics: {str(e)}",
        )


@router.get("/team/{team_id}/trends")
async def get_trends(
    team_id: str,
    repo_url: str = Query(..., description="Repository URL"),
    weeks: int = Query(12, ge=1, le=52, description="Number of weeks to analyze"),
    db: Session = Depends(get_db),
):
    """
    Get vulnerability trends over time.

    Returns weekly trend data including:
    - findings_count: Total findings per week
    - acceptance_rate: Weekly acceptance rate
    - critical/high/medium/low: Findings by severity
    - top_category: Most common category

    Useful for identifying patterns and improvements/regressions.
    """
    try:
        generator = InsightsGenerator(db)
        trends = generator.analyze_trends(
            repo_url=repo_url,
            weeks=weeks,
        )

        # Save trends to database
        for trend_data in trends:
            generator.save_insights_trend(
                repo_url,
                trend_data["week"],
                trend_data,
            )

        return {
            "team_id": team_id,
            "repo_url": repo_url,
            "weeks": weeks,
            "trends": trends,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error analyzing trends: {str(e)}",
        )


@router.get("/team/{team_id}/anti-patterns")
async def get_anti_patterns(
    team_id: str,
    repo_url: str = Query(..., description="Repository URL"),
    db: Session = Depends(get_db),
):
    """
    Detect common anti-patterns (frequently rejected findings).

    Returns patterns that the system often gets wrong, helping improve accuracy.
    Useful for understanding what the system should avoid suggesting.

    Returns:
    - pattern: Anti-pattern description
    - occurrences: Number of times detected
    - rejection_rate: % of rejections for this pattern
    - category: Finding category
    - notes: Additional context
    """
    try:
        generator = InsightsGenerator(db)
        patterns = generator.detect_anti_patterns(repo_url=repo_url)

        return {
            "team_id": team_id,
            "repo_url": repo_url,
            "patterns": patterns,
            "pattern_count": len(patterns),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error detecting anti-patterns: {str(e)}",
        )


@router.get("/team/{team_id}/learning-paths")
async def get_learning_paths(
    team_id: str,
    repo_url: str = Query(..., description="Repository URL"),
    top_n: int = Query(5, ge=1, le=20, description="Number of top paths to return"),
    db: Session = Depends(get_db),
):
    """
    Get prioritized learning paths for team improvement.

    Identifies top areas where the team can improve with highest impact:
    - vulnerability_type: Type of security issue
    - category: Category (security/performance/best_practice)
    - current_rate: Current acceptance rate
    - potential_rate: Estimated rate after improvement
    - improvement_potential: Potential gain
    - occurrences: How many times found
    - hours_saved: Estimated hours that could be saved
    - priority_score: Composite priority (0-1)
    - resources: Recommended learning resources

    Ranked by priority (highest first).
    """
    try:
        generator = InsightsGenerator(db)
        paths = generator.generate_learning_paths(
            repo_url=repo_url,
            top_n=top_n,
        )

        # Save paths to database
        generator.save_learning_paths(repo_url, paths)

        return {
            "team_id": team_id,
            "repo_url": repo_url,
            "learning_paths": paths,
            "path_count": len(paths),
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating learning paths: {str(e)}",
        )


@router.get("/team/{team_id}/roi")
async def get_roi(
    team_id: str,
    repo_url: str = Query(..., description="Repository URL"),
    period_days: int = Query(90, ge=1, le=365, description="Analysis period in days"),
    db: Session = Depends(get_db),
):
    """
    Get ROI analysis for automated code review.

    Calculates hours saved and monetary value from:
    - Accepted suggestions (developer review time saved)
    - Auto-fix applications (automatic fixing)

    Returns:
    - period_days: Analysis period
    - total_findings_reviewed: Findings processed
    - suggestions_accepted: Helpful suggestions
    - auto_fixes_applied: Auto-fixes applied
    - hours_saved_from_suggestions: Review time saved
    - hours_saved_from_autofix: Automatic fixing time
    - total_hours_saved: Total productivity gain
    - monetary_value: Value at $120/hour dev rate
    - roi_percentage: Return on tool investment

    Useful for management visibility and cost-benefit analysis.
    """
    try:
        generator = InsightsGenerator(db)
        roi = generator.calculate_roi(
            repo_url=repo_url,
            period_days=period_days,
        )

        return {
            "team_id": team_id,
            "repo_url": repo_url,
            **roi,
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error calculating ROI: {str(e)}",
        )


@router.get("/health")
async def insights_health():
    """
    Health check endpoint for insights service.

    Returns:
        dict: Status indicator
    """
    return {"status": "healthy", "service": "insights"}
