"""
AI-Powered Suggestions Module

Enriches code review findings with AI-generated suggestions including:
- Auto-fixes (code corrections with confidence scores)
- Explanations (educational context)
- Improvement suggestions (best practices)

Severity-based logic:
- High/Critical findings get all 3 suggestion types
- Medium/Low findings get explanation only

Graceful degradation:
- Findings work even if suggestion generation fails
- Errors are logged but don't break the review
"""

from src.suggestions.enrichment import (
    SuggestionSet,
    EnrichedFinding,
    enrich_findings_with_suggestions,
)

__all__ = [
    "SuggestionSet",
    "EnrichedFinding",
    "enrich_findings_with_suggestions",
]
