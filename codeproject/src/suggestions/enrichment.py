"""
Suggestion Enrichment Module

Enriches findings with AI-generated suggestions based on severity level.
Supports auto-fixes, explanations, and improvement suggestions.

Architecture:
- High/Critical severity: auto_fix + explanation + improvements
- Medium severity: explanation only
- Low severity: explanation only
- Graceful degradation: Findings work even if suggestion generation fails
- Synchronous: Complete response with all suggestions before returning
"""

import logging
import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING

from src.llm.provider import LLMProvider
from src.database import FindingSeverity
from src.suggestions.cache import generate_cache_key, get_cache

if TYPE_CHECKING:
    from src.analysis.analyzer import AnalyzedFinding

logger = logging.getLogger(__name__)


# ============================================================================
# Suggestion Data Models
# ============================================================================


class SuggestionSet:
    """
    Complete set of suggestions for a finding.

    Attributes:
        auto_fix: Generated code fix (JSON with confidence)
        auto_fix_confidence: Confidence score for auto_fix (0.0-1.0)
        explanation: Educational explanation of the issue
        improvements: Best practice suggestions (bullet points)
    """

    def __init__(
        self,
        auto_fix: Optional[str] = None,
        auto_fix_confidence: float = 0.0,
        explanation: Optional[str] = None,
        improvements: Optional[str] = None,
    ):
        self.auto_fix = auto_fix
        self.auto_fix_confidence = auto_fix_confidence
        self.explanation = explanation
        self.improvements = improvements

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {}

        # Only include auto_fix if confidence >= 0.8
        if self.auto_fix and self.auto_fix_confidence >= 0.8:
            result["auto_fix"] = self.auto_fix
            result["auto_fix_confidence"] = self.auto_fix_confidence

        if self.explanation:
            result["explanation"] = self.explanation

        if self.improvements:
            result["improvements"] = self.improvements

        return result

    def has_suggestions(self) -> bool:
        """Check if any suggestions are available."""
        return bool(self.auto_fix or self.explanation or self.improvements)


class EnrichedFinding:
    """
    Finding with enriched suggestions.

    Extends AnalyzedFinding with AI-generated suggestions.
    Provides transparent access to underlying finding properties for compatibility.
    """

    def __init__(self, finding: "AnalyzedFinding", suggestions: Optional[SuggestionSet] = None):
        self.finding = finding
        self.suggestions = suggestions or SuggestionSet()

    # Convenience properties for accessing underlying finding attributes
    @property
    def category(self):
        """Access finding category."""
        return self.finding.category

    @property
    def severity(self):
        """Access finding severity."""
        return self.finding.severity

    @property
    def title(self):
        """Access finding title."""
        return self.finding.title

    @property
    def description(self):
        """Access finding description."""
        return self.finding.description

    @property
    def file_path(self):
        """Access finding file path."""
        return self.finding.file_path

    @property
    def line_number(self):
        """Access finding line number."""
        return self.finding.line_number

    @property
    def suggested_fix(self):
        """Access finding suggested fix."""
        return self.finding.suggested_fix

    @property
    def confidence(self):
        """Access finding confidence."""
        return self.finding.confidence

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        result = {
            "category": self.finding.category.value,
            "severity": self.finding.severity.value,
            "title": self.finding.title,
            "description": self.finding.description,
            "file_path": self.finding.file_path,
            "line_number": self.finding.line_number,
            "confidence": self.finding.confidence,
        }

        if self.finding.suggested_fix:
            result["suggested_fix"] = self.finding.suggested_fix

        # Add suggestions if available
        suggestions_dict = self.suggestions.to_dict()
        if suggestions_dict:
            result["suggestions"] = suggestions_dict

        return result


# ============================================================================
# Enrichment Engine
# ============================================================================


def enrich_findings_with_suggestions(
    findings: List["AnalyzedFinding"],
    code_diff: str,
    llm_provider: LLMProvider,
    dry_run: bool = False,
) -> List[EnrichedFinding]:
    """
    Enrich findings with AI-generated suggestions.

    Severity-based logic:
    - High/Critical: auto_fix + explanation + improvements
    - Medium: explanation only
    - Low: explanation only

    Graceful degradation:
    - Findings are returned even if suggestion generation fails
    - Errors are logged but don't fail the entire enrichment process
    - Partial suggestions are included if available

    Args:
        findings: List of AnalyzedFinding objects from code analyzer
        code_diff: Code diff string for context
        llm_provider: LLM provider instance for generating suggestions
        dry_run: If True, only analyze without generating suggestions (for testing)

    Returns:
        List of EnrichedFinding objects with suggestions attached

    Note:
        This function is synchronous - it blocks until all suggestions
        are generated before returning. Use with timeout handling at
        the caller level (e.g., webhook timeout or async wrapper).
    """
    if not findings:
        return []

    enriched = []

    for idx, finding in enumerate(findings):
        try:
            logger.debug(
                f"Enriching finding {idx+1}/{len(findings)}: "
                f"{finding.title} ({finding.severity.value})"
            )

            # Determine what suggestions to generate based on severity
            suggestions = _generate_suggestions_for_finding(
                finding, code_diff, llm_provider, dry_run
            )

            # Create enriched finding
            enriched_finding = EnrichedFinding(finding, suggestions)
            enriched.append(enriched_finding)

            logger.debug(
                f"Successfully enriched finding: {finding.title} "
                f"({suggestions.to_dict().keys() if suggestions else 'no suggestions'})"
            )

        except Exception as e:
            # Log error but continue with next finding
            logger.warning(
                f"Failed to enrich finding '{finding.title}': {str(e)}. "
                f"Returning finding without suggestions."
            )
            enriched_finding = EnrichedFinding(finding, SuggestionSet())
            enriched.append(enriched_finding)

    return enriched


def _generate_suggestions_for_finding(
    finding: "AnalyzedFinding",
    code_diff: str,
    llm_provider: LLMProvider,
    dry_run: bool = False,
) -> SuggestionSet:
    """
    Generate suggestions for a single finding based on severity.

    Implements caching to avoid redundant LLM calls for identical findings.
    Cache key is based on finding title and file path.

    Args:
        finding: AnalyzedFinding to generate suggestions for
        code_diff: Code diff for context
        llm_provider: LLM provider instance
        dry_run: If True, skip actual LLM calls

    Returns:
        SuggestionSet with generated suggestions
    """
    suggestions = SuggestionSet()

    # Dry run: return empty suggestions
    if dry_run:
        return suggestions

    # Check cache first
    cache = get_cache()
    cache_key = generate_cache_key(finding.title, finding.file_path, code_diff[:500])

    cached = cache.get(cache_key)
    if cached is not None:
        logger.debug(f"Cache hit for finding '{finding.title}'")
        # Build SuggestionSet from cached data
        suggestions = SuggestionSet(
            auto_fix=cached.get("auto_fix"),
            auto_fix_confidence=cached.get("auto_fix_confidence", 0.0),
            explanation=cached.get("explanation"),
            improvements=cached.get("improvement_suggestions"),
        )
        return suggestions

    # Cache miss: generate suggestions
    logger.debug(f"Cache miss for finding '{finding.title}', generating new suggestions")

    # Always generate explanation (for all severities)
    explanation = _generate_explanation(finding, code_diff, llm_provider)
    if explanation:
        suggestions.explanation = explanation

    # For high/critical: generate auto_fix and improvements
    if finding.severity in [FindingSeverity.HIGH, FindingSeverity.CRITICAL]:
        # Generate auto-fix
        auto_fix, confidence = _generate_auto_fix(finding, code_diff, llm_provider)
        if auto_fix and confidence >= 0.8:
            suggestions.auto_fix = auto_fix
            suggestions.auto_fix_confidence = confidence

        # Generate improvement suggestions
        improvements = _generate_improvement_suggestions(finding, code_diff, llm_provider)
        if improvements:
            suggestions.improvements = improvements

    # Store in cache for future use
    cache.set(
        cache_key,
        finding.title,
        finding.file_path,
        auto_fix=suggestions.auto_fix,
        auto_fix_confidence=suggestions.auto_fix_confidence,
        explanation=suggestions.explanation,
        improvement_suggestions=suggestions.improvements,
    )

    return suggestions


def _generate_explanation(
    finding: "AnalyzedFinding",
    code_diff: str,
    llm_provider: LLMProvider,
) -> Optional[str]:
    """
    Generate educational explanation for a finding.

    Args:
        finding: AnalyzedFinding to explain
        code_diff: Code diff for context
        llm_provider: LLM provider instance

    Returns:
        Explanation text or None if generation fails
    """
    try:
        finding_dict = {
            "severity": finding.severity.value,
            "title": finding.title,
            "category": finding.category.value,
            "file_path": finding.file_path,
            "line_number": finding.line_number,
        }

        explanation = llm_provider.generate_explanation(finding_dict, code_diff)

        # Validate explanation
        if not explanation or not isinstance(explanation, str):
            logger.warning("Invalid explanation response from LLM")
            return None

        explanation = explanation.strip()
        if not explanation:
            return None

        logger.debug(f"Generated explanation ({len(explanation)} chars)")
        return explanation

    except TimeoutError as e:
        logger.warning(f"Timeout generating explanation: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Error generating explanation: {str(e)}")
        return None


def _generate_auto_fix(
    finding: "AnalyzedFinding",
    code_diff: str,
    llm_provider: LLMProvider,
) -> tuple[Optional[str], float]:
    """
    Generate auto-fix for a finding.

    Returns None, 0.0 if:
    - LLM times out
    - LLM returns invalid JSON
    - Confidence score is below 0.8
    - LLM declines to generate fix

    Args:
        finding: AnalyzedFinding to generate fix for
        code_diff: Code diff for context
        llm_provider: LLM provider instance

    Returns:
        Tuple of (auto_fix_code, confidence) where confidence is 0.0-1.0
    """
    try:
        finding_dict = {
            "severity": finding.severity.value,
            "title": finding.title,
            "category": finding.category.value,
            "file_path": finding.file_path,
            "line_number": finding.line_number,
        }

        response = llm_provider.generate_auto_fix(finding_dict, code_diff)

        # Parse response
        try:
            data = json.loads(response)
        except json.JSONDecodeError as e:
            logger.warning(f"Invalid JSON in auto_fix response: {str(e)}")
            return None, 0.0

        # Extract auto_fix and confidence
        auto_fix = data.get("auto_fix")
        confidence = data.get("confidence", 0.0)

        # Validate
        if not isinstance(confidence, (int, float)):
            confidence = 0.0

        confidence = max(0.0, min(1.0, float(confidence)))

        # If auto_fix is None or confidence too low, return with 0 confidence
        if not auto_fix:
            logger.debug("LLM declined to generate auto_fix")
            return None, 0.0

        if confidence < 0.8:
            logger.debug(f"Auto-fix confidence too low ({confidence:.2f}), rejecting")
            return None, confidence

        logger.debug(f"Generated auto-fix with confidence {confidence:.2f}")
        return auto_fix, confidence

    except TimeoutError as e:
        logger.warning(f"Timeout generating auto-fix: {str(e)}")
        return None, 0.0
    except Exception as e:
        logger.warning(f"Error generating auto-fix: {str(e)}")
        return None, 0.0


def _generate_improvement_suggestions(
    finding: "AnalyzedFinding",
    code_diff: str,
    llm_provider: LLMProvider,
) -> Optional[str]:
    """
    Generate improvement suggestions for a finding.

    Args:
        finding: AnalyzedFinding to generate suggestions for
        code_diff: Code diff for context
        llm_provider: LLM provider instance

    Returns:
        Suggestions text (bullet points) or None if generation fails
    """
    try:
        finding_dict = {
            "severity": finding.severity.value,
            "title": finding.title,
            "category": finding.category.value,
            "file_path": finding.file_path,
            "line_number": finding.line_number,
        }

        suggestions = llm_provider.generate_improvement_suggestions(finding_dict, code_diff)

        # Validate suggestions
        if not suggestions or not isinstance(suggestions, str):
            logger.warning("Invalid suggestions response from LLM")
            return None

        suggestions = suggestions.strip()
        if not suggestions:
            return None

        logger.debug(f"Generated improvement suggestions ({len(suggestions)} chars)")
        return suggestions

    except TimeoutError as e:
        logger.warning(f"Timeout generating improvements: {str(e)}")
        return None
    except Exception as e:
        logger.warning(f"Error generating improvements: {str(e)}")
        return None
