"""
Integration tests for suggestion enrichment functionality.

Tests the enrichment layer that adds AI-generated suggestions to code findings.
Covers severity-based logic, error handling, and graceful degradation.
"""

import json
import pytest
from unittest.mock import patch, MagicMock
from typing import Dict, Any

from src.analysis.analyzer import AnalyzedFinding
from src.database import FindingCategory, FindingSeverity
from src.suggestions import enrich_findings_with_suggestions, EnrichedFinding, SuggestionSet


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def mock_llm_provider():
    """Create a mock LLM provider for testing."""
    provider = MagicMock()
    return provider


@pytest.fixture
def critical_finding() -> AnalyzedFinding:
    """Create a critical severity finding."""
    return AnalyzedFinding(
        category=FindingCategory.SECURITY,
        severity=FindingSeverity.CRITICAL,
        title="SQL Injection Vulnerability",
        description="Unsanitized user input in database query",
        file_path="app/models.py",
        line_number=42,
        confidence=0.95,
    )


@pytest.fixture
def high_finding() -> AnalyzedFinding:
    """Create a high severity finding."""
    return AnalyzedFinding(
        category=FindingCategory.SECURITY,
        severity=FindingSeverity.HIGH,
        title="Hardcoded Credentials",
        description="API key hardcoded in source file",
        file_path="app/config.py",
        line_number=10,
        confidence=0.92,
    )


@pytest.fixture
def medium_finding() -> AnalyzedFinding:
    """Create a medium severity finding."""
    return AnalyzedFinding(
        category=FindingCategory.PERFORMANCE,
        severity=FindingSeverity.MEDIUM,
        title="N+1 Query Problem",
        description="Loop queries database for each iteration",
        file_path="app/services.py",
        line_number=87,
        confidence=0.85,
    )


@pytest.fixture
def low_finding() -> AnalyzedFinding:
    """Create a low severity finding."""
    return AnalyzedFinding(
        category=FindingCategory.BEST_PRACTICE,
        severity=FindingSeverity.LOW,
        title="Missing Type Hints",
        description="Function missing type annotations",
        file_path="app/utils.py",
        line_number=15,
        confidence=0.8,
    )


@pytest.fixture
def code_snippet() -> str:
    """Sample code snippet for testing."""
    return """
# File: app/models.py
# Changes: +5/-2

- user = db.query(User).filter("id = " + user_id).first()
+ user = db.query(User).filter(User.id == user_id).first()
"""


# ============================================================================
# Enrichment Logic Tests
# ============================================================================


class TestSeverityBasedEnrichment:
    """Test that suggestions are generated based on severity."""

    def test_critical_finding_gets_all_suggestions(
        self, mock_llm_provider, critical_finding, code_snippet
    ):
        """Critical severity findings should get auto_fix, explanation, and improvements."""
        # Setup mock responses
        mock_llm_provider.generate_auto_fix.return_value = json.dumps(
            {
                "auto_fix": 'user = db.query(User).filter(User.id == user_id).first()',
                "confidence": 0.95,
            }
        )
        mock_llm_provider.generate_explanation.return_value = (
            "SQL injection allows attackers to execute arbitrary SQL commands."
        )
        mock_llm_provider.generate_improvement_suggestions.return_value = (
            "- Use parameterized queries\n- Validate all inputs"
        )

        # Enrich finding
        enriched = enrich_findings_with_suggestions(
            [critical_finding], code_snippet, mock_llm_provider
        )

        assert len(enriched) == 1
        assert isinstance(enriched[0], EnrichedFinding)
        assert enriched[0].suggestions.auto_fix is not None
        assert enriched[0].suggestions.explanation is not None
        assert enriched[0].suggestions.improvements is not None

    def test_high_finding_gets_all_suggestions(
        self, mock_llm_provider, high_finding, code_snippet
    ):
        """High severity findings should also get all suggestions."""
        # Setup mock responses
        mock_llm_provider.generate_auto_fix.return_value = json.dumps(
            {"auto_fix": "import os; API_KEY = os.getenv('API_KEY')", "confidence": 0.88}
        )
        mock_llm_provider.generate_explanation.return_value = (
            "Hardcoded credentials can be extracted from source code."
        )
        mock_llm_provider.generate_improvement_suggestions.return_value = (
            "- Use environment variables\n- Use secure vaults"
        )

        enriched = enrich_findings_with_suggestions(
            [high_finding], code_snippet, mock_llm_provider
        )

        assert len(enriched) == 1
        suggestions = enriched[0].suggestions.to_dict()
        assert "auto_fix" in suggestions
        assert "explanation" in suggestions
        assert "improvements" in suggestions

    def test_medium_finding_gets_explanation_only(
        self, mock_llm_provider, medium_finding, code_snippet
    ):
        """Medium severity findings should get explanation only (no auto_fix)."""
        # Setup mock responses
        mock_llm_provider.generate_explanation.return_value = (
            "N+1 queries cause multiple database roundtrips, impacting performance."
        )

        enriched = enrich_findings_with_suggestions(
            [medium_finding], code_snippet, mock_llm_provider
        )

        assert len(enriched) == 1
        suggestions = enriched[0].suggestions.to_dict()
        assert "explanation" in suggestions
        assert "auto_fix" not in suggestions
        assert "improvements" not in suggestions

        # Verify auto_fix and improvements were NOT attempted
        mock_llm_provider.generate_auto_fix.assert_not_called()
        mock_llm_provider.generate_improvement_suggestions.assert_not_called()

    def test_low_finding_gets_explanation_only(
        self, mock_llm_provider, low_finding, code_snippet
    ):
        """Low severity findings should get explanation only."""
        mock_llm_provider.generate_explanation.return_value = (
            "Type hints improve code maintainability."
        )

        enriched = enrich_findings_with_suggestions(
            [low_finding], code_snippet, mock_llm_provider
        )

        assert len(enriched) == 1
        suggestions = enriched[0].suggestions.to_dict()
        assert "explanation" in suggestions
        assert "auto_fix" not in suggestions
        assert "improvements" not in suggestions


class TestConfidenceFiltering:
    """Test that low-confidence auto-fixes are rejected."""

    def test_auto_fix_below_threshold_rejected(
        self, mock_llm_provider, critical_finding, code_snippet
    ):
        """Auto-fixes with confidence below 0.8 should be rejected."""
        # Setup low-confidence auto-fix
        mock_llm_provider.generate_auto_fix.return_value = json.dumps(
            {"auto_fix": "risky code", "confidence": 0.75}
        )
        mock_llm_provider.generate_explanation.return_value = "Some explanation"
        mock_llm_provider.generate_improvement_suggestions.return_value = "- Some suggestion"

        enriched = enrich_findings_with_suggestions(
            [critical_finding], code_snippet, mock_llm_provider
        )

        assert len(enriched) == 1
        suggestions = enriched[0].suggestions.to_dict()
        # Auto-fix should not be included
        assert "auto_fix" not in suggestions
        # But explanation and improvements should be
        assert "explanation" in suggestions
        assert "improvements" in suggestions

    def test_auto_fix_high_confidence_included(
        self, mock_llm_provider, critical_finding, code_snippet
    ):
        """Auto-fixes with confidence >= 0.8 should be included."""
        mock_llm_provider.generate_auto_fix.return_value = json.dumps(
            {"auto_fix": "safe code", "confidence": 0.95}
        )
        mock_llm_provider.generate_explanation.return_value = "Explanation"
        mock_llm_provider.generate_improvement_suggestions.return_value = "- Suggestion"

        enriched = enrich_findings_with_suggestions(
            [critical_finding], code_snippet, mock_llm_provider
        )

        assert len(enriched) == 1
        suggestions = enriched[0].suggestions.to_dict()
        assert "auto_fix" in suggestions
        assert suggestions["auto_fix_confidence"] == 0.95

    def test_null_auto_fix_excluded(
        self, mock_llm_provider, critical_finding, code_snippet
    ):
        """If LLM returns null auto_fix, it should be excluded."""
        mock_llm_provider.generate_auto_fix.return_value = json.dumps(
            {"auto_fix": None, "confidence": 0.0}
        )
        mock_llm_provider.generate_explanation.return_value = "Explanation"
        mock_llm_provider.generate_improvement_suggestions.return_value = "- Suggestion"

        enriched = enrich_findings_with_suggestions(
            [critical_finding], code_snippet, mock_llm_provider
        )

        assert len(enriched) == 1
        suggestions = enriched[0].suggestions.to_dict()
        assert "auto_fix" not in suggestions


class TestGracefulDegradation:
    """Test graceful degradation when suggestion generation fails."""

    def test_findings_returned_if_enrichment_fails(
        self, mock_llm_provider, critical_finding, code_snippet
    ):
        """Findings should be returned even if suggestion generation fails."""
        # Make LLM provider raise exception
        mock_llm_provider.generate_explanation.side_effect = RuntimeError("LLM timeout")

        enriched = enrich_findings_with_suggestions(
            [critical_finding], code_snippet, mock_llm_provider
        )

        # Finding should still be returned
        assert len(enriched) == 1
        assert enriched[0].finding == critical_finding
        # But without suggestions
        assert not enriched[0].suggestions.has_suggestions()

    def test_timeout_handling(
        self, mock_llm_provider, critical_finding, code_snippet
    ):
        """Timeouts should be caught and handled gracefully."""
        mock_llm_provider.generate_explanation.side_effect = TimeoutError("Request timed out")

        enriched = enrich_findings_with_suggestions(
            [critical_finding], code_snippet, mock_llm_provider
        )

        assert len(enriched) == 1
        assert not enriched[0].suggestions.has_suggestions()

    def test_invalid_json_response_handled(
        self, mock_llm_provider, critical_finding, code_snippet
    ):
        """Invalid JSON responses should be handled gracefully."""
        mock_llm_provider.generate_auto_fix.return_value = "invalid json {{{"
        mock_llm_provider.generate_explanation.return_value = "Explanation"
        mock_llm_provider.generate_improvement_suggestions.return_value = "- Suggestion"

        enriched = enrich_findings_with_suggestions(
            [critical_finding], code_snippet, mock_llm_provider
        )

        assert len(enriched) == 1
        suggestions = enriched[0].suggestions.to_dict()
        assert "auto_fix" not in suggestions
        assert "explanation" in suggestions
        assert "improvements" in suggestions

    def test_partial_suggestions_included(
        self, mock_llm_provider, critical_finding, code_snippet
    ):
        """If some suggestions fail, others should still be included."""
        mock_llm_provider.generate_auto_fix.return_value = json.dumps(
            {"auto_fix": "fixed code", "confidence": 0.9}
        )
        mock_llm_provider.generate_explanation.side_effect = RuntimeError("Failed")
        mock_llm_provider.generate_improvement_suggestions.return_value = (
            "- Best practice"
        )

        enriched = enrich_findings_with_suggestions(
            [critical_finding], code_snippet, mock_llm_provider
        )

        assert len(enriched) == 1
        suggestions = enriched[0].suggestions.to_dict()
        assert "auto_fix" in suggestions
        assert "explanation" not in suggestions
        assert "improvements" in suggestions


class TestMultipleFindingsEnrichment:
    """Test enrichment of multiple findings."""

    def test_enrich_multiple_findings(
        self,
        mock_llm_provider,
        critical_finding,
        high_finding,
        medium_finding,
        code_snippet,
    ):
        """Should enrich multiple findings independently."""
        # Setup responses for all calls
        mock_llm_provider.generate_auto_fix.return_value = json.dumps(
            {"auto_fix": "fixed", "confidence": 0.9}
        )
        mock_llm_provider.generate_explanation.return_value = "Explanation"
        mock_llm_provider.generate_improvement_suggestions.return_value = "- Suggestion"

        findings = [critical_finding, high_finding, medium_finding]
        enriched = enrich_findings_with_suggestions(
            findings, code_snippet, mock_llm_provider
        )

        assert len(enriched) == 3
        # Critical and high should have auto-fix
        assert enriched[0].suggestions.auto_fix is not None
        assert enriched[1].suggestions.auto_fix is not None
        # Medium should not have auto-fix
        assert enriched[2].suggestions.auto_fix is None

    def test_enrichment_continues_after_individual_failure(
        self, mock_llm_provider, critical_finding, high_finding, code_snippet
    ):
        """If one finding fails, enrichment should continue with others."""
        # First finding will fail, second will succeed
        mock_llm_provider.generate_explanation.side_effect = [
            RuntimeError("Failed"),
            "Explanation for second finding",
        ]
        mock_llm_provider.generate_auto_fix.side_effect = [
            json.dumps({"auto_fix": "fixed", "confidence": 0.9}),
            json.dumps({"auto_fix": "fixed", "confidence": 0.85}),
        ]
        mock_llm_provider.generate_improvement_suggestions.side_effect = [
            "- Suggestion 1",
            "- Suggestion 2",
        ]

        findings = [critical_finding, high_finding]
        enriched = enrich_findings_with_suggestions(
            findings, code_snippet, mock_llm_provider
        )

        assert len(enriched) == 2
        # First finding has auto-fix but no explanation
        assert enriched[0].suggestions.auto_fix is not None
        assert enriched[0].suggestions.explanation is None
        # Second finding has both
        assert enriched[1].suggestions.auto_fix is not None
        assert enriched[1].suggestions.explanation is not None

    def test_empty_findings_list(self, mock_llm_provider, code_snippet):
        """Empty findings list should return empty enriched list."""
        enriched = enrich_findings_with_suggestions(
            [], code_snippet, mock_llm_provider
        )

        assert len(enriched) == 0
        # LLM should not be called
        mock_llm_provider.generate_explanation.assert_not_called()


class TestDryRun:
    """Test dry-run mode for enrichment."""

    def test_dry_run_skips_llm_calls(
        self, mock_llm_provider, critical_finding, code_snippet
    ):
        """Dry-run mode should skip all LLM calls."""
        enriched = enrich_findings_with_suggestions(
            [critical_finding], code_snippet, mock_llm_provider, dry_run=True
        )

        assert len(enriched) == 1
        # Finding should be returned
        assert enriched[0].finding == critical_finding
        # But without suggestions
        assert not enriched[0].suggestions.has_suggestions()
        # And no LLM calls made
        mock_llm_provider.generate_explanation.assert_not_called()
        mock_llm_provider.generate_auto_fix.assert_not_called()
        mock_llm_provider.generate_improvement_suggestions.assert_not_called()


class TestSuggestionSetSerialization:
    """Test suggestion set serialization to dict."""

    def test_empty_suggestion_set_serialization(self):
        """Empty suggestion set should serialize to empty dict."""
        suggestions = SuggestionSet()
        result = suggestions.to_dict()
        assert result == {}

    def test_full_suggestion_set_serialization(self):
        """Full suggestion set should include all fields."""
        suggestions = SuggestionSet(
            auto_fix="fixed code",
            auto_fix_confidence=0.95,
            explanation="This is why it matters",
            improvements="- Use pattern X\n- Use library Y",
        )
        result = suggestions.to_dict()
        assert result["auto_fix"] == "fixed code"
        assert result["auto_fix_confidence"] == 0.95
        assert result["explanation"] == "This is why it matters"
        assert result["improvements"] == "- Use pattern X\n- Use library Y"

    def test_low_confidence_auto_fix_excluded(self):
        """Auto-fixes below 0.8 confidence should be excluded."""
        suggestions = SuggestionSet(
            auto_fix="risky code",
            auto_fix_confidence=0.7,
            explanation="Explanation",
        )
        result = suggestions.to_dict()
        assert "auto_fix" not in result
        assert result["explanation"] == "Explanation"

    def test_null_fields_excluded(self):
        """None/null fields should not appear in serialization."""
        suggestions = SuggestionSet(
            explanation="Only this field",
            auto_fix=None,
            improvements=None,
        )
        result = suggestions.to_dict()
        assert len(result) == 1
        assert "explanation" in result


class TestEnrichedFindingSerialization:
    """Test enriched finding serialization."""

    def test_enriched_finding_with_suggestions(self, critical_finding):
        """Enriched finding should include suggestions in serialization."""
        suggestions = SuggestionSet(
            auto_fix="fixed code",
            auto_fix_confidence=0.95,
            explanation="Explanation",
        )
        enriched = EnrichedFinding(critical_finding, suggestions)

        result = enriched.to_dict()
        assert result["severity"] == "critical"
        assert result["title"] == "SQL Injection Vulnerability"
        assert "suggestions" in result
        assert result["suggestions"]["auto_fix"] == "fixed code"

    def test_enriched_finding_without_suggestions(self, critical_finding):
        """Enriched finding without suggestions should not include suggestions key."""
        suggestions = SuggestionSet()  # Empty
        enriched = EnrichedFinding(critical_finding, suggestions)

        result = enriched.to_dict()
        assert result["severity"] == "critical"
        assert "suggestions" not in result

    def test_enriched_finding_preserves_metadata(self, high_finding):
        """Enriched finding should preserve all original finding metadata."""
        suggestions = SuggestionSet(explanation="Explanation")
        enriched = EnrichedFinding(high_finding, suggestions)

        result = enriched.to_dict()
        assert result["category"] == "security"
        assert result["severity"] == "high"
        assert result["file_path"] == "app/config.py"
        assert result["line_number"] == 10
        assert result["confidence"] == 0.92
        assert result["title"] == "Hardcoded Credentials"


class TestIntegrationWithAnalyzer:
    """Integration tests with CodeAnalyzer."""

    def test_enrichment_enabled_initialization(self):
        """CodeAnalyzer should accept enrich_suggestions parameter."""
        from src.analysis.analyzer import CodeAnalyzer

        # Setup mock provider
        mock_provider = MagicMock()
        analyzer = CodeAnalyzer(
            llm_provider=mock_provider, enrich_suggestions=True, use_tools=False
        )

        # Verify enrich_suggestions is set
        assert analyzer.enrich_suggestions is True

    def test_enrichment_disabled_initialization(self):
        """CodeAnalyzer should accept enrich_suggestions=False."""
        from src.analysis.analyzer import CodeAnalyzer

        # Setup mock provider
        mock_provider = MagicMock()
        analyzer = CodeAnalyzer(
            llm_provider=mock_provider, enrich_suggestions=False, use_tools=False
        )

        # Verify enrich_suggestions is set
        assert analyzer.enrich_suggestions is False

    def test_enrichment_default_enabled(self):
        """enrich_suggestions should default to True."""
        from src.analysis.analyzer import CodeAnalyzer

        # Setup mock provider
        mock_provider = MagicMock()
        analyzer = CodeAnalyzer(llm_provider=mock_provider, use_tools=False)

        # Should default to True
        assert analyzer.enrich_suggestions is True
