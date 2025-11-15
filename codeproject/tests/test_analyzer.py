"""
Tests for code analyzer.

Tests LLM orchestration, response parsing, deduplication, and finding sorting.
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from src.analysis.analyzer import CodeAnalyzer, AnalyzedFinding
from src.analysis.diff_parser import CodeChange, FileDiff, DiffParser
from src.database import FindingCategory, FindingSeverity
from src.llm.provider import LLMProvider


# ============================================================================
# Test Data & Fixtures
# ============================================================================

@pytest.fixture
def mock_llm_provider():
    """Provide a mock LLM provider."""
    provider = MagicMock(spec=LLMProvider)
    return provider


@pytest.fixture
def analyzer(mock_llm_provider):
    """Provide a CodeAnalyzer with mock LLM provider."""
    return CodeAnalyzer(llm_provider=mock_llm_provider)


@pytest.fixture
def sample_file_diff():
    """Sample file diff with security and performance issues."""
    changes = [
        CodeChange(
            file_path="app.py",
            line_number=42,
            old_line_number=42,
            change_type="remove",
            content="    query = f\"SELECT * FROM users WHERE id={user_id}\"",
            context_before=["def get_user(user_id):"],
            context_after=["    cursor.execute(query)"],
        ),
        CodeChange(
            file_path="app.py",
            line_number=43,
            old_line_number=None,
            change_type="add",
            content="    query = \"SELECT * FROM users WHERE id=?\"",
            context_before=["def get_user(user_id):"],
            context_after=["    cursor.execute(query, (user_id,))"],
        ),
    ]
    return FileDiff(
        file_path="app.py",
        old_path=None,
        is_binary=False,
        additions=1,
        deletions=1,
        changes=changes,
    )


@pytest.fixture
def sample_security_response():
    """Sample security analysis response from LLM."""
    return json.dumps({
        "findings": [
            {
                "severity": "critical",
                "title": "SQL Injection Vulnerability",
                "description": "User input directly interpolated in SQL query",
                "file_path": "app.py",
                "line_number": 42,
                "suggested_fix": "Use parameterized queries",
                "confidence": 0.98,
            }
        ]
    })


@pytest.fixture
def sample_performance_response():
    """Sample performance analysis response from LLM."""
    return json.dumps({
        "findings": [
            {
                "severity": "high",
                "title": "N+1 Query Problem",
                "description": "Loop querying database inefficiently",
                "file_path": "app.py",
                "line_number": 50,
                "suggested_fix": "Use batch queries or join",
                "confidence": 0.85,
            }
        ]
    })


# ============================================================================
# Test AnalyzedFinding
# ============================================================================

class TestAnalyzedFinding:
    """Tests for AnalyzedFinding class."""

    def test_create_analyzed_finding(self):
        """Test creating an AnalyzedFinding."""
        finding = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="SQL Injection",
            description="Input is not sanitized",
            file_path="app.py",
            line_number=42,
            suggested_fix="Use parameterized queries",
            confidence=0.95,
        )

        assert finding.category == FindingCategory.SECURITY
        assert finding.severity == FindingSeverity.CRITICAL
        assert finding.title == "SQL Injection"
        assert finding.confidence == 0.95

    def test_dedup_key_identical_findings(self):
        """Test dedup key for identical findings."""
        finding1 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="SQL Injection",
            description="Description 1",
            file_path="app.py",
            line_number=42,
        )

        finding2 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,  # Different severity
            title="SQL Injection",
            description="Description 2",  # Different description
            file_path="app.py",
            line_number=42,
        )

        # Same dedup key despite different severity/description
        assert finding1.dedup_key() == finding2.dedup_key()

    def test_dedup_key_different_files(self):
        """Test dedup key differs for different files."""
        finding1 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="SQL Injection",
            description="Issue",
            file_path="app.py",
            line_number=42,
        )

        finding2 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="SQL Injection",
            description="Issue",
            file_path="main.py",  # Different file
            line_number=42,
        )

        assert finding1.dedup_key() != finding2.dedup_key()

    def test_dedup_key_case_insensitive_title(self):
        """Test dedup key normalizes title case."""
        finding1 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="SQL INJECTION",
            description="Issue",
            file_path="app.py",
            line_number=42,
        )

        finding2 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="sql injection",  # Different case
            description="Issue",
            file_path="app.py",
            line_number=42,
        )

        # Should be same after normalization
        assert finding1.dedup_key() == finding2.dedup_key()

    def test_dedup_key_none_line_number(self):
        """Test dedup key handles None line numbers."""
        finding1 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Module-level issue",
            description="Issue",
            file_path="app.py",
            line_number=None,
        )

        finding2 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Module-level issue",
            description="Issue",
            file_path="app.py",
            line_number=None,
        )

        assert finding1.dedup_key() == finding2.dedup_key()


# ============================================================================
# Test Code Snippet Conversion
# ============================================================================

class TestCodeSnippetConversion:
    """Tests for converting diffs to code snippets."""

    def test_diffs_to_code_snippet_basic(self, analyzer, sample_file_diff):
        """Test converting file diff to code snippet."""
        snippet = analyzer._diffs_to_code_snippet([sample_file_diff])

        assert "# File: app.py" in snippet
        assert "+1/-1" in snippet
        assert "SQL Injection" not in snippet  # Just the changes, not analysis

    def test_diffs_to_code_snippet_empty(self, analyzer):
        """Test converting empty diff list."""
        snippet = analyzer._diffs_to_code_snippet([])
        assert snippet == ""

    def test_diffs_to_code_snippet_multiple_files(self, analyzer):
        """Test converting multiple file diffs."""
        diff1 = FileDiff(
            file_path="file1.py",
            old_path=None,
            is_binary=False,
            additions=1,
            deletions=0,
            changes=[
                CodeChange(
                    file_path="file1.py",
                    line_number=10,
                    old_line_number=None,
                    change_type="add",
                    content="new code",
                ),
            ],
        )

        diff2 = FileDiff(
            file_path="file2.py",
            old_path=None,
            is_binary=False,
            additions=2,
            deletions=1,
            changes=[],
        )

        snippet = analyzer._diffs_to_code_snippet([diff1, diff2])

        assert "# File: file1.py" in snippet
        assert "# File: file2.py" in snippet


# ============================================================================
# Test Response Parsing
# ============================================================================

class TestResponseParsing:
    """Tests for parsing LLM responses into findings."""

    def test_parse_valid_security_response(self, analyzer, sample_security_response):
        """Test parsing valid security response."""
        findings = analyzer._parse_findings_response(
            sample_security_response,
            FindingCategory.SECURITY,
        )

        assert len(findings) == 1
        finding = findings[0]
        assert finding.title == "SQL Injection Vulnerability"
        assert finding.severity == FindingSeverity.CRITICAL
        assert finding.confidence == 0.98

    def test_parse_multiple_findings(self, analyzer):
        """Test parsing response with multiple findings."""
        response = json.dumps({
            "findings": [
                {
                    "severity": "critical",
                    "title": "Issue 1",
                    "description": "Desc 1",
                    "file_path": "file1.py",
                    "line_number": 10,
                },
                {
                    "severity": "high",
                    "title": "Issue 2",
                    "description": "Desc 2",
                    "file_path": "file2.py",
                    "line_number": 20,
                },
            ]
        })

        findings = analyzer._parse_findings_response(
            response,
            FindingCategory.SECURITY,
        )

        assert len(findings) == 2

    def test_parse_missing_findings_key(self, analyzer):
        """Test parsing response missing 'findings' key."""
        response = json.dumps({"errors": []})
        findings = analyzer._parse_findings_response(
            response,
            FindingCategory.SECURITY,
        )

        assert findings == []

    def test_parse_findings_not_list(self, analyzer):
        """Test parsing response where findings is not a list."""
        response = json.dumps({"findings": "not a list"})
        findings = analyzer._parse_findings_response(
            response,
            FindingCategory.SECURITY,
        )

        assert findings == []

    def test_parse_invalid_json(self, analyzer):
        """Test parsing invalid JSON response."""
        response = "not valid json {]["
        findings = analyzer._parse_findings_response(
            response,
            FindingCategory.SECURITY,
        )

        assert findings == []

    def test_parse_malformed_finding_skipped(self, analyzer):
        """Test that malformed findings are skipped, not fatal."""
        response = json.dumps({
            "findings": [
                {
                    "severity": "critical",
                    "title": "Valid issue",
                    "description": "Desc",
                    "file_path": "file.py",
                },
                {
                    # Missing required fields
                    "severity": "high",
                },
                {
                    "severity": "medium",
                    "title": "Another valid issue",
                    "description": "Desc",
                    "file_path": "file.py",
                },
            ]
        })

        findings = analyzer._parse_findings_response(
            response,
            FindingCategory.SECURITY,
        )

        # Should have 2 valid findings, skip the malformed one
        assert len(findings) == 2
        assert findings[0].title == "Valid issue"
        assert findings[1].title == "Another valid issue"

    def test_parse_optional_suggested_fix(self, analyzer):
        """Test parsing finding without suggested_fix."""
        response = json.dumps({
            "findings": [
                {
                    "severity": "high",
                    "title": "Issue",
                    "description": "Desc",
                    "file_path": "file.py",
                    # No suggested_fix
                }
            ]
        })

        findings = analyzer._parse_findings_response(
            response,
            FindingCategory.SECURITY,
        )

        assert len(findings) == 1
        assert findings[0].suggested_fix is None

    def test_parse_optional_line_number(self, analyzer):
        """Test parsing finding without line_number."""
        response = json.dumps({
            "findings": [
                {
                    "severity": "high",
                    "title": "Module-level issue",
                    "description": "Desc",
                    "file_path": "file.py",
                    # No line_number
                }
            ]
        })

        findings = analyzer._parse_findings_response(
            response,
            FindingCategory.SECURITY,
        )

        assert len(findings) == 1
        assert findings[0].line_number is None

    def test_parse_invalid_severity(self, analyzer):
        """Test parsing finding with invalid severity."""
        response = json.dumps({
            "findings": [
                {
                    "severity": "super-critical",  # Invalid
                    "title": "Issue",
                    "description": "Desc",
                    "file_path": "file.py",
                }
            ]
        })

        findings = analyzer._parse_findings_response(
            response,
            FindingCategory.SECURITY,
        )

        # Should skip finding with invalid severity
        assert len(findings) == 0

    def test_parse_severity_case_insensitive(self, analyzer):
        """Test parsing handles severity case variations."""
        response = json.dumps({
            "findings": [
                {
                    "severity": "CRITICAL",  # Uppercase
                    "title": "Issue",
                    "description": "Desc",
                    "file_path": "file.py",
                },
                {
                    "severity": "High",  # Mixed case
                    "title": "Issue 2",
                    "description": "Desc",
                    "file_path": "file.py",
                },
            ]
        })

        findings = analyzer._parse_findings_response(
            response,
            FindingCategory.SECURITY,
        )

        assert len(findings) == 2
        assert findings[0].severity == FindingSeverity.CRITICAL
        assert findings[1].severity == FindingSeverity.HIGH

    def test_parse_confidence_default(self, analyzer):
        """Test parsing uses default confidence when not provided."""
        response = json.dumps({
            "findings": [
                {
                    "severity": "high",
                    "title": "Issue",
                    "description": "Desc",
                    "file_path": "file.py",
                    # No confidence
                }
            ]
        })

        findings = analyzer._parse_findings_response(
            response,
            FindingCategory.SECURITY,
        )

        assert len(findings) == 1
        assert findings[0].confidence == 0.95  # Default

    def test_parse_confidence_clamped(self, analyzer):
        """Test parsing clamps confidence to [0.0, 1.0]."""
        response = json.dumps({
            "findings": [
                {
                    "severity": "high",
                    "title": "Issue 1",
                    "description": "Desc",
                    "file_path": "file.py",
                    "confidence": 2.5,  # Over 1.0
                },
                {
                    "severity": "high",
                    "title": "Issue 2",
                    "description": "Desc",
                    "file_path": "file.py",
                    "confidence": -0.5,  # Below 0.0
                },
            ]
        })

        findings = analyzer._parse_findings_response(
            response,
            FindingCategory.SECURITY,
        )

        assert findings[0].confidence == 1.0
        assert findings[1].confidence == 0.0


# ============================================================================
# Test Deduplication
# ============================================================================

class TestDeduplication:
    """Tests for finding deduplication."""

    def test_deduplicate_identical_findings(self, analyzer):
        """Test deduplicating identical findings."""
        finding1 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="SQL Injection",
            description="Desc 1",
            file_path="app.py",
            line_number=42,
            confidence=0.95,
        )

        finding2 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,  # Different severity
            title="SQL Injection",
            description="Desc 2",  # Different description
            file_path="app.py",
            line_number=42,
            confidence=0.80,
        )

        deduplicated = analyzer._deduplicate_findings([finding1, finding2])

        assert len(deduplicated) == 1
        # Should keep the one with higher confidence
        assert deduplicated[0].confidence == 0.95
        assert deduplicated[0].severity == FindingSeverity.CRITICAL

    def test_deduplicate_keeps_higher_confidence(self, analyzer):
        """Test dedup keeps finding with highest confidence."""
        finding1 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Issue",
            description="Desc",
            file_path="app.py",
            line_number=10,
            confidence=0.75,
        )

        finding2 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="Issue",
            description="Different",
            file_path="app.py",
            line_number=10,
            confidence=0.99,
        )

        deduplicated = analyzer._deduplicate_findings([finding1, finding2])

        assert len(deduplicated) == 1
        assert deduplicated[0].confidence == 0.99

    def test_deduplicate_no_duplicates(self, analyzer):
        """Test dedup with no duplicate findings."""
        finding1 = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="SQL Injection",
            description="Desc",
            file_path="app.py",
            line_number=10,
        )

        finding2 = AnalyzedFinding(
            category=FindingCategory.PERFORMANCE,
            severity=FindingSeverity.HIGH,
            title="N+1 Query",
            description="Desc",
            file_path="app.py",
            line_number=20,
        )

        deduplicated = analyzer._deduplicate_findings([finding1, finding2])

        assert len(deduplicated) == 2

    def test_deduplicate_empty(self, analyzer):
        """Test dedup with empty list."""
        deduplicated = analyzer._deduplicate_findings([])
        assert deduplicated == []


# ============================================================================
# Test Sorting
# ============================================================================

class TestSorting:
    """Tests for sorting findings by severity."""

    def test_sort_by_severity(self, analyzer):
        """Test sorting findings by severity."""
        findings = [
            AnalyzedFinding(
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.LOW,
                title="Low",
                description="Desc",
                file_path="file.py",
            ),
            AnalyzedFinding(
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.CRITICAL,
                title="Critical",
                description="Desc",
                file_path="file.py",
            ),
            AnalyzedFinding(
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.MEDIUM,
                title="Medium",
                description="Desc",
                file_path="file.py",
            ),
            AnalyzedFinding(
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.HIGH,
                title="High",
                description="Desc",
                file_path="file.py",
            ),
        ]

        sorted_findings = analyzer._sort_by_severity(findings)

        assert sorted_findings[0].title == "Critical"
        assert sorted_findings[1].title == "High"
        assert sorted_findings[2].title == "Medium"
        assert sorted_findings[3].title == "Low"

    def test_sort_empty(self, analyzer):
        """Test sorting empty list."""
        sorted_findings = analyzer._sort_by_severity([])
        assert sorted_findings == []


# ============================================================================
# Test End-to-End Analysis
# ============================================================================

class TestEndToEndAnalysis:
    """Tests for end-to-end analysis flow."""

    def test_analyze_code_changes_basic(self, analyzer, sample_file_diff, sample_security_response, sample_performance_response):
        """Test analyzing code changes."""
        analyzer.llm_provider.analyze_security.return_value = sample_security_response
        analyzer.llm_provider.analyze_performance.return_value = sample_performance_response

        findings = analyzer.analyze_code_changes([sample_file_diff])

        assert len(findings) == 2
        # Should be sorted by severity (critical first)
        assert findings[0].severity == FindingSeverity.CRITICAL
        assert findings[1].severity == FindingSeverity.HIGH

    def test_analyze_empty_diffs(self, analyzer):
        """Test analyzing empty diff list."""
        findings = analyzer.analyze_code_changes([])
        assert findings == []

    def test_analyze_handles_security_error(self, analyzer, sample_file_diff, sample_performance_response):
        """Test analysis continues if security check fails."""
        analyzer.llm_provider.analyze_security.side_effect = Exception("API error")
        analyzer.llm_provider.analyze_performance.return_value = sample_performance_response

        findings = analyzer.analyze_code_changes([sample_file_diff])

        # Should still get performance findings
        assert len(findings) == 1
        assert findings[0].category == FindingCategory.PERFORMANCE

    def test_analyze_handles_performance_error(self, analyzer, sample_file_diff, sample_security_response):
        """Test analysis continues if performance check fails."""
        analyzer.llm_provider.analyze_security.return_value = sample_security_response
        analyzer.llm_provider.analyze_performance.side_effect = Exception("API error")

        findings = analyzer.analyze_code_changes([sample_file_diff])

        # Should still get security findings
        assert len(findings) == 1
        assert findings[0].category == FindingCategory.SECURITY

    def test_analyze_handles_both_errors(self, analyzer, sample_file_diff):
        """Test analysis handles both checks failing gracefully."""
        analyzer.llm_provider.analyze_security.side_effect = Exception("API error")
        analyzer.llm_provider.analyze_performance.side_effect = Exception("API error")

        findings = analyzer.analyze_code_changes([sample_file_diff])

        assert findings == []

    def test_analyze_deduplicates_within_category(self, analyzer, sample_file_diff):
        """Test deduplication works within same category."""
        # Same analyzer reports same finding twice (shouldn't happen but test robustness)
        response = json.dumps({
            "findings": [
                {
                    "severity": "high",
                    "title": "Duplicate Issue",
                    "description": "Same issue",
                    "file_path": "app.py",
                    "line_number": 42,
                    "confidence": 0.9,
                },
                {
                    "severity": "high",
                    "title": "Duplicate Issue",
                    "description": "Same issue",
                    "file_path": "app.py",
                    "line_number": 42,
                    "confidence": 0.85,
                },
            ]
        })

        analyzer.llm_provider.analyze_security.return_value = response
        analyzer.llm_provider.analyze_performance.return_value = json.dumps({"findings": []})

        findings = analyzer.analyze_code_changes([sample_file_diff])

        # Should deduplicate (keep higher confidence)
        assert len(findings) == 1
        assert findings[0].confidence == 0.9

    def test_analyze_sorts_final_results(self, analyzer, sample_file_diff):
        """Test final results are sorted by severity."""
        security_response = json.dumps({
            "findings": [
                {
                    "severity": "low",
                    "title": "Minor issue",
                    "description": "Desc",
                    "file_path": "app.py",
                    "line_number": 1,
                }
            ]
        })

        performance_response = json.dumps({
            "findings": [
                {
                    "severity": "critical",
                    "title": "Critical issue",
                    "description": "Desc",
                    "file_path": "app.py",
                    "line_number": 2,
                }
            ]
        })

        analyzer.llm_provider.analyze_security.return_value = security_response
        analyzer.llm_provider.analyze_performance.return_value = performance_response

        findings = analyzer.analyze_code_changes([sample_file_diff])

        # Should be sorted (critical first)
        assert findings[0].severity == FindingSeverity.CRITICAL
        assert findings[1].severity == FindingSeverity.LOW
