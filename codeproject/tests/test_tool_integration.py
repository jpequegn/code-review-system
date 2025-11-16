"""
Tests for tool integration module.

Tests tool runners, parsers, unification, and analyzer integration.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, Mock

from src.tools.runner import ToolRunner, ToolExecutionError, ToolParsingError
from src.tools.parsers.pylint_parser import PylintParser
from src.tools.parsers.bandit_parser import BanditParser
from src.tools.parsers.mypy_parser import MypyParser
from src.tools.parsers.coverage_parser import CoverageParser
from src.tools.unifier import UnifiedFinding, FindingsUnifier
from src.analysis.analyzer import CodeAnalyzer
from src.analysis.diff_parser import FileDiff, CodeChange
from src.database import FindingCategory, FindingSeverity
from src.llm.provider import LLMProvider


# ============================================================================
# Test Data & Fixtures
# ============================================================================


@pytest.fixture
def sample_pylint_output():
    """Sample pylint JSON output."""
    return [
        {
            "type": "convention",
            "module": "test",
            "obj": "test_func",
            "line": 10,
            "column": 4,
            "message": "Missing docstring",
            "symbol": "missing-docstring",
            "message-id": "C0111",
        },
        {
            "type": "error",
            "module": "test",
            "obj": "bad_func",
            "line": 20,
            "column": 0,
            "message": "Undefined variable 'x'",
            "symbol": "undefined-variable",
            "message-id": "E0602",
        },
    ]


@pytest.fixture
def sample_bandit_output():
    """Sample bandit JSON output."""
    return {
        "results": [
            {
                "test_id": "B101",
                "test_name": "assert_used",
                "issue_text": "Use of assert detected. The enclosed code will be removed when compiling to optimized byte code.",
                "severity": "LOW",
                "confidence": "HIGH",
                "line_number": 15,
                "line_range": [15],
                "filename": "test.py",
            },
            {
                "test_id": "B602",
                "test_name": "shell_injection",
                "issue_text": "shell=True identified, possible security issue.",
                "severity": "HIGH",
                "confidence": "MEDIUM",
                "line_number": 45,
                "line_range": [45],
                "filename": "test.py",
            },
        ],
        "metrics": {},
    }


@pytest.fixture
def sample_mypy_output():
    """Sample mypy JSON output."""
    return [
        {
            "file": "test.py",
            "line": 10,
            "column": 5,
            "message": "Name 'x' is not defined",
            "error_code": "name-defined",
        },
        {
            "file": "test.py",
            "line": 20,
            "column": 10,
            "message": "Incompatible types in assignment",
            "error_code": "assignment",
        },
    ]


@pytest.fixture
def sample_coverage_data():
    """Sample coverage.py data."""
    return {
        "meta": {"version": "6.0"},
        "files": {
            "src/test.py": {
                "summary": {
                    "covered_lines": 60,
                    "num_statements": 100,
                    "excluded_lines": 10,
                },
                "excluded_lines": [50, 51, 52],
                "missing_lines": list(range(70, 100)),
            },
            "src/utils.py": {
                "summary": {
                    "covered_lines": 95,
                    "num_statements": 100,
                    "excluded_lines": 0,
                },
                "excluded_lines": [],
                "missing_lines": [50, 51, 52],
            },
        },
    }


# ============================================================================
# Parser Tests
# ============================================================================


class TestPylintParser:
    """Tests for PylintParser."""

    def test_parse_valid_output(self, sample_pylint_output):
        """Test parsing valid pylint output."""
        parser = PylintParser()
        findings = parser.parse(sample_pylint_output, "test.py")

        assert len(findings) == 2
        assert findings[0]["tool"] == "pylint"
        assert findings[0]["category"] == "quality"
        assert findings[0]["severity"] == "low"
        assert findings[1]["severity"] == "high"

    def test_parse_empty_output(self):
        """Test parsing empty output."""
        parser = PylintParser()
        findings = parser.parse([], "test.py")
        assert len(findings) == 0

    def test_parse_invalid_message(self):
        """Test parsing with invalid message."""
        parser = PylintParser()
        invalid_output = [{"type": "error"}]  # Missing required fields
        findings = parser.parse(invalid_output, "test.py")
        # Should skip invalid message
        assert len(findings) == 0


class TestBanditParser:
    """Tests for BanditParser."""

    def test_parse_valid_output(self, sample_bandit_output):
        """Test parsing valid bandit output."""
        parser = BanditParser()
        findings = parser.parse(sample_bandit_output, "test.py")

        assert len(findings) == 2
        assert findings[0]["tool"] == "bandit"
        assert findings[0]["category"] == "security"
        assert findings[0]["severity"] == "low"
        assert findings[1]["severity"] == "high"

    def test_parse_empty_results(self):
        """Test parsing with no results."""
        parser = BanditParser()
        output = {"results": []}
        findings = parser.parse(output, "test.py")
        assert len(findings) == 0


class TestMypyParser:
    """Tests for MypyParser."""

    def test_parse_valid_output(self, sample_mypy_output):
        """Test parsing valid mypy output."""
        parser = MypyParser()
        findings = parser.parse(sample_mypy_output, "test.py")

        assert len(findings) == 2
        assert findings[0]["tool"] == "mypy"
        assert findings[0]["category"] == "quality"
        # Error codes should affect severity
        assert findings[1]["severity"] == "high"  # assignment error

    def test_parse_empty_output(self):
        """Test parsing empty output."""
        parser = MypyParser()
        findings = parser.parse([], "test.py")
        assert len(findings) == 0


class TestCoverageParser:
    """Tests for CoverageParser."""

    def test_parse_low_coverage(self, sample_coverage_data):
        """Test parsing coverage with low percentage."""
        parser = CoverageParser()
        findings = parser.parse(sample_coverage_data)

        # Should flag src/test.py (60% coverage < 80%)
        assert len(findings) > 0
        low_coverage_finding = findings[0]
        assert low_coverage_finding["tool"] == "coverage"
        assert low_coverage_finding["category"] == "testing"
        assert low_coverage_finding["file_path"] == "src/test.py"
        assert "60.0%" in low_coverage_finding["title"]

    def test_parse_high_coverage(self, sample_coverage_data):
        """Test parsing coverage with high percentage."""
        parser = CoverageParser()
        findings = parser.parse(sample_coverage_data)

        # src/utils.py has 95% coverage, should not be flagged
        file_paths = [f["file_path"] for f in findings]
        assert "src/utils.py" not in file_paths


# ============================================================================
# Unifier Tests
# ============================================================================


class TestUnifiedFinding:
    """Tests for UnifiedFinding dataclass."""

    def test_creation(self):
        """Test creating UnifiedFinding."""
        finding = UnifiedFinding(
            file_path="test.py",
            category="security",
            severity="high",
            title="SQL Injection",
            description="Potential SQL injection vulnerability",
            line_number=10,
            combined_confidence=0.95,
        )

        assert finding.file_path == "test.py"
        assert finding.severity == "high"
        assert finding.combined_confidence == 0.95

    def test_dedup_key(self):
        """Test deduplication key generation."""
        finding = UnifiedFinding(
            file_path="test.py",
            category="security",
            severity="high",
            title="SQL Injection",
            description="",
            line_number=10,
        )

        key = finding.dedup_key()
        assert len(key) == 4
        assert key[0] == "security"
        assert key[1] == "test.py"
        assert key[2] == 10


class TestFindingsUnifier:
    """Tests for FindingsUnifier."""

    def test_deduplicate_tool_findings(self):
        """Test deduplication of findings across tools."""
        unifier = FindingsUnifier()

        tool_findings = {
            "pylint": [
                {
                    "category": "security",
                    "file_path": "test.py",
                    "line_number": 10,
                    "title": "Hard-coded password",
                    "confidence": 0.8,
                    "severity": "high",
                }
            ],
            "bandit": [
                {
                    "category": "security",
                    "file_path": "test.py",
                    "line_number": 10,
                    "title": "Hard-coded password",
                    "confidence": 0.9,
                    "severity": "high",
                }
            ],
        }

        deduplicated = unifier._deduplicate_tool_findings(tool_findings)
        # Both tools report same issue, should result in 1 deduplicated entry
        assert len(deduplicated) == 1
        # Check that both tools are recorded
        dedup_entry = list(deduplicated.values())[0]
        assert dedup_entry["tool_count"] == 2
        assert "pylint" in dedup_entry["tools"]
        assert "bandit" in dedup_entry["tools"]

    def test_confidence_boost_on_agreement(self):
        """Test confidence boost when multiple tools agree."""
        unifier = FindingsUnifier()

        finding_info = {
            "finding": {
                "category": "security",
                "severity": "high",
                "file_path": "test.py",
                "line_number": 10,
                "title": "Vulnerability",
                "description": "Test",
                "confidence": 0.8,
            },
            "tools": {"pylint": {}, "bandit": {}},
            "tool_count": 2,
            "confidence_scores": [0.8, 0.85],
        }

        unified = unifier._create_unified_finding(finding_info)
        # Base: (0.8 + 0.85) / 2 = 0.825
        # Bonus: +0.10 (2 tools * 0.05)
        # Total should be ≈ 0.925
        assert unified.combined_confidence > 0.9

    def test_unify_with_llm_findings(self):
        """Test unifying tool and LLM findings."""
        unifier = FindingsUnifier()

        tool_findings = {
            "pylint": [
                {
                    "category": "quality",
                    "file_path": "test.py",
                    "line_number": 5,
                    "title": "Missing docstring",
                    "confidence": 0.7,
                    "severity": "low",
                }
            ]
        }

        llm_findings = [
            {
                "category": "security",
                "file_path": "test.py",
                "line_number": 10,
                "title": "SQL Injection Risk",
                "confidence": 0.85,
                "severity": "high",
            }
        ]

        unified = unifier.unify_findings(tool_findings, llm_findings)
        # Should have 2 findings (no overlap)
        assert len(unified) == 2

    def test_sorting_by_severity(self):
        """Test sorting findings by severity and confidence."""
        unifier = FindingsUnifier()

        findings = [
            UnifiedFinding(
                file_path="a.py",
                category="quality",
                severity="low",
                title="Low severity",
                description="",
                combined_confidence=0.9,
            ),
            UnifiedFinding(
                file_path="b.py",
                category="security",
                severity="critical",
                title="Critical issue",
                description="",
                combined_confidence=0.7,
            ),
            UnifiedFinding(
                file_path="c.py",
                category="security",
                severity="high",
                title="High severity",
                description="",
                combined_confidence=0.95,
            ),
        ]

        sorted_findings = unifier.sort_by_severity_and_confidence(findings)
        # Critical should be first
        assert sorted_findings[0].severity == "critical"
        # High should be second
        assert sorted_findings[1].severity == "high"
        # Low should be last
        assert sorted_findings[2].severity == "low"


# ============================================================================
# Analyzer Integration Tests
# ============================================================================


class TestAnalyzerToolIntegration:
    """Tests for CodeAnalyzer with tool integration."""

    @pytest.fixture
    def mock_llm_provider(self):
        """Mock LLM provider."""
        provider = MagicMock(spec=LLMProvider)
        provider.analyze_security.return_value = json.dumps({"findings": []})
        provider.analyze_performance.return_value = json.dumps({"findings": []})
        return provider

    @pytest.fixture
    def mock_tool_runner(self):
        """Mock tool runner."""
        runner = MagicMock(spec=ToolRunner)
        runner.run_all_tools.return_value = {}
        return runner

    def test_analyzer_initialization_with_tools(self, mock_llm_provider, mock_tool_runner):
        """Test analyzer initialization with tool runner."""
        analyzer = CodeAnalyzer(
            llm_provider=mock_llm_provider,
            tool_runner=mock_tool_runner,
            use_tools=True,
        )

        assert analyzer.use_tools is True
        assert analyzer.tool_runner is mock_tool_runner

    def test_analyzer_disables_tools(self, mock_llm_provider):
        """Test analyzer can disable tool execution."""
        analyzer = CodeAnalyzer(
            llm_provider=mock_llm_provider,
            use_tools=False,
        )

        assert analyzer.use_tools is False

    def test_analyze_with_tools_and_llm(
        self,
        mock_llm_provider,
        mock_tool_runner,
    ):
        """Test full analysis pipeline with tools and LLM."""
        # Setup
        analyzer = CodeAnalyzer(
            llm_provider=mock_llm_provider,
            tool_runner=mock_tool_runner,
            use_tools=True,
        )

        mock_tool_runner.run_all_tools.return_value = {
            "pylint": [
                {
                    "category": "quality",
                    "severity": "medium",
                    "file_path": "test.py",
                    "line_number": 10,
                    "title": "Style issue",
                    "description": "Line too long",
                    "confidence": 0.8,
                }
            ],
            "bandit": [],
            "mypy": [],
            "coverage": [],
        }

        mock_llm_provider.analyze_security.return_value = json.dumps({"findings": []})
        mock_llm_provider.analyze_performance.return_value = json.dumps({"findings": []})

        # Create sample diff
        change = CodeChange(
            file_path="test.py",
            line_number=10,
            old_line_number=10,
            change_type="add",
            content="x = 1",
            context_before=[],
            context_after=[],
        )
        file_diff = FileDiff(
            file_path="test.py",
            old_path=None,
            is_binary=False,
            additions=1,
            deletions=0,
            changes=[change],
        )

        # Execute
        findings = analyzer.analyze_code_changes([file_diff])

        # Verify
        assert len(findings) >= 0  # May have tool findings
        mock_tool_runner.run_all_tools.assert_called_once()

    def test_tool_execution_error_handled(self, mock_llm_provider):
        """Test that tool execution errors are handled gracefully."""
        mock_runner = MagicMock(spec=ToolRunner)
        mock_runner.run_all_tools.side_effect = ToolExecutionError("pylint failed")

        analyzer = CodeAnalyzer(
            llm_provider=mock_llm_provider,
            tool_runner=mock_runner,
            use_tools=True,
        )

        mock_llm_provider.analyze_security.return_value = json.dumps({"findings": []})
        mock_llm_provider.analyze_performance.return_value = json.dumps({"findings": []})

        change = CodeChange(
            file_path="test.py",
            line_number=1,
            old_line_number=1,
            change_type="add",
            content="x = 1",
            context_before=[],
            context_after=[],
        )
        file_diff = FileDiff(
            file_path="test.py",
            old_path=None,
            is_binary=False,
            additions=1,
            deletions=0,
            changes=[change],
        )

        # Should not raise, but handle error gracefully
        findings = analyzer.analyze_code_changes([file_diff])
        assert isinstance(findings, list)


# ============================================================================
# End-to-End Tests
# ============================================================================


class TestEndToEnd:
    """End-to-end integration tests."""

    def test_full_tool_pipeline(self):
        """Test complete tool pipeline."""
        parser_py = PylintParser()
        parser_bandit = BanditParser()
        unifier = FindingsUnifier()

        # Sample outputs
        pylint_out = [
            {
                "type": "error",
                "line": 10,
                "column": 5,
                "message": "Undefined variable",
                "symbol": "undefined-variable",
            }
        ]
        bandit_out = {
            "results": [
                {
                    "test_id": "B602",
                    "test_name": "shell_injection",
                    "issue_text": "shell=True found",
                    "severity": "HIGH",
                    "confidence": "HIGH",
                    "line_number": 10,
                }
            ]
        }

        # Parse
        pylint_findings = parser_py.parse(pylint_out, "test.py")
        bandit_findings = parser_bandit.parse(bandit_out, "test.py")

        # Unify
        tool_findings = {"pylint": pylint_findings, "bandit": bandit_findings}
        unified = unifier.unify_findings(tool_findings)

        # Verify
        assert len(unified) > 0
        # Findings should have confidence scores
        for finding in unified:
            assert 0.0 <= finding.combined_confidence <= 1.0
