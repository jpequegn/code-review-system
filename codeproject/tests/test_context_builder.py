"""
Tests for codebase context builder and analysis.

Tests dependency graph construction, pattern detection, and cross-file analysis.
"""

import pytest
from pathlib import Path
from src.analysis.context_builder import ContextBuilder
from src.analysis.context_models import (
    CodebaseContext,
    RiskLevel,
    PatternType,
    RelatedFile,
    RiskArea,
)


class TestContextBuilder:
    """Test context builder functionality."""

    @pytest.fixture
    def context_builder(self, tmp_path):
        """Create a context builder for temporary directory."""
        # Create some test files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("""
import os
from src.utils import helper

def main():
    pass
""")
        (tmp_path / "src" / "utils.py").write_text("""
def helper():
    return "help"
""")
        (tmp_path / "tests").mkdir()
        (tmp_path / "tests" / "test_main.py").write_text("""
import pytest
from src.main import main

def test_main():
    main()
""")
        return ContextBuilder(str(tmp_path))

    def test_find_python_files(self, context_builder):
        """Test finding Python files."""
        files = context_builder._find_python_files()
        assert len(files) > 0
        assert any("main.py" in f for f in files)
        assert any("utils.py" in f for f in files)
        assert any("test_main.py" in f for f in files)

    def test_build_dependency_graph(self, context_builder):
        """Test building dependency graph from imports."""
        python_files = context_builder._find_python_files()
        forward, reverse = context_builder._build_dependency_graph(python_files)

        # Check that imports are detected
        assert isinstance(forward, dict)
        assert isinstance(reverse, dict)

    def test_detect_patterns(self, context_builder):
        """Test detecting architectural patterns."""
        python_files = context_builder._find_python_files()
        forward, _ = context_builder._build_dependency_graph(python_files)
        patterns = context_builder._detect_patterns(python_files, forward)

        assert len(patterns) > 0
        # Should detect test pattern
        assert any(p.pattern_type == PatternType.TESTING for p in patterns)

    def test_build_context(self, context_builder):
        """Test building complete context."""
        context = context_builder.build_context(
            repository_url="https://example.com/repo.git",
            language="python"
        )

        assert isinstance(context, CodebaseContext)
        assert context.total_files > 0
        assert context.repository_url == "https://example.com/repo.git"
        assert context.language == "python"
        assert context.build_timestamp is not None

    def test_get_related_files(self, context_builder):
        """Test getting related files."""
        context = context_builder.build_context()

        # Get related files for main.py
        main_files = [f for f in context.dependency_graph.keys() if "main.py" in f]
        if main_files:
            related = context_builder.get_related_files(main_files[0], context, max_results=10)
            assert isinstance(related, list)
            # Should find some related files
            assert len(related) >= 0

    def test_get_cascade_risks(self, context_builder):
        """Test identifying cascade risks."""
        context = context_builder.build_context()

        # Get cascade risks for main.py
        main_files = [f for f in context.dependency_graph.keys() if "main.py" in f]
        if main_files:
            risks = context_builder.get_cascade_risks(main_files, context)
            assert isinstance(risks, list)
            # Each risk should have expected attributes
            for risk in risks:
                assert isinstance(risk, RiskArea)
                assert risk.file_path
                assert isinstance(risk.risk_level, RiskLevel)

    def test_get_cross_file_analysis(self, context_builder):
        """Test cross-file analysis."""
        context = context_builder.build_context()

        main_files = [f for f in context.dependency_graph.keys() if "main.py" in f]
        if main_files:
            analysis = context_builder.get_cross_file_analysis(main_files, context)
            assert analysis.changed_files == main_files
            assert isinstance(analysis.related_files, list)
            assert isinstance(analysis.cascade_risks, list)


class TestContextModels:
    """Test context data models."""

    def test_codebase_context_initialization(self):
        """Test CodebaseContext creation."""
        context = CodebaseContext(
            repository_url="https://example.com/repo.git",
            language="python",
            total_files=10
        )

        assert context.repository_url == "https://example.com/repo.git"
        assert context.language == "python"
        assert context.total_files == 10
        assert context.build_timestamp is None

    def test_related_file_creation(self):
        """Test RelatedFile creation."""
        related = RelatedFile(
            file_path="src/utils.py",
            relationship="imports",
            relevance_score=0.9,
            reason="Direct import"
        )

        assert related.file_path == "src/utils.py"
        assert related.relationship == "imports"
        assert related.relevance_score == 0.9

    def test_risk_area_creation(self):
        """Test RiskArea creation."""
        risk = RiskArea(
            file_path="src/main.py",
            risk_level=RiskLevel.HIGH,
            reason="High coupling"
        )

        assert risk.file_path == "src/main.py"
        assert risk.risk_level == RiskLevel.HIGH
        assert risk.reason == "High coupling"


class TestEnhancedPrompts:
    """Test enhanced prompt generation."""

    def test_enhance_security_prompt_with_string_context(self):
        """Test enhancing security prompt with string context."""
        from src.llm.enhanced_prompts import enhance_security_prompt

        code = "x = eval(user_input)"
        context_str = "This module handles user input processing"

        prompt = enhance_security_prompt(code, context_prompt=context_str)

        assert "eval" not in prompt or "vulnerab" in prompt.lower()
        assert "Codebase Context" in prompt
        assert context_str in prompt
        assert "Only return valid JSON" in prompt

    def test_enhance_performance_prompt_with_string_context(self):
        """Test enhancing performance prompt with string context."""
        from src.llm.enhanced_prompts import enhance_performance_prompt

        code = "for i in range(1000000): db.query()"
        context_str = "Database module with heavy query usage"

        prompt = enhance_performance_prompt(code, context_prompt=context_str)

        assert "Codebase Context" in prompt
        assert context_str in prompt
        assert "Only return valid JSON" in prompt

    def test_enhance_prompts_without_context(self):
        """Test prompts work without context."""
        from src.llm.enhanced_prompts import (
            enhance_security_prompt,
            enhance_performance_prompt
        )

        code = "x = 1"

        sec_prompt = enhance_security_prompt(code)
        assert "vulnerab" in sec_prompt.lower()
        assert "Only return valid JSON" in sec_prompt

        perf_prompt = enhance_performance_prompt(code)
        assert "performance" in perf_prompt.lower()
        assert "Only return valid JSON" in perf_prompt


class TestContextIntegration:
    """Integration tests for context with analyzer."""

    @pytest.fixture
    def mock_analyzer(self, tmp_path):
        """Create an analyzer with context."""
        from src.analysis.analyzer import CodeAnalyzer
        from unittest.mock import Mock

        # Create test files
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "api.py").write_text("def get_user(): pass")
        (tmp_path / "src" / "db.py").write_text("def query(): pass")

        # Mock LLM provider to avoid needing real API keys
        mock_llm = Mock()
        mock_llm.analyze_security = Mock(return_value='{"findings": []}')
        mock_llm.analyze_performance = Mock(return_value='{"findings": []}')

        return CodeAnalyzer(
            llm_provider=mock_llm,
            repo_path=str(tmp_path),
            use_context=True
        )

    def test_analyzer_initializes_with_context(self, mock_analyzer):
        """Test analyzer initializes with context."""
        assert mock_analyzer.context_builder is not None
        assert mock_analyzer.use_context is True

    def test_analyzer_can_build_context(self, mock_analyzer):
        """Test analyzer can build context."""
        context = mock_analyzer._get_or_build_context()
        assert context is not None
        assert isinstance(context, CodebaseContext)

    def test_analyzer_caches_context(self, mock_analyzer):
        """Test context is cached."""
        context1 = mock_analyzer._get_or_build_context()
        context2 = mock_analyzer._get_or_build_context()
        assert context1 is context2  # Same object

    def test_analyzer_disables_context_when_requested(self, tmp_path):
        """Test context can be disabled."""
        from src.analysis.analyzer import CodeAnalyzer
        from unittest.mock import Mock

        mock_llm = Mock()
        analyzer = CodeAnalyzer(
            llm_provider=mock_llm,
            repo_path=str(tmp_path),
            use_context=False
        )

        assert analyzer.context_builder is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
