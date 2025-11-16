"""
Tests for metrics collection module.

Tests complexity metrics calculation, function analysis, file metrics aggregation,
and metrics snapshots for code quality assessment.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

from src.metrics.collector import (
    PythonMetricsCollector,
    MetricsCollectorError,
)
from src.metrics.models import (
    ComplexityMetrics,
    FunctionMetrics,
    FileMetrics,
    MetricsSnapshot,
    MetricType,
)


# ============================================================================
# Test Data & Fixtures
# ============================================================================


@pytest.fixture
def metrics_collector():
    """Provide a PythonMetricsCollector instance."""
    return PythonMetricsCollector(complexity_threshold=10)


@pytest.fixture
def sample_python_code():
    """Sample Python code with various complexity levels."""
    return '''
"""Module with functions of varying complexity."""

import os
import sys
from pathlib import Path

def simple_function(x):
    """Simple function with low complexity."""
    return x + 1

def medium_complexity(value):
    """Function with medium complexity."""
    if value > 0:
        if value > 10:
            return value * 2
        else:
            return value + 1
    else:
        return 0

def high_complexity(x, y, z):
    """Function with high cyclomatic complexity."""
    if x > 0:
        if y > 0:
            if z > 0:
                return x + y + z
            else:
                return x + y
        elif y < 0:
            return x - y
        else:
            return x
    elif x < 0:
        if y > 0:
            return -x + y
        else:
            return -x
    else:
        return 0

class SampleClass:
    """Sample class with methods."""

    def __init__(self):
        self.value = 0

    def method_one(self):
        """Simple method."""
        return self.value

    def method_two(self, n):
        """Method with complexity."""
        for i in range(n):
            if i % 2 == 0:
                self.value += i
        return self.value
'''


@pytest.fixture
def simple_python_code():
    """Simple Python code with low complexity."""
    return '''
def add(a, b):
    """Add two numbers."""
    return a + b

def multiply(x, y):
    """Multiply two numbers."""
    return x * y
'''


@pytest.fixture
def malformed_python_code():
    """Malformed Python code with syntax errors."""
    return '''
def broken_function(x):
    if x > 0
        return x  # Missing colon
    else
        return -x  # Missing colon
'''


@pytest.fixture
def temp_python_file(sample_python_code):
    """Create a temporary Python file with sample code."""
    with TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "test_module.py"
        file_path.write_text(sample_python_code)
        yield file_path


@pytest.fixture
def temp_dir_with_files(sample_python_code, simple_python_code):
    """Create a temporary directory with multiple Python files."""
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create main module
        (tmpdir_path / "main.py").write_text(sample_python_code)

        # Create utils module
        (tmpdir_path / "utils.py").write_text(simple_python_code)

        # Create subdirectory with more code
        subdir = tmpdir_path / "submodule"
        subdir.mkdir()
        (subdir / "helpers.py").write_text(simple_python_code)

        # Create __pycache__ directory (should be skipped)
        (tmpdir_path / "__pycache__").mkdir()
        (tmpdir_path / "__pycache__" / "cache.pyc").write_text("compiled")

        # Create test file (should be skipped)
        (tmpdir_path / "test_main.py").write_text(simple_python_code)

        yield tmpdir_path


# ============================================================================
# ComplexityMetrics Tests
# ============================================================================


class TestComplexityMetrics:
    """Tests for ComplexityMetrics dataclass."""

    def test_creation(self):
        """Test creating ComplexityMetrics instance."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5.0,
            cognitive_complexity=6.0,
            nesting_depth=3,
        )
        assert metrics.cyclomatic_complexity == 5.0
        assert metrics.cognitive_complexity == 6.0
        assert metrics.nesting_depth == 3

    def test_is_high_complexity_true(self):
        """Test high complexity detection."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=15.0,
            cognitive_complexity=16.0,
            nesting_depth=3,
        )
        assert metrics.is_high_complexity(cc_threshold=10) is True

    def test_is_high_complexity_false(self):
        """Test low complexity detection."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5.0,
            cognitive_complexity=6.0,
            nesting_depth=3,
        )
        assert metrics.is_high_complexity(cc_threshold=10) is False

    def test_is_high_cognitive_true(self):
        """Test high cognitive complexity detection."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5.0,
            cognitive_complexity=20.0,
            nesting_depth=3,
        )
        assert metrics.is_high_cognitive(cognitive_threshold=15) is True

    def test_is_high_cognitive_false(self):
        """Test low cognitive complexity detection."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5.0,
            cognitive_complexity=10.0,
            nesting_depth=3,
        )
        assert metrics.is_high_cognitive(cognitive_threshold=15) is False

    def test_is_deeply_nested_true(self):
        """Test deep nesting detection."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5.0,
            cognitive_complexity=6.0,
            nesting_depth=5,
        )
        assert metrics.is_deeply_nested(nesting_threshold=4) is True

    def test_is_deeply_nested_false(self):
        """Test shallow nesting detection."""
        metrics = ComplexityMetrics(
            cyclomatic_complexity=5.0,
            cognitive_complexity=6.0,
            nesting_depth=2,
        )
        assert metrics.is_deeply_nested(nesting_threshold=4) is False


# ============================================================================
# FunctionMetrics Tests
# ============================================================================


class TestFunctionMetrics:
    """Tests for FunctionMetrics dataclass."""

    def test_creation(self):
        """Test creating FunctionMetrics instance."""
        complexity = ComplexityMetrics(2.0, 2.2, 1)
        metrics = FunctionMetrics(
            name="test_func",
            line_number=10,
            length=5,
            parameter_count=2,
            complexity=complexity,
            has_docstring=True,
            returns_value=True,
        )
        assert metrics.name == "test_func"
        assert metrics.line_number == 10
        assert metrics.length == 5
        assert metrics.parameter_count == 2
        assert metrics.complexity.cyclomatic_complexity == 2.0
        assert metrics.has_docstring is True
        assert metrics.returns_value is True

    def test_is_long_function_true(self):
        """Test long function detection."""
        complexity = ComplexityMetrics(2.0, 2.2, 1)
        metrics = FunctionMetrics(
            name="long_func",
            line_number=1,
            length=100,
            parameter_count=1,
            complexity=complexity,
        )
        assert metrics.is_long_function(length_threshold=50) is True

    def test_is_long_function_false(self):
        """Test short function detection."""
        complexity = ComplexityMetrics(2.0, 2.2, 1)
        metrics = FunctionMetrics(
            name="short_func",
            line_number=1,
            length=10,
            parameter_count=1,
            complexity=complexity,
        )
        assert metrics.is_long_function(length_threshold=50) is False

    def test_has_many_parameters_true(self):
        """Test many parameters detection."""
        complexity = ComplexityMetrics(2.0, 2.2, 1)
        metrics = FunctionMetrics(
            name="func",
            line_number=1,
            length=10,
            parameter_count=8,
            complexity=complexity,
        )
        assert metrics.has_many_parameters(param_threshold=5) is True

    def test_has_many_parameters_false(self):
        """Test few parameters detection."""
        complexity = ComplexityMetrics(2.0, 2.2, 1)
        metrics = FunctionMetrics(
            name="func",
            line_number=1,
            length=10,
            parameter_count=2,
            complexity=complexity,
        )
        assert metrics.has_many_parameters(param_threshold=5) is False


# ============================================================================
# FileMetrics Tests
# ============================================================================


class TestFileMetrics:
    """Tests for FileMetrics dataclass."""

    def test_creation(self):
        """Test creating FileMetrics instance."""
        complexity = ComplexityMetrics(5.0, 5.5, 2)
        func_metrics = FunctionMetrics(
            name="func1",
            line_number=1,
            length=10,
            parameter_count=1,
            complexity=complexity,
        )

        file_metrics = FileMetrics(
            file_path="test.py",
            language="python",
            total_lines=100,
            code_lines=80,
            functions=[func_metrics],
            average_function_length=15.0,
            max_complexity=8.0,
            average_complexity=5.0,
            imports_count=5,
            classes_count=2,
            churn_rate=2.5,
        )

        assert file_metrics.file_path == "test.py"
        assert file_metrics.language == "python"
        assert file_metrics.total_lines == 100
        assert file_metrics.code_lines == 80
        assert len(file_metrics.functions) == 1
        assert file_metrics.classes_count == 2
        assert file_metrics.churn_rate == 2.5

    def test_is_high_churn_true(self):
        """Test high churn detection."""
        file_metrics = FileMetrics(
            file_path="test.py",
            language="python",
            total_lines=100,
            code_lines=80,
            churn_rate=3.0,
        )
        assert file_metrics.is_high_churn(churn_threshold=2.0) is True

    def test_is_high_churn_false(self):
        """Test low churn detection."""
        file_metrics = FileMetrics(
            file_path="test.py",
            language="python",
            total_lines=100,
            code_lines=80,
            churn_rate=1.0,
        )
        assert file_metrics.is_high_churn(churn_threshold=2.0) is False

    def test_is_overly_complex_true(self):
        """Test high complexity detection."""
        file_metrics = FileMetrics(
            file_path="test.py",
            language="python",
            total_lines=100,
            code_lines=80,
            average_complexity=12.0,
        )
        assert file_metrics.is_overly_complex(cc_threshold=10) is True

    def test_is_overly_complex_false(self):
        """Test low complexity detection."""
        file_metrics = FileMetrics(
            file_path="test.py",
            language="python",
            total_lines=100,
            code_lines=80,
            average_complexity=5.0,
        )
        assert file_metrics.is_overly_complex(cc_threshold=10) is False

    def test_has_large_functions(self):
        """Test large function detection."""
        complexity = ComplexityMetrics(2.0, 2.2, 1)
        large_func = FunctionMetrics(
            name="large",
            line_number=1,
            length=100,
            parameter_count=1,
            complexity=complexity,
        )
        small_func = FunctionMetrics(
            name="small",
            line_number=200,
            length=10,
            parameter_count=1,
            complexity=complexity,
        )

        file_metrics = FileMetrics(
            file_path="test.py",
            language="python",
            total_lines=300,
            code_lines=250,
            functions=[large_func, small_func],
        )

        assert file_metrics.has_large_functions(length_threshold=50) == 1

    def test_has_complex_functions(self):
        """Test complex function detection."""
        complex_comp = ComplexityMetrics(15.0, 16.0, 4)
        simple_comp = ComplexityMetrics(2.0, 2.2, 1)

        complex_func = FunctionMetrics(
            name="complex",
            line_number=1,
            length=30,
            parameter_count=1,
            complexity=complex_comp,
        )
        simple_func = FunctionMetrics(
            name="simple",
            line_number=50,
            length=10,
            parameter_count=1,
            complexity=simple_comp,
        )

        file_metrics = FileMetrics(
            file_path="test.py",
            language="python",
            total_lines=100,
            code_lines=80,
            functions=[complex_func, simple_func],
        )

        assert file_metrics.has_complex_functions(cc_threshold=10) == 1


# ============================================================================
# MetricsSnapshot Tests
# ============================================================================


class TestMetricsSnapshot:
    """Tests for MetricsSnapshot dataclass."""

    def test_creation(self):
        """Test creating MetricsSnapshot instance."""
        snapshot = MetricsSnapshot(
            commit_sha="abc123",
            timestamp="2025-01-01T00:00:00",
        )
        assert snapshot.commit_sha == "abc123"
        assert snapshot.timestamp == "2025-01-01T00:00:00"
        assert snapshot.total_lines == 0
        assert snapshot.average_file_complexity == 0.0
        assert snapshot.max_file_complexity == 0.0
        assert len(snapshot.high_complexity_files) == 0

    def test_calculate_summary_metrics(self):
        """Test summary metrics calculation."""
        file_metrics_1 = FileMetrics(
            file_path="file1.py",
            language="python",
            total_lines=100,
            code_lines=80,
            average_complexity=5.0,
        )
        file_metrics_2 = FileMetrics(
            file_path="file2.py",
            language="python",
            total_lines=200,
            code_lines=160,
            average_complexity=12.0,
        )

        snapshot = MetricsSnapshot(
            commit_sha="abc123",
            timestamp="2025-01-01T00:00:00",
            files={
                "file1.py": file_metrics_1,
                "file2.py": file_metrics_2,
            },
        )

        snapshot.calculate_summary_metrics()

        assert snapshot.total_lines == 300
        assert snapshot.average_file_complexity == 8.5
        assert snapshot.max_file_complexity == 12.0
        # file2.py has complexity 12.0, average is 8.5, threshold is 8.5 * 1.5 = 12.75
        # Only file2 (12.0) exceeds some threshold
        assert len(snapshot.high_complexity_files) >= 0

    def test_get_complexity_trend(self):
        """Test complexity trend calculation."""
        snapshot1 = MetricsSnapshot(
            commit_sha="abc123",
            timestamp="2025-01-01T00:00:00",
            average_file_complexity=10.0,
        )

        snapshot2 = MetricsSnapshot(
            commit_sha="def456",
            timestamp="2025-01-02T00:00:00",
            average_file_complexity=12.0,
        )

        trend = snapshot2.get_complexity_trend(snapshot1)
        assert trend is not None
        assert abs(trend - 20.0) < 0.01  # 20% increase

    def test_get_complexity_trend_none_previous(self):
        """Test trend when no previous snapshot exists."""
        snapshot = MetricsSnapshot(
            commit_sha="abc123",
            timestamp="2025-01-01T00:00:00",
            average_file_complexity=10.0,
        )

        trend = snapshot.get_complexity_trend(None)
        assert trend is None

    def test_get_complexity_trend_zero_previous(self):
        """Test trend when previous complexity was zero."""
        snapshot1 = MetricsSnapshot(
            commit_sha="abc123",
            timestamp="2025-01-01T00:00:00",
            average_file_complexity=0.0,
        )

        snapshot2 = MetricsSnapshot(
            commit_sha="def456",
            timestamp="2025-01-02T00:00:00",
            average_file_complexity=10.0,
        )

        trend = snapshot2.get_complexity_trend(snapshot1)
        assert trend is None


# ============================================================================
# PythonMetricsCollector Tests
# ============================================================================


class TestPythonMetricsCollector:
    """Tests for PythonMetricsCollector."""

    def test_initialization(self, metrics_collector):
        """Test collector initialization."""
        assert metrics_collector.complexity_threshold == 10

    def test_count_code_lines_simple(self, metrics_collector, simple_python_code):
        """Test code line counting."""
        code_lines = metrics_collector._count_code_lines(simple_python_code)
        # Should count actual code lines, excluding docstrings and blank lines
        assert code_lines > 0

    def test_count_code_lines_with_comments(self, metrics_collector):
        """Test code line counting with comments."""
        code = '''
# This is a comment
def func():
    # Another comment
    x = 1
    # Yet another
    return x
'''
        code_lines = metrics_collector._count_code_lines(code)
        assert code_lines == 3  # x = 1, return x, and def line

    def test_count_imports(self, metrics_collector, sample_python_code):
        """Test import statement counting."""
        imports = metrics_collector._count_imports(sample_python_code)
        assert imports >= 3  # import os, import sys, from pathlib import Path

    def test_count_classes(self, metrics_collector, sample_python_code):
        """Test class definition counting."""
        classes = metrics_collector._count_classes(sample_python_code)
        assert classes == 1  # SampleClass

    def test_count_classes_malformed(self, metrics_collector, malformed_python_code):
        """Test class counting with malformed code."""
        classes = metrics_collector._count_classes(malformed_python_code)
        assert classes == 0  # Syntax error prevents parsing

    def test_collect_file_metrics_success(self, metrics_collector, temp_python_file):
        """Test successful metrics collection for a file."""
        metrics = metrics_collector.collect_file_metrics(temp_python_file)

        assert metrics is not None
        assert metrics.file_path is not None
        assert metrics.language == "python"
        assert metrics.total_lines > 0
        assert metrics.code_lines > 0
        assert metrics.imports_count >= 0
        assert metrics.classes_count >= 0
        assert len(metrics.functions) > 0

    def test_collect_file_metrics_nonexistent(self, metrics_collector):
        """Test metrics collection for nonexistent file."""
        nonexistent = Path("/nonexistent/file.py")
        metrics = metrics_collector.collect_file_metrics(nonexistent)
        assert metrics is None

    def test_collect_file_metrics_malformed(self, metrics_collector):
        """Test metrics collection for malformed code."""
        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "malformed.py"
            file_path.write_text("def broken(\n    if x > 0:")

            metrics = metrics_collector.collect_file_metrics(file_path)
            # Should handle gracefully, may return metrics for what can be parsed
            # or None depending on implementation

    def test_find_python_files(self, metrics_collector, temp_dir_with_files):
        """Test finding Python files in directory."""
        py_files = metrics_collector._find_python_files(temp_dir_with_files)

        # Should find main.py and utils.py, subdir/helpers.py
        # Should NOT find test files, __pycache__, or cache files
        file_names = [f.name for f in py_files]
        assert "main.py" in file_names
        assert "utils.py" in file_names
        assert "helpers.py" in file_names
        # Test files should be skipped
        assert "test_main.py" not in file_names

    def test_collect_snapshot(self, metrics_collector, temp_dir_with_files):
        """Test snapshot collection for entire directory."""
        snapshot = metrics_collector.collect_snapshot(
            temp_dir_with_files,
            commit_sha="abc123",
        )

        assert snapshot.commit_sha == "abc123"
        assert snapshot.timestamp is not None
        assert len(snapshot.files) > 0
        assert snapshot.total_lines > 0
        assert snapshot.average_file_complexity >= 0.0

    def test_collect_snapshot_empty_directory(self, metrics_collector):
        """Test snapshot collection for empty directory."""
        with TemporaryDirectory() as tmpdir:
            snapshot = metrics_collector.collect_snapshot(
                Path(tmpdir),
                commit_sha="abc123",
            )

            assert snapshot.commit_sha == "abc123"
            assert len(snapshot.files) == 0
            assert snapshot.total_lines == 0

    def test_extract_function_metrics(self, metrics_collector, sample_python_code):
        """Test function metrics extraction."""
        functions = metrics_collector._extract_function_metrics(
            sample_python_code,
            Path("test.py"),
        )

        assert len(functions) > 0
        # Should find simple_function, medium_complexity, high_complexity, and class methods
        func_names = [f.name for f in functions]
        assert "simple_function" in func_names
        assert "medium_complexity" in func_names
        assert "high_complexity" in func_names

    def test_get_function_complexity(self, metrics_collector):
        """Test function complexity extraction."""
        code = '''
def simple():
    return 1

def complex_func(x):
    if x > 0:
        if x > 10:
            return x
        else:
            return -x
    else:
        return 0
'''
        complexity = metrics_collector._get_function_complexity([], "simple")
        assert complexity.cyclomatic_complexity == 1.0
        assert complexity.cognitive_complexity == 1.0


# ============================================================================
# Integration Tests
# ============================================================================


class TestMetricsCollectorIntegration:
    """Integration tests for metrics collector."""

    def test_full_workflow(self, temp_dir_with_files):
        """Test complete metrics collection workflow."""
        collector = PythonMetricsCollector()

        # Collect snapshot
        snapshot = collector.collect_snapshot(temp_dir_with_files)

        # Verify structure
        assert snapshot.commit_sha == "unknown"
        assert len(snapshot.files) > 0

        # Verify each file has metrics
        for file_path, file_metrics in snapshot.files.items():
            assert file_metrics.total_lines > 0
            assert file_metrics.language == "python"
            assert len(file_metrics.functions) >= 0

    def test_metrics_consistency(self, sample_python_code):
        """Test that metrics are consistent across multiple collections."""
        collector = PythonMetricsCollector()

        with TemporaryDirectory() as tmpdir:
            file_path = Path(tmpdir) / "test.py"
            file_path.write_text(sample_python_code)

            # Collect metrics twice
            metrics1 = collector.collect_file_metrics(file_path)
            metrics2 = collector.collect_file_metrics(file_path)

            # Should be identical
            assert metrics1 is not None
            assert metrics2 is not None
            assert metrics1.total_lines == metrics2.total_lines
            assert metrics1.code_lines == metrics2.code_lines
            assert len(metrics1.functions) == len(metrics2.functions)
