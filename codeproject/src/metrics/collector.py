"""
Metrics Collection Module

Provides abstraction for collecting code metrics across different languages.
Implements complexity, maintainability, and structural metrics.
"""

import logging
import ast
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timezone

from radon.complexity import cc_visit, SCORE
from radon.metrics import mi_visit, mi_parameters

from src.metrics.models import (
    ComplexityMetrics,
    FunctionMetrics,
    FileMetrics,
    MetricsSnapshot,
)

logger = logging.getLogger(__name__)


class MetricsCollectorError(Exception):
    """Base exception for metrics collection errors."""
    pass


class MetricsCollector(ABC):
    """
    Abstract base class for metrics collection.

    Subclasses implement language-specific metrics collection.
    """

    @abstractmethod
    def collect_file_metrics(self, file_path: Path) -> Optional[FileMetrics]:
        """
        Collect metrics for a single file.

        Args:
            file_path: Path to file to analyze

        Returns:
            FileMetrics object or None if file cannot be analyzed
        """
        pass

    @abstractmethod
    def collect_snapshot(self, repo_path: Path) -> MetricsSnapshot:
        """
        Collect complete metrics snapshot for a repository.

        Args:
            repo_path: Root path of repository

        Returns:
            MetricsSnapshot with all collected metrics
        """
        pass


class PythonMetricsCollector(MetricsCollector):
    """
    Metrics collector for Python code.

    Uses radon for complexity metrics and AST analysis for structural metrics.
    """

    CODE_EXTENSIONS = {".py"}
    SKIP_PATTERNS = {"__pycache__", ".venv", "venv", "site-packages", ".egg-info"}

    def __init__(self, complexity_threshold: float = 10):
        """
        Initialize Python metrics collector.

        Args:
            complexity_threshold: CC threshold for flagging high complexity
        """
        self.complexity_threshold = complexity_threshold

    def collect_file_metrics(self, file_path: Path) -> Optional[FileMetrics]:
        """
        Collect metrics for a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            FileMetrics or None if file cannot be analyzed
        """
        if not file_path.exists():
            logger.warning(f"File does not exist: {file_path}")
            return None

        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Calculate metrics
            total_lines = len(content.splitlines())
            code_lines = self._count_code_lines(content)
            functions = self._extract_function_metrics(content, file_path)

            # Calculate averages
            avg_function_length = (
                sum(f.length for f in functions) / len(functions)
                if functions
                else 0
            )
            avg_complexity = (
                sum(f.complexity.cyclomatic_complexity for f in functions) / len(functions)
                if functions
                else 0
            )
            max_complexity = (
                max(f.complexity.cyclomatic_complexity for f in functions)
                if functions
                else 0
            )

            # Count imports and classes
            imports_count = self._count_imports(content)
            classes_count = self._count_classes(content)

            # Calculate maintainability index
            mi_score = self._calculate_maintainability_index(content)

            return FileMetrics(
                file_path=str(file_path.relative_to(file_path.parent.parent.parent)),
                language="python",
                total_lines=total_lines,
                code_lines=code_lines,
                functions=functions,
                average_function_length=avg_function_length,
                max_complexity=max_complexity,
                average_complexity=avg_complexity,
                imports_count=imports_count,
                classes_count=classes_count,
            )

        except Exception as e:
            logger.error(f"Error collecting metrics for {file_path}: {e}")
            return None

    def collect_snapshot(
        self, repo_path: Path, commit_sha: Optional[str] = None
    ) -> MetricsSnapshot:
        """
        Collect complete metrics snapshot for repository.

        Args:
            repo_path: Root path of repository
            commit_sha: Optional git commit SHA for this snapshot

        Returns:
            MetricsSnapshot with all collected metrics
        """
        files_metrics: Dict[str, FileMetrics] = {}

        # Walk repository and collect metrics for all Python files
        for py_file in self._find_python_files(repo_path):
            try:
                file_metrics = self.collect_file_metrics(py_file)
                if file_metrics:
                    files_metrics[str(py_file)] = file_metrics
            except Exception as e:
                logger.warning(f"Skipping {py_file}: {e}")

        # Create snapshot
        snapshot = MetricsSnapshot(
            commit_sha=commit_sha or "unknown",
            timestamp=datetime.now(timezone.utc).isoformat(),
            files=files_metrics,
        )

        # Calculate summary metrics
        snapshot.calculate_summary_metrics()

        return snapshot

    def _find_python_files(self, root_path: Path) -> List[Path]:
        """Find all Python files in directory, excluding test/config files."""
        python_files = []

        for item in root_path.rglob("*.py"):
            # Skip directories
            if item.is_dir():
                continue

            # Skip if in excluded directory
            if any(skip in item.parts for skip in self.SKIP_PATTERNS):
                continue

            # Skip test files (optional - can be included if needed)
            if "test" in item.name or "conftest" in item.name:
                continue

            python_files.append(item)

        return sorted(python_files)

    def _count_code_lines(self, content: str) -> int:
        """Count lines of actual code (excluding blank lines and comments)."""
        code_lines = 0
        for line in content.splitlines():
            stripped = line.strip()
            if stripped and not stripped.startswith("#"):
                code_lines += 1
        return code_lines

    def _extract_function_metrics(
        self, content: str, file_path: Path
    ) -> List[FunctionMetrics]:
        """Extract metrics for all functions in a file."""
        functions = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            logger.warning(f"Syntax error in {file_path}, skipping function analysis")
            return functions

        # Get complexity from radon
        try:
            cc_results = cc_visit(content)
        except Exception as e:
            logger.warning(f"Error calculating complexity in {file_path}: {e}")
            cc_results = []

        # Extract function definitions
        function_nodes = [
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
        ]

        for func_node in function_nodes:
            # Get complexity for this function
            complexity = self._get_function_complexity(cc_results, func_node.name)

            # Extract basic info
            func_metrics = FunctionMetrics(
                name=func_node.name,
                line_number=func_node.lineno,
                length=func_node.end_lineno - func_node.lineno + 1 if func_node.end_lineno else 0,
                parameter_count=len(func_node.args.args),
                complexity=complexity,
                has_docstring=ast.get_docstring(func_node) is not None,
                returns_value=self._has_return_statement(func_node),
            )
            functions.append(func_metrics)

        return sorted(functions, key=lambda f: f.line_number)

    def _get_function_complexity(
        self, cc_results: List, func_name: str
    ) -> ComplexityMetrics:
        """
        Get complexity metrics for a function from radon results.

        Args:
            cc_results: Results from radon.complexity.cc_visit()
            func_name: Name of function to find

        Returns:
            ComplexityMetrics with calculated values
        """
        # Find matching function in results
        for result in cc_results:
            if hasattr(result, "name") and result.name == func_name:
                # radon returns McCabe complexity, use as cyclomatic
                cyclomatic = result.complexity
                # Estimate cognitive complexity (heuristic: similar to cyclomatic)
                cognitive = cyclomatic * 1.1
                # Estimate nesting depth (heuristic - 2 base + complexity/5)
                nesting = 2 + int(cyclomatic / 5)

                return ComplexityMetrics(
                    cyclomatic_complexity=float(cyclomatic),
                    cognitive_complexity=float(cognitive),
                    nesting_depth=nesting,
                )

        # Default if not found
        return ComplexityMetrics(
            cyclomatic_complexity=1.0,
            cognitive_complexity=1.0,
            nesting_depth=1,
        )

    def _count_imports(self, content: str) -> int:
        """Count number of import statements."""
        count = 0
        for line in content.splitlines():
            stripped = line.strip()
            if stripped.startswith("import ") or stripped.startswith("from "):
                count += 1
        return count

    def _count_classes(self, content: str) -> int:
        """Count number of class definitions."""
        try:
            tree = ast.parse(content)
            return len([node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)])
        except SyntaxError:
            return 0

    def _calculate_maintainability_index(self, content: str) -> float:
        """
        Calculate maintainability index using radon.

        Returns value between 0-100, higher is more maintainable.
        """
        try:
            mi_score = mi_visit(content, multi=False)
            return float(mi_score) if mi_score else 0.0
        except Exception:
            return 0.0

    def _has_return_statement(self, func_node: ast.FunctionDef) -> bool:
        """Check if function has any return statement with a value."""
        for node in ast.walk(func_node):
            if isinstance(node, ast.Return) and node.value is not None:
                return True
        return False
