"""
Metrics Data Models

Dataclasses for representing code metrics:
- ComplexityMetrics: Cyclomatic, cognitive complexity, nesting depth
- FunctionMetrics: Per-function metrics
- FileMetrics: Per-file metrics
- MetricsSnapshot: Complete metrics snapshot for a codebase state
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
from enum import Enum


class MetricType(Enum):
    """Types of metrics collected."""
    CYCLOMATIC_COMPLEXITY = "cyclomatic_complexity"
    COGNITIVE_COMPLEXITY = "cognitive_complexity"
    NESTING_DEPTH = "nesting_depth"
    LOC = "lines_of_code"
    PARAMETER_COUNT = "parameter_count"
    MAINTAINABILITY_INDEX = "maintainability_index"


@dataclass
class ComplexityMetrics:
    """
    Complexity metrics for a function or code block.

    Attributes:
        cyclomatic_complexity: Cyclomatic complexity (CC) - counts linear independent paths
        cognitive_complexity: Cognitive complexity - measures how hard to understand
        nesting_depth: Maximum nesting depth (if/for/while nesting levels)
    """
    cyclomatic_complexity: float
    cognitive_complexity: float
    nesting_depth: int

    def is_high_complexity(self, cc_threshold: float = 10) -> bool:
        """Check if cyclomatic complexity exceeds threshold."""
        return self.cyclomatic_complexity > cc_threshold

    def is_high_cognitive(self, cognitive_threshold: float = 15) -> bool:
        """Check if cognitive complexity exceeds threshold."""
        return self.cognitive_complexity > cognitive_threshold

    def is_deeply_nested(self, nesting_threshold: int = 4) -> bool:
        """Check if nesting depth exceeds threshold."""
        return self.nesting_depth > nesting_threshold


@dataclass
class FunctionMetrics:
    """
    Metrics for a single function.

    Attributes:
        name: Function name
        line_number: Starting line number in file
        length: Number of lines of code
        parameter_count: Number of parameters
        complexity: ComplexityMetrics for this function
        has_docstring: Whether function has a docstring
        returns_value: Whether function returns a value
    """
    name: str
    line_number: int
    length: int
    parameter_count: int
    complexity: ComplexityMetrics
    has_docstring: bool = False
    returns_value: bool = False

    def is_long_function(self, length_threshold: int = 50) -> bool:
        """Check if function exceeds line count threshold."""
        return self.length > length_threshold

    def has_many_parameters(self, param_threshold: int = 5) -> bool:
        """Check if function has too many parameters."""
        return self.parameter_count > param_threshold


@dataclass
class FileMetrics:
    """
    Metrics for a single file.

    Attributes:
        file_path: Relative path to file
        language: Programming language
        total_lines: Total lines including blank/comments
        code_lines: Lines of actual code (excluding blank/comments)
        functions: List of FunctionMetrics for each function
        average_function_length: Average lines per function
        max_complexity: Maximum cyclomatic complexity in file
        average_complexity: Average cyclomatic complexity
        imports_count: Number of imports
        classes_count: Number of classes defined
        churn_rate: How often this file changes (commits in last N days)
    """
    file_path: str
    language: str
    total_lines: int
    code_lines: int
    functions: List[FunctionMetrics] = field(default_factory=list)
    average_function_length: float = 0.0
    max_complexity: float = 0.0
    average_complexity: float = 0.0
    imports_count: int = 0
    classes_count: int = 0
    churn_rate: float = 0.0  # Commits per month

    def is_high_churn(self, churn_threshold: float = 2.0) -> bool:
        """Check if file changes too frequently."""
        return self.churn_rate > churn_threshold

    def is_overly_complex(self, cc_threshold: float = 10) -> bool:
        """Check if average complexity exceeds threshold."""
        return self.average_complexity > cc_threshold

    def has_large_functions(self, length_threshold: int = 50) -> int:
        """Count functions exceeding line count threshold."""
        return sum(1 for f in self.functions if f.is_long_function(length_threshold))

    def has_complex_functions(self, cc_threshold: float = 10) -> int:
        """Count functions exceeding complexity threshold."""
        return sum(1 for f in self.functions if f.complexity.is_high_complexity(cc_threshold))


@dataclass
class MetricsSnapshot:
    """
    Complete metrics snapshot for a codebase at a specific point in time.

    Attributes:
        commit_sha: Git commit SHA for this snapshot
        timestamp: When snapshot was created
        files: Dictionary of file_path → FileMetrics
        total_lines: Total lines of code
        average_file_complexity: Average complexity across all files
        max_file_complexity: Highest complexity file
        high_complexity_files: Files above complexity threshold
    """
    commit_sha: str
    timestamp: str
    files: Dict[str, FileMetrics] = field(default_factory=dict)
    total_lines: int = 0
    average_file_complexity: float = 0.0
    max_file_complexity: float = 0.0
    high_complexity_files: List[str] = field(default_factory=list)

    def calculate_summary_metrics(self) -> None:
        """Calculate summary metrics from individual file metrics."""
        if not self.files:
            return

        self.total_lines = sum(f.total_lines for f in self.files.values())
        complexities = [f.average_complexity for f in self.files.values()]
        if complexities:
            self.average_file_complexity = sum(complexities) / len(complexities)
            self.max_file_complexity = max(complexities)

        # Find high complexity files
        threshold = self.average_file_complexity * 1.5  # 50% above average
        self.high_complexity_files = [
            path for path, metrics in self.files.items()
            if metrics.average_complexity > threshold
        ]

    def get_complexity_trend(self, previous: Optional["MetricsSnapshot"]) -> Optional[float]:
        """
        Calculate complexity trend compared to previous snapshot.

        Returns:
            Percentage change in average complexity, or None if no previous snapshot
        """
        if not previous:
            return None

        if previous.average_file_complexity == 0:
            return None

        change = (
            (self.average_file_complexity - previous.average_file_complexity) /
            previous.average_file_complexity * 100
        )
        return change
