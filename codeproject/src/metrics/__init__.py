"""
Metrics collection module.

Provides metrics detection and analysis for code quality assessment.
"""

from src.metrics.collector import (
    MetricsCollector,
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

__all__ = [
    "MetricsCollector",
    "PythonMetricsCollector",
    "MetricsCollectorError",
    "ComplexityMetrics",
    "FunctionMetrics",
    "FileMetrics",
    "MetricsSnapshot",
    "MetricType",
]
