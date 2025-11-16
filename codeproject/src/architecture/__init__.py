"""Architectural intelligence and analysis module."""

from src.architecture.dependency_graph import DependencyGraph
from src.architecture.patterns import PatternDetector, ArchitecturePattern
from src.architecture.metrics import CouplingMetrics, CohesionMetrics
from src.architecture.antipatterns import AntiPatternDetector
from src.architecture.cascade_analyzer import CascadeAnalyzer
from src.architecture.suggestions import SuggestionEngine

__all__ = [
    "DependencyGraph",
    "PatternDetector",
    "ArchitecturePattern",
    "CouplingMetrics",
    "CohesionMetrics",
    "AntiPatternDetector",
    "CascadeAnalyzer",
    "SuggestionEngine",
]
