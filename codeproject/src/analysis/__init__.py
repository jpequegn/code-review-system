"""
Analysis Package

Provides code analysis utilities including diff parsing and code quality assessment.
"""

from src.analysis.diff_parser import (
    DiffParser,
    CodeChange,
    FileDiff,
)
from src.analysis.analyzer import (
    CodeAnalyzer,
    AnalyzedFinding,
)

__all__ = [
    "DiffParser",
    "CodeChange",
    "FileDiff",
    "CodeAnalyzer",
    "AnalyzedFinding",
]
