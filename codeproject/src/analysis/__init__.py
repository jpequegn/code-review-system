"""
Analysis Package

Provides code analysis utilities including diff parsing and code quality assessment.
"""

from src.analysis.diff_parser import (
    DiffParser,
    CodeChange,
    FileDiff,
)

__all__ = [
    "DiffParser",
    "CodeChange",
    "FileDiff",
]
