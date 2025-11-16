"""
Data models for codebase context and analysis.

Provides dataclasses for representing:
- Codebase context (dependency graph, patterns, history)
- Related files and their relationships
- Architectural patterns and conventions
- Risk areas and cascade impacts
- File history and churn metrics
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from enum import Enum


class RiskLevel(str, Enum):
    """Risk level for cascade impacts."""
    CRITICAL = "critical"  # Likely to break multiple things
    HIGH = "high"  # Likely to cause issues
    MEDIUM = "medium"  # Possible impacts
    LOW = "low"  # Minor risk


class PatternType(str, Enum):
    """Types of architectural patterns."""
    API_ENDPOINT = "api_endpoint"  # HTTP endpoint pattern
    DATABASE_ACCESS = "database_access"  # DB query pattern
    ERROR_HANDLING = "error_handling"  # Error handling pattern
    ASYNC_OPERATION = "async_operation"  # Async/await pattern
    TESTING = "testing"  # Test structure pattern
    CONFIGURATION = "configuration"  # Config pattern
    CACHING = "caching"  # Caching pattern
    LOGGING = "logging"  # Logging pattern
    VALIDATION = "validation"  # Input validation pattern
    AUTHENTICATION = "authentication"  # Auth pattern


@dataclass
class DependencyEdge:
    """Represents a dependency between files."""
    source_file: str  # File that depends on another
    target_file: str  # File being depended on
    import_statements: List[str] = field(default_factory=list)  # The actual imports
    coupling_strength: float = 0.5  # 0.0-1.0, higher = more tightly coupled


@dataclass
class RelatedFile:
    """A file related to a changed file."""
    file_path: str  # Path to related file
    relationship: str  # "imports", "imported_by", "same_directory", "similar_purpose"
    relevance_score: float = 0.5  # 0.0-1.0, higher = more relevant
    reason: Optional[str] = None  # Why it's related


@dataclass
class RiskArea:
    """An area at risk from cascade impacts."""
    file_path: str  # File at risk
    risk_level: RiskLevel  # How likely this will be affected
    affected_functions: List[str] = field(default_factory=list)  # Specific functions
    reason: str = ""  # Why this area is at risk
    impact_description: str = ""  # What might break


@dataclass
class ArchitecturalPattern:
    """A detected architectural pattern in the codebase."""
    pattern_type: PatternType  # Type of pattern
    name: str  # Human-readable name
    description: str  # What this pattern does
    example_files: List[str] = field(default_factory=list)  # Files using this pattern
    consistency_score: float = 0.8  # How consistently it's used (0.0-1.0)
    conventions: List[str] = field(default_factory=list)  # Conventions this pattern follows


@dataclass
class FileHistory:
    """Historical analysis of a file."""
    file_path: str  # Path to file
    change_frequency: int = 0  # How many times changed in history
    bug_count: int = 0  # Bugs that appeared in this file
    bug_types: List[str] = field(default_factory=list)  # Types of bugs
    last_changed: Optional[str] = None  # Last commit that changed it
    churn_rate: float = 0.0  # Lines changed per commit
    stability_score: float = 0.8  # 0.0-1.0, higher = more stable
    risk_score: float = 0.2  # 0.0-1.0, higher = more risky


@dataclass
class BugPattern:
    """A pattern of bugs that appeared in the codebase."""
    pattern_name: str  # e.g., "off-by-one", "resource-leak"
    occurrences: int = 0  # How many times it appeared
    affected_files: List[str] = field(default_factory=list)  # Where it appeared
    description: str = ""  # What the bug pattern is
    prevention_strategies: List[str] = field(default_factory=list)  # How to prevent


@dataclass
class CodebaseContext:
    """Complete context about a codebase for LLM analysis."""

    # Dependency information
    dependency_graph: Dict[str, List[str]] = field(default_factory=dict)
    """Dict mapping file path → list of files it imports"""

    reverse_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    """Dict mapping file path → list of files that import it"""

    circular_dependencies: List[Tuple[str, str]] = field(default_factory=list)
    """List of circular dependency pairs"""

    bottleneck_modules: List[Tuple[str, float]] = field(default_factory=list)
    """List of (file_path, coupling_score) for highly coupled modules"""

    # Pattern information
    architectural_patterns: List[ArchitecturalPattern] = field(default_factory=list)
    """Detected patterns in the codebase"""

    pattern_deviations: List[Tuple[str, str]] = field(default_factory=list)
    """List of (file_path, deviation_description) for code not following patterns"""

    # Historical information
    file_histories: Dict[str, FileHistory] = field(default_factory=dict)
    """Per-file historical analysis"""

    bug_patterns: List[BugPattern] = field(default_factory=list)
    """Common bug patterns found in the codebase"""

    high_risk_files: List[str] = field(default_factory=list)
    """Files with history of bugs"""

    # Repository metadata
    repository_url: str = ""
    """URL of the repository"""

    language: str = "python"
    """Primary programming language"""

    total_files: int = 0
    """Total number of files analyzed"""

    build_timestamp: Optional[str] = None
    """When this context was built"""


@dataclass
class CrossFileAnalysis:
    """Analysis of cross-file impacts and relationships."""

    changed_files: List[str]  # Files being changed
    related_files: List[RelatedFile]  # Potentially affected files
    cascade_risks: List[RiskArea]  # Cascade impact risks
    shared_dependencies: Dict[str, List[str]] = field(default_factory=dict)
    """Modules that multiple changed files depend on (shared deps)"""

    potentially_broken_tests: List[str] = field(default_factory=list)
    """Test files that might be affected"""

    api_contract_changes: List[str] = field(default_factory=list)
    """Changes to exported APIs or contracts"""

    database_schema_impacts: List[str] = field(default_factory=list)
    """Potential database schema impacts"""
