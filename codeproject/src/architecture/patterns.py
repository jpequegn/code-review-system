"""
Architectural Pattern Detection

Identifies architectural patterns and detects violations.
Recognizes common patterns like layered, modular, MVC, etc.
"""

from enum import Enum
from typing import List, Dict, Set
from dataclasses import dataclass
from pathlib import Path

from src.architecture.dependency_graph import DependencyGraph


class ArchitecturePattern(str, Enum):
    """Common architectural patterns."""

    LAYERED = "layered"  # Presentation → Business → Data
    MODULAR = "modular"  # Independent modules with clear boundaries
    MICROSERVICES = "microservices"  # Separate deployable units
    MVC = "mvc"  # Model-View-Controller
    PLUGIN = "plugin"  # Core + plugins
    UNKNOWN = "unknown"  # Pattern not clearly identified


@dataclass
class PatternViolation:
    """Represents a violation of the detected pattern."""

    violation_type: str
    module: str
    description: str
    severity: str  # "critical", "high", "medium", "low"


class PatternDetector:
    """
    Detects architectural patterns in codebases.

    Identifies whether code follows layered, modular, MVC, or other patterns.
    Also detects violations of those patterns.
    """

    def __init__(self, repo_path: str):
        """
        Initialize pattern detector.

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)
        self.graph = DependencyGraph(repo_path)
        self.graph.build_graph()

    def detect_pattern(self) -> ArchitecturePattern:
        """
        Detect the architectural pattern used in the codebase.

        Returns:
            Detected ArchitecturePattern
        """
        # Heuristic-based detection using directory structure and imports
        repo_path = self.repo_path

        # Check for common directory patterns
        has_layers = self._detect_layers()
        has_mvc = self._detect_mvc()
        has_plugins = self._detect_plugins()
        is_modular = self._detect_modularity()

        # Return the best match
        if has_layers and not has_mvc:
            return ArchitecturePattern.LAYERED
        elif has_mvc:
            return ArchitecturePattern.MVC
        elif has_plugins:
            return ArchitecturePattern.PLUGIN
        elif is_modular:
            return ArchitecturePattern.MODULAR
        else:
            return ArchitecturePattern.UNKNOWN

    def _detect_layers(self) -> bool:
        """Detect if using layered architecture."""
        layer_indicators = ["presentation", "business", "data", "service", "repository"]
        repo_dirs = set(d.name for d in self.repo_path.iterdir() if d.is_dir())

        return len(repo_dirs & set(layer_indicators)) >= 2

    def _detect_mvc(self) -> bool:
        """Detect if using MVC architecture."""
        mvc_indicators = ["models", "views", "controllers", "model", "view", "controller"]
        repo_dirs = set(d.name for d in self.repo_path.iterdir() if d.is_dir())

        return len(repo_dirs & set(mvc_indicators)) >= 2

    def _detect_plugins(self) -> bool:
        """Detect if using plugin architecture."""
        plugin_indicators = ["plugins", "core", "extensions", "plugin", "extension"]
        repo_dirs = set(d.name for d in self.repo_path.iterdir() if d.is_dir())

        return len(repo_dirs & set(plugin_indicators)) >= 2

    def _detect_modularity(self) -> bool:
        """Detect if code is well-modularized."""
        # Good modularity = many modules with low coupling
        cycles = self.graph.find_cycles()
        bottlenecks = self.graph.find_bottlenecks()
        metrics = self.graph.get_graph_metrics()

        # No cycles and few bottlenecks = good modularity
        return len(cycles) == 0 and len(bottlenecks) < 3

    def detect_violations(self) -> List[PatternViolation]:
        """
        Detect violations of the identified pattern.

        Returns:
            List of pattern violations
        """
        detected_pattern = self.detect_pattern()
        violations = []

        # Detect circular dependencies (violation of most patterns)
        cycles = self.graph.find_cycles()
        for cycle in cycles:
            violations.append(PatternViolation(
                violation_type="circular_dependency",
                module=" → ".join(cycle.modules),
                description=f"Circular dependency detected: {' → '.join(cycle.modules)}",
                severity="critical"
            ))

        # Detect high-coupling bottlenecks
        bottlenecks = self.graph.find_bottlenecks(threshold=0.4)
        for bottleneck in bottlenecks:
            violations.append(PatternViolation(
                violation_type="bottleneck",
                module=bottleneck.module,
                description=f"Module is a bottleneck with coupling score {bottleneck.coupling_score}",
                severity="high"
            ))

        # Pattern-specific violations
        if detected_pattern == ArchitecturePattern.LAYERED:
            violations.extend(self._detect_layer_violations())
        elif detected_pattern == ArchitecturePattern.MVC:
            violations.extend(self._detect_mvc_violations())

        return violations

    def _detect_layer_violations(self) -> List[PatternViolation]:
        """Detect violations of layered architecture."""
        violations = []

        # In layered architecture, lower layers shouldn't depend on upper layers
        # This would be a skip-level dependency violation
        for module in self.graph.graph.nodes():
            dependencies = self.graph.graph.successors(module)
            for dep in dependencies:
                # Simple heuristic: check if a data layer imports from business
                if "data" in module and "business" in dep:
                    violations.append(PatternViolation(
                        violation_type="skip_layer",
                        module=module,
                        description=f"{module} depends on {dep} (skip-layer dependency)",
                        severity="medium"
                    ))

        return violations

    def _detect_mvc_violations(self) -> List[PatternViolation]:
        """Detect violations of MVC architecture."""
        violations = []

        # Models shouldn't depend on views/controllers
        for module in self.graph.graph.nodes():
            if "model" in module:
                dependencies = self.graph.graph.successors(module)
                for dep in dependencies:
                    if "view" in dep or "controller" in dep:
                        violations.append(PatternViolation(
                            violation_type="mvc_violation",
                            module=module,
                            description=f"Model {module} shouldn't depend on {dep}",
                            severity="high"
                        ))

        return violations

    def suggest_pattern_fit(self) -> ArchitecturePattern:
        """
        Suggest the best architectural pattern for the codebase.

        Based on structure and metrics, recommend which pattern would fit best.

        Returns:
            Recommended ArchitecturePattern
        """
        # If code has no cycles and good modularity, recommend modular
        if self.graph.is_acyclic() and len(self.graph.find_bottlenecks(0.3)) < 2:
            return ArchitecturePattern.MODULAR

        # If code has clear layers, recommend layered
        if self._detect_layers():
            return ArchitecturePattern.LAYERED

        # If code has MVC structure, recommend MVC
        if self._detect_mvc():
            return ArchitecturePattern.MVC

        # Default to modular
        return ArchitecturePattern.MODULAR
