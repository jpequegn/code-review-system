"""
Anti-Pattern Detection

Detects common architectural anti-patterns like god objects, hub modules, etc.
"""

from dataclasses import dataclass
from typing import List

from src.architecture.dependency_graph import DependencyGraph


@dataclass
class AntiPattern:
    """Represents an architectural anti-pattern."""

    pattern_type: str
    module: str
    description: str
    severity: str  # "critical", "high", "medium", "low"
    remediation: str


class AntiPatternDetector:
    """Detects architectural anti-patterns in codebases."""

    def __init__(self, repo_path: str):
        """Initialize anti-pattern detector."""
        self.repo_path = repo_path
        self.graph = DependencyGraph(repo_path)
        self.graph.build_graph()

    def detect_all_antipatterns(self) -> List[AntiPattern]:
        """Detect all anti-patterns in the codebase."""
        antipatterns = []

        antipatterns.extend(self.detect_circular_dependencies())
        antipatterns.extend(self.detect_hub_pattern())
        antipatterns.extend(self.detect_god_objects())
        antipatterns.extend(self.detect_unstable_modules())

        return antipatterns

    def detect_circular_dependencies(self) -> List[AntiPattern]:
        """Detect circular dependencies (tight coupling)."""
        antipatterns = []
        cycles = self.graph.find_cycles()

        for cycle in cycles:
            modules_str = " ↔ ".join(cycle.modules)
            antipatterns.append(AntiPattern(
                pattern_type="circular_dependency",
                module=modules_str,
                description=f"Circular dependency: {modules_str}",
                severity="critical",
                remediation="Break the cycle by extracting shared code or refactoring dependencies"
            ))

        return antipatterns

    def detect_hub_pattern(self) -> List[AntiPattern]:
        """Detect hub pattern (one module everyone depends on)."""
        antipatterns = []
        bottlenecks = self.graph.find_bottlenecks(threshold=0.3)

        for bottleneck in bottlenecks:
            if bottleneck.afferent_coupling > 5:
                antipatterns.append(AntiPattern(
                    pattern_type="hub_pattern",
                    module=bottleneck.module,
                    description=f"{bottleneck.module} is a hub with {bottleneck.afferent_coupling} dependents",
                    severity="high",
                    remediation="Consider breaking up the module or using interfaces to reduce coupling"
                ))

        return antipatterns

    def detect_god_objects(self) -> List[AntiPattern]:
        """Detect god objects (modules doing too much)."""
        antipatterns = []

        # Heuristic: modules with high efferent coupling (many dependencies) are likely god objects
        for module in self.graph.graph.nodes():
            efferent = self.graph.graph.out_degree(module)

            # If a module depends on many others, it might be a god object
            if efferent > 10:
                antipatterns.append(AntiPattern(
                    pattern_type="god_object",
                    module=module,
                    description=f"{module} has {efferent} dependencies (possible god object)",
                    severity="medium",
                    remediation="Consider breaking up this module by extracting responsibilities"
                ))

        return antipatterns

    def detect_unstable_modules(self) -> List[AntiPattern]:
        """Detect unstable modules (high efferent coupling, no afferent)."""
        antipatterns = []

        for module in self.graph.graph.nodes():
            afferent = self.graph.graph.in_degree(module)
            efferent = self.graph.graph.out_degree(module)

            # Unstable: depends on many things but nothing depends on it
            if efferent > 5 and afferent == 0:
                instability = efferent / (afferent + efferent)
                antipatterns.append(AntiPattern(
                    pattern_type="unstable_module",
                    module=module,
                    description=f"{module} is unstable (instability: {instability:.2f})",
                    severity="medium",
                    remediation="Consider making this module more stable by reducing dependencies"
                ))

        return antipatterns
