"""
Architectural Suggestions

Generates suggestions for improving architecture.
Identifies refactoring opportunities and optimization points.
"""

from dataclasses import dataclass
from typing import List

from src.architecture.dependency_graph import DependencyGraph
from src.architecture.metrics import ArchitectureMetrics
from src.architecture.antipatterns import AntiPatternDetector


@dataclass
class Suggestion:
    """Represents an architectural improvement suggestion."""

    suggestion_type: str  # "refactoring", "modularization", "decoupling", etc.
    module: str
    priority: str  # "high", "medium", "low"
    description: str
    estimated_effort: str  # "low", "medium", "high"
    benefit: str


class SuggestionEngine:
    """Generates architectural improvement suggestions."""

    def __init__(self, repo_path: str):
        """Initialize suggestion engine."""
        self.repo_path = repo_path
        self.graph = DependencyGraph(repo_path)
        self.graph.build_graph()
        self.metrics = ArchitectureMetrics(repo_path)
        self.antipattern_detector = AntiPatternDetector(repo_path)

    def suggest_refactorings(self) -> List[Suggestion]:
        """Suggest refactorings for the codebase."""
        suggestions = []

        # Get anti-patterns and convert to refactoring suggestions
        antipatterns = self.antipattern_detector.detect_all_antipatterns()
        for ap in antipatterns:
            suggestions.append(Suggestion(
                suggestion_type="refactoring",
                module=ap.module,
                priority="high" if ap.severity == "critical" else "medium",
                description=ap.description,
                estimated_effort="high" if ap.severity == "critical" else "medium",
                benefit=ap.remediation
            ))

        # Suggest decoupling for high-coupling modules
        bottlenecks = self.graph.find_bottlenecks(0.3)
        for bottleneck in bottlenecks[:5]:
            suggestions.append(Suggestion(
                suggestion_type="decoupling",
                module=bottleneck.module,
                priority="medium",
                description=f"Decouple {bottleneck.module} (coupling: {bottleneck.coupling_score})",
                estimated_effort="medium",
                benefit="Reduce module coupling and improve modularity"
            ))

        return suggestions

    def suggest_modularization_points(self) -> List[Suggestion]:
        """Suggest where to split modules or extract responsibilities."""
        suggestions = []

        # Find god objects and suggest splitting them
        for module in self.graph.graph.nodes():
            efferent = self.graph.graph.out_degree(module)

            # If module has many dependencies, suggest extracting
            if efferent > 10:
                suggestions.append(Suggestion(
                    suggestion_type="modularization",
                    module=module,
                    priority="medium",
                    description=f"Extract responsibilities from {module} ({efferent} dependencies)",
                    estimated_effort="medium",
                    benefit="Reduce complexity and improve maintainability"
                ))

        # Find isolated modules that could be reused
        for module in self.graph.graph.nodes():
            afferent = self.graph.graph.in_degree(module)
            efferent = self.graph.graph.out_degree(module)

            # Stable module with few dependencies = good candidate for reuse
            if afferent == 0 and efferent < 3:
                suggestions.append(Suggestion(
                    suggestion_type="modularization",
                    module=module,
                    priority="low",
                    description=f"Consider making {module} reusable (stable, independent)",
                    estimated_effort="low",
                    benefit="Increase code reuse and reduce duplication"
                ))

        return suggestions

    def suggest_dependency_reduction(self, module: str) -> List[Suggestion]:
        """Suggest how to reduce dependencies for a specific module."""
        suggestions = []

        # Get current dependencies
        deps = self.graph.get_module_dependencies(module)
        imports = deps["imports"]

        if not imports:
            return suggestions

        # Suggest removing unnecessary dependencies
        for dependency in imports:
            # Check if dependency is also imported by many others
            dependent_count = self.graph.graph.in_degree(dependency)

            if dependent_count > 10:
                suggestions.append(Suggestion(
                    suggestion_type="dependency_reduction",
                    module=module,
                    priority="low",
                    description=f"Consider abstracting dependency on {dependency}",
                    estimated_effort="medium",
                    benefit="Reduce tight coupling to popular module"
                ))

        # Suggest consolidating many dependencies
        if len(imports) > 10:
            suggestions.append(Suggestion(
                suggestion_type="dependency_reduction",
                module=module,
                priority="medium",
                description=f"Consolidate {len(imports)} dependencies using facades or interfaces",
                estimated_effort="high",
                benefit="Simplify module dependencies and improve clarity"
            ))

        return suggestions

    def get_all_suggestions(self) -> dict:
        """Get all architectural suggestions."""
        return {
            "refactorings": self.suggest_refactorings(),
            "modularization": self.suggest_modularization_points(),
            "summary": {
                "total_suggestions": len(self.suggest_refactorings()) + len(self.suggest_modularization_points()),
                "high_priority": len([s for s in self.suggest_refactorings() if s.priority == "high"]),
                "medium_priority": len([s for s in self.suggest_refactorings() if s.priority == "medium"]),
            }
        }
