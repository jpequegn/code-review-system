"""
Cascade Analysis

Analyzes the impact of changes through the dependency graph.
Shows what breaks when a module changes.
"""

from dataclasses import dataclass
from typing import List

from src.architecture.dependency_graph import DependencyGraph, CascadeImpact


@dataclass
class ImpactTree:
    """Represents the tree of impacts from a change."""

    changed_module: str
    direct_impacts: List[str]  # Modules directly affected
    indirect_impacts: List[str]  # Modules indirectly affected
    total_impact_count: int  # Total affected modules
    risk_score: float  # 0-1 risk level


class CascadeAnalyzer:
    """Analyzes cascade impacts of changes."""

    def __init__(self, repo_path: str):
        """Initialize cascade analyzer."""
        self.repo_path = repo_path
        self.graph = DependencyGraph(repo_path)
        self.graph.build_graph()

    def show_impact_tree(self, changed_file: str) -> ImpactTree:
        """
        Show the full impact tree of a change.

        Args:
            changed_file: File that changed

        Returns:
            ImpactTree with direct and indirect impacts
        """
        cascade = self.graph.get_cascade_impact(changed_file)

        # Separate direct and indirect impacts
        direct = set(self.graph.graph.predecessors(changed_file))
        indirect = set(cascade.affected_modules) - direct

        return ImpactTree(
            changed_module=changed_file,
            direct_impacts=sorted(list(direct)),
            indirect_impacts=sorted(list(indirect)),
            total_impact_count=cascade.impact_depth,
            risk_score=cascade.risk_score
        )

    def estimate_risk(self, changed_module: str) -> str:
        """
        Estimate the risk level of a change.

        Args:
            changed_module: Module that changed

        Returns:
            Risk level: "critical", "high", "medium", "low"
        """
        cascade = self.graph.get_cascade_impact(changed_module)

        risk_score = cascade.risk_score
        if risk_score > 0.7:
            return "critical"
        elif risk_score > 0.5:
            return "high"
        elif risk_score > 0.3:
            return "medium"
        else:
            return "low"

    def show_related_failures(self, failure_module: str) -> List[str]:
        """
        Show likely related failures if a module fails.

        Args:
            failure_module: Module that failed

        Returns:
            List of modules likely to fail as a result
        """
        # Find all modules that depend on the failed module
        dependents = list(self.graph.graph.predecessors(failure_module))

        # Filter to only critical dependents (those with few alternatives)
        critical_dependents = []
        for dependent in dependents:
            # If dependent has only one source of this dependency, it's critical
            dependencies = self.graph.get_module_dependencies(dependent)
            if len(dependencies["imports"]) == 1:
                critical_dependents.append(dependent)

        return sorted(critical_dependents)

    def analyze_change_impact(self, changed_modules: List[str]) -> dict:
        """
        Analyze the impact of changing multiple modules.

        Args:
            changed_modules: List of modules that changed

        Returns:
            Dict with comprehensive impact analysis
        """
        all_affected = set()
        max_risk = 0.0

        for module in changed_modules:
            cascade = self.graph.get_cascade_impact(module)
            all_affected.update(cascade.affected_modules)
            max_risk = max(max_risk, cascade.risk_score)

        return {
            "changed_modules": changed_modules,
            "affected_count": len(all_affected),
            "affected_modules": sorted(list(all_affected)),
            "risk_level": self._risk_score_to_level(max_risk),
            "risk_score": round(max_risk, 2),
            "recommendations": self._get_recommendations(changed_modules, max_risk)
        }

    def _risk_score_to_level(self, risk_score: float) -> str:
        """Convert risk score to level."""
        if risk_score > 0.7:
            return "critical"
        elif risk_score > 0.5:
            return "high"
        elif risk_score > 0.3:
            return "medium"
        else:
            return "low"

    def _get_recommendations(self, changed_modules: list, risk_score: float) -> List[str]:
        """Generate recommendations based on risk."""
        recommendations = []

        if risk_score > 0.7:
            recommendations.append("High risk change - Extensive testing recommended")
            recommendations.append("Consider splitting the change into smaller, safer changes")

        if len(changed_modules) > 5:
            recommendations.append("Multiple modules changed - Verify compatibility")

        # Check for circular dependencies in changed modules
        for module in changed_modules:
            if self.graph.find_cycles():
                recommendations.append(f"Circular dependencies exist - be cautious with {module}")
                break

        if not recommendations:
            recommendations.append("Change appears low risk - Standard testing should suffice")

        return recommendations
