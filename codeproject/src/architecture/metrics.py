"""
Architectural Metrics

Calculates coupling, cohesion, and other architectural quality metrics.
"""

from dataclasses import dataclass
from typing import Dict, Optional

from src.architecture.dependency_graph import DependencyGraph


@dataclass
class CouplingMetrics:
    """Coupling metrics for a module."""

    module: str
    afferent: int  # Modules that depend on this
    efferent: int  # Modules this depends on
    instability: float  # Ce / (Ca + Ce), 0=stable, 1=unstable


@dataclass
class CohesionMetrics:
    """Cohesion metrics for a module."""

    module: str
    internal_relationships: int  # Functions/classes using each other
    external_references: int  # Dependencies on other modules
    cohesion_score: float  # 0-1, higher is better


class ArchitectureMetrics:
    """Calculate architectural quality metrics."""

    def __init__(self, repo_path: str):
        """Initialize metrics calculator."""
        self.repo_path = repo_path
        self.graph = DependencyGraph(repo_path)
        self.graph.build_graph()

    def calculate_coupling_metrics(self, module: str) -> Optional[CouplingMetrics]:
        """Calculate coupling metrics for a module."""
        if module not in self.graph.graph.nodes():
            return None

        afferent = self.graph.graph.in_degree(module)
        efferent = self.graph.graph.out_degree(module)

        # Instability = Ce / (Ca + Ce)
        # 0 = stable (many depend on it), 1 = unstable (depends on many)
        total = afferent + efferent
        instability = efferent / total if total > 0 else 0.0

        return CouplingMetrics(
            module=module,
            afferent=afferent,
            efferent=efferent,
            instability=round(instability, 2)
        )

    def calculate_cohesion_metrics(self, module: str) -> CohesionMetrics:
        """Calculate cohesion metrics for a module."""
        # Simplified: use internal vs. external dependencies
        dependencies = self.graph.get_module_dependencies(module)

        internal = len(dependencies["imports"])
        external = len(dependencies["imported_by"])
        total = internal + external

        cohesion = internal / total if total > 0 else 0.5

        return CohesionMetrics(
            module=module,
            internal_relationships=internal,
            external_references=external,
            cohesion_score=round(cohesion, 2)
        )

    def calculate_modularity_score(self) -> float:
        """Calculate overall modularity score (0-1)."""
        metrics = self.graph.get_graph_metrics()

        if metrics["node_count"] == 0:
            return 0.5

        # Components: low cycles (good), low density (good), balanced coupling
        cycles = metrics["cycle_count"]
        density = metrics["density"]

        # Normalize to 0-1
        cycle_penalty = min(1.0, cycles / metrics["node_count"])
        density_penalty = density  # Already 0-1

        modularity = 1.0 - ((cycle_penalty + density_penalty) / 2)
        return round(modularity, 2)

    def get_all_metrics(self) -> Dict:
        """Get comprehensive metrics for the entire architecture."""
        metrics = self.graph.get_graph_metrics()
        modularity = self.calculate_modularity_score()
        cycles = self.graph.find_cycles()
        bottlenecks = self.graph.find_bottlenecks()

        return {
            "node_count": metrics["node_count"],
            "edge_count": metrics["edge_count"],
            "density": round(metrics["density"], 2),
            "cycle_count": metrics["cycle_count"],
            "cycles": [{"modules": c.modules} for c in cycles[:5]],
            "modularity_score": modularity,
            "bottleneck_count": len(bottlenecks),
            "bottlenecks": [
                {
                    "module": b.module,
                    "coupling": b.coupling_score,
                    "afferent": b.afferent_coupling,
                    "efferent": b.efferent_coupling,
                }
                for b in bottlenecks[:5]
            ],
        }
