"""
Dependency Graph Engine

Analyzes and maintains dependency graphs for architectural understanding.
Detects cycles, calculates coupling metrics, and analyzes cascade impacts.
"""

import ast
from pathlib import Path
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass

try:
    import networkx as nx
except ImportError:
    nx = None


@dataclass
class Cycle:
    """Represents a circular dependency."""
    modules: List[str]  # Modules involved in the cycle
    length: int  # Length of the cycle

    def __repr__(self) -> str:
        return f"Cycle({' → '.join(self.modules)} → {self.modules[0]})"


@dataclass
class Bottleneck:
    """Represents a module that's a bottleneck (high coupling)."""
    module: str
    afferent_coupling: int  # How many depend on this
    efferent_coupling: int  # How many this depends on
    coupling_score: float  # 0-1 score


@dataclass
class CascadeImpact:
    """Represents the impact of changing a module."""
    changed_module: str
    affected_modules: List[str]  # Modules affected by the change
    impact_depth: int  # How many levels deep
    risk_score: float  # 0-1 risk score


class DependencyGraph:
    """
    Builds and analyzes dependency graphs for Python codebases.

    Uses networkx to detect cycles, calculate coupling metrics,
    and analyze cascade impacts.
    """

    def __init__(self, repo_path: str):
        """
        Initialize dependency graph builder.

        Args:
            repo_path: Path to repository root
        """
        if nx is None:
            raise ImportError("networkx is required for dependency graph analysis")

        self.repo_path = Path(repo_path)
        self.graph = nx.DiGraph()
        self.module_map: Dict[str, Path] = {}  # Map module names to file paths
        self.import_statements: Dict[str, Set[str]] = {}  # module -> set of imports

    def build_graph(self) -> 'DependencyGraph':
        """
        Build the dependency graph from Python imports.

        Scans the repository for .py files and extracts import statements
        to build a directed graph of dependencies.

        Returns:
            Self for chaining
        """
        python_files = list(self.repo_path.rglob("*.py"))

        # Build initial module map and extract imports
        for py_file in python_files:
            relative_path = py_file.relative_to(self.repo_path)
            module_name = str(relative_path.with_suffix("")).replace("/", ".")

            self.module_map[module_name] = py_file
            self.import_statements[module_name] = set()

            # Extract imports
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
                    self._extract_imports(tree, module_name)
            except (SyntaxError, UnicodeDecodeError):
                # Skip files with syntax errors
                pass

        # Build the graph
        for module, imports in self.import_statements.items():
            if module not in self.graph:
                self.graph.add_node(module)

            for imported_module in imports:
                if imported_module not in self.graph:
                    self.graph.add_node(imported_module)
                self.graph.add_edge(module, imported_module)

        return self

    def _extract_imports(self, tree: ast.AST, module_name: str) -> None:
        """Extract imports from AST."""
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.import_statements[module_name].add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    self.import_statements[module_name].add(node.module)

    def find_cycles(self) -> List[Cycle]:
        """
        Find circular dependencies in the graph.

        Returns:
            List of cycles detected
        """
        cycles = []
        try:
            for cycle in nx.simple_cycles(self.graph):
                cycles.append(Cycle(
                    modules=cycle,
                    length=len(cycle)
                ))
        except Exception:
            # Handle edge cases in cycle detection
            pass

        return sorted(cycles, key=lambda c: c.length)

    def calculate_coupling(self, module_a: str, module_b: str) -> float:
        """
        Calculate coupling strength between two modules.

        Args:
            module_a: First module name
            module_b: Second module name

        Returns:
            Coupling strength (0-1)
        """
        if module_a not in self.graph or module_b not in self.graph:
            return 0.0

        # Check if there's a path between them
        try:
            path_length = nx.shortest_path_length(self.graph, module_a, module_b)
            # Closer modules are more tightly coupled
            return 1.0 / (1.0 + path_length)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return 0.0

    def get_cascade_impact(self, changed_module: str) -> CascadeImpact:
        """
        Analyze the cascade impact of changing a module.

        Shows all modules affected by a change to the given module.

        Args:
            changed_module: Module that changed

        Returns:
            CascadeImpact with affected modules and risk score
        """
        if changed_module not in self.graph:
            return CascadeImpact(changed_module, [], 0, 0.0)

        # Find all modules that depend on the changed module
        affected = set()
        visited = set()

        def traverse_dependents(module: str, depth: int) -> None:
            if module in visited or depth > 5:  # Limit traversal depth
                return
            visited.add(module)

            # Find all modules that depend on this one
            for predecessor in self.graph.predecessors(module):
                affected.add(predecessor)
                traverse_dependents(predecessor, depth + 1)

        traverse_dependents(changed_module, 0)

        # Calculate risk score based on number of affected modules
        total_modules = len(self.graph)
        risk_score = min(1.0, len(affected) / max(1, total_modules))

        return CascadeImpact(
            changed_module=changed_module,
            affected_modules=sorted(list(affected)),
            impact_depth=len(affected),
            risk_score=risk_score
        )

    def find_bottlenecks(self, threshold: float = 0.3) -> List[Bottleneck]:
        """
        Find bottleneck modules (high coupling hub pattern).

        Args:
            threshold: Minimum coupling score to be considered a bottleneck

        Returns:
            List of bottleneck modules sorted by coupling
        """
        bottlenecks = []

        for module in self.graph.nodes():
            # Calculate in-degree (afferent coupling) and out-degree (efferent coupling)
            in_degree = self.graph.in_degree(module)
            out_degree = self.graph.out_degree(module)

            total_nodes = len(self.graph)
            if total_nodes > 0:
                afferent = in_degree / total_nodes
                efferent = out_degree / total_nodes
                coupling_score = (afferent + efferent) / 2

                if coupling_score >= threshold:
                    bottlenecks.append(Bottleneck(
                        module=module,
                        afferent_coupling=in_degree,
                        efferent_coupling=out_degree,
                        coupling_score=round(coupling_score, 2)
                    ))

        return sorted(bottlenecks, key=lambda b: b.coupling_score, reverse=True)

    def get_module_dependencies(self, module: str) -> Dict[str, List[str]]:
        """
        Get incoming and outgoing dependencies for a module.

        Args:
            module: Module name

        Returns:
            Dict with 'imports' and 'imported_by' lists
        """
        if module not in self.graph:
            return {"imports": [], "imported_by": []}

        return {
            "imports": list(self.graph.successors(module)),
            "imported_by": list(self.graph.predecessors(module)),
        }

    def get_graph_metrics(self) -> Dict[str, float]:
        """
        Calculate overall graph metrics.

        Returns:
            Dict with modularity, density, and other metrics
        """
        if len(self.graph) == 0:
            return {
                "node_count": 0,
                "edge_count": 0,
                "density": 0.0,
                "cycle_count": 0,
            }

        return {
            "node_count": self.graph.number_of_nodes(),
            "edge_count": self.graph.number_of_edges(),
            "density": nx.density(self.graph),
            "cycle_count": len(list(nx.simple_cycles(self.graph))),
        }

    def find_connected_components(self) -> List[Set[str]]:
        """
        Find connected components (groups of interconnected modules).

        Returns:
            List of component sets
        """
        # Convert to undirected for component analysis
        undirected = self.graph.to_undirected()
        components = list(nx.connected_components(undirected))
        return components

    def get_shortest_path(self, source: str, target: str) -> Optional[List[str]]:
        """
        Find the shortest dependency path between two modules.

        Args:
            source: Source module
            target: Target module

        Returns:
            List of modules in the path, or None if no path exists
        """
        try:
            return nx.shortest_path(self.graph, source, target)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return None

    def is_acyclic(self) -> bool:
        """
        Check if the dependency graph is acyclic.

        Returns:
            True if the graph has no cycles
        """
        return nx.is_directed_acyclic_graph(self.graph)

    def get_topological_order(self) -> Optional[List[str]]:
        """
        Get topological order of modules (if acyclic).

        Returns:
            Topologically sorted module list, or None if cycles exist
        """
        if not self.is_acyclic():
            return None

        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError:
            return None
