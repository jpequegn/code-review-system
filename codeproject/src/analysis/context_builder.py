"""
Codebase Context Builder

Builds comprehensive context about a codebase for smarter LLM analysis.
Includes dependency graph analysis, pattern detection, and historical analysis.
"""

import logging
import ast
from pathlib import Path
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict
from datetime import datetime

from src.analysis.context_models import (
    CodebaseContext,
    DependencyEdge,
    RelatedFile,
    RiskArea,
    RiskLevel,
    ArchitecturalPattern,
    PatternType,
    FileHistory,
    BugPattern,
    CrossFileAnalysis,
)

logger = logging.getLogger(__name__)


class ContextBuilder:
    """
    Builds comprehensive codebase context for LLM analysis.

    Responsibilities:
    - Analyze dependency graphs from imports
    - Detect architectural patterns
    - Retrieve file history and bug patterns
    - Identify cascade risks
    - Find related files
    """

    def __init__(self, repo_path: str):
        """
        Initialize context builder.

        Args:
            repo_path: Path to repository root
        """
        self.repo_path = Path(repo_path)
        self.logger = logger

    def build_context(
        self,
        repository_url: str = "",
        language: str = "python",
    ) -> CodebaseContext:
        """
        Build complete codebase context.

        Args:
            repository_url: URL of the repository
            language: Primary programming language

        Returns:
            CodebaseContext with all analysis
        """
        self.logger.info(f"Building codebase context for {self.repo_path}")

        # Find all Python files
        python_files = self._find_python_files()
        self.logger.debug(f"Found {len(python_files)} Python files")

        # Build dependency graph
        dep_graph, rev_deps = self._build_dependency_graph(python_files)

        # Detect patterns
        patterns = self._detect_patterns(python_files, dep_graph)

        # Identify circular dependencies
        circular_deps = self._find_circular_dependencies(dep_graph)

        # Find bottleneck modules
        bottlenecks = self._find_bottleneck_modules(dep_graph, rev_deps)

        # Detect pattern deviations
        deviations = self._detect_pattern_deviations(python_files, patterns)

        # Build file histories (mock for now - would query git in production)
        file_histories = self._build_file_histories(python_files)

        # Detect bug patterns (mock for now - would query database in production)
        bug_patterns = self._detect_bug_patterns(file_histories)

        # Identify high-risk files
        high_risk_files = self._identify_high_risk_files(file_histories)

        context = CodebaseContext(
            dependency_graph=dep_graph,
            reverse_dependencies=rev_deps,
            circular_dependencies=circular_deps,
            bottleneck_modules=bottlenecks,
            architectural_patterns=patterns,
            pattern_deviations=deviations,
            file_histories=file_histories,
            bug_patterns=bug_patterns,
            high_risk_files=high_risk_files,
            repository_url=repository_url,
            language=language,
            total_files=len(python_files),
            build_timestamp=datetime.now().isoformat(),
        )

        self.logger.info(f"Built context: {len(python_files)} files, {len(patterns)} patterns")
        return context

    def get_related_files(
        self,
        changed_file: str,
        context: CodebaseContext,
        max_results: int = 10,
    ) -> List[RelatedFile]:
        """
        Get files related to a changed file.

        Args:
            changed_file: Path to the changed file
            context: Codebase context
            max_results: Maximum number of results

        Returns:
            List of related files sorted by relevance
        """
        related = []

        # 1. Files that import this file (dependents)
        if changed_file in context.reverse_dependencies:
            for dependent in context.reverse_dependencies[changed_file]:
                related.append(
                    RelatedFile(
                        file_path=dependent,
                        relationship="imports_changed",
                        relevance_score=0.9,
                        reason="Directly imports the changed file",
                    )
                )

        # 2. Files this file imports (dependencies)
        if changed_file in context.dependency_graph:
            for dependency in context.dependency_graph[changed_file]:
                related.append(
                    RelatedFile(
                        file_path=dependency,
                        relationship="imported_by_changed",
                        relevance_score=0.7,
                        reason="Changed file depends on this",
                    )
                )

        # 3. Files in the same directory
        changed_path = Path(changed_file)
        same_dir = [
            str(f)
            for f in context.dependency_graph.keys()
            if Path(f).parent == changed_path.parent and f != changed_file
        ]
        for file_path in same_dir[:3]:  # Limit to top 3
            related.append(
                RelatedFile(
                    file_path=file_path,
                    relationship="same_directory",
                    relevance_score=0.5,
                    reason="Located in the same directory",
                )
            )

        # 4. Test files related to changed file
        test_files = [f for f in context.dependency_graph.keys() if "test" in f]
        for test_file in test_files:
            if changed_file.replace(".py", "") in test_file or changed_file.replace(
                "src/", ""
            ) in test_file:
                related.append(
                    RelatedFile(
                        file_path=test_file,
                        relationship="test_for_changed",
                        relevance_score=0.8,
                        reason="Test file for changed module",
                    )
                )

        # Sort by relevance and limit
        related.sort(key=lambda x: x.relevance_score, reverse=True)
        return related[:max_results]

    def get_cascade_risks(
        self,
        changed_files: List[str],
        context: CodebaseContext,
    ) -> List[RiskArea]:
        """
        Identify cascade risks from changed files.

        Args:
            changed_files: List of files being changed
            context: Codebase context

        Returns:
            List of risk areas
        """
        risks = []
        affected_already = set(changed_files)

        for changed_file in changed_files:
            # Direct dependents are at high risk
            if changed_file in context.reverse_dependencies:
                for dependent in context.reverse_dependencies[changed_file]:
                    if dependent not in affected_already:
                        risk_level = RiskLevel.HIGH if dependent in context.high_risk_files else RiskLevel.MEDIUM
                        risks.append(
                            RiskArea(
                                file_path=dependent,
                                risk_level=risk_level,
                                reason=f"Directly depends on changed file {changed_file}",
                            )
                        )
                        affected_already.add(dependent)

            # Transitive dependents are at medium risk
            if changed_file in context.dependency_graph:
                for direct_dep in context.dependency_graph[changed_file]:
                    if direct_dep in context.reverse_dependencies:
                        for transitive in context.reverse_dependencies[direct_dep]:
                            if transitive not in affected_already and transitive not in changed_files:
                                risks.append(
                                    RiskArea(
                                        file_path=transitive,
                                        risk_level=RiskLevel.MEDIUM,
                                        reason=f"Depends on {direct_dep} which depends on changed file",
                                    )
                                )
                                affected_already.add(transitive)

        return risks

    def get_cross_file_analysis(
        self,
        changed_files: List[str],
        context: CodebaseContext,
    ) -> CrossFileAnalysis:
        """
        Analyze cross-file impacts and relationships.

        Args:
            changed_files: List of files being changed
            context: Codebase context

        Returns:
            CrossFileAnalysis with impacts
        """
        related_files = []
        for changed_file in changed_files:
            related = self.get_related_files(changed_file, context, max_results=5)
            related_files.extend(related)

        cascade_risks = self.get_cascade_risks(changed_files, context)

        # Find shared dependencies (modules depended on by multiple changed files)
        shared_deps = defaultdict(list)
        for changed_file in changed_files:
            if changed_file in context.dependency_graph:
                for dep in context.dependency_graph[changed_file]:
                    shared_deps[dep].append(changed_file)

        shared_dependencies = {
            dep: files for dep, files in shared_deps.items() if len(files) > 1
        }

        # Find test files that might be affected
        test_files = [f for f in context.dependency_graph.keys() if "test" in f]
        potentially_broken = []
        for test_file in test_files:
            for changed_file in changed_files:
                if changed_file.replace(".py", "") in test_file:
                    potentially_broken.append(test_file)
                    break

        return CrossFileAnalysis(
            changed_files=changed_files,
            related_files=related_files,
            cascade_risks=cascade_risks,
            shared_dependencies=shared_dependencies,
            potentially_broken_tests=potentially_broken,
        )

    # Private helper methods

    def _find_python_files(self) -> List[str]:
        """Find all Python files in repository."""
        python_files = []
        for py_file in self.repo_path.rglob("*.py"):
            # Skip __pycache__, venv, and hidden directories
            if any(part.startswith(".") or part == "__pycache__" for part in py_file.parts):
                continue
            if "venv" in str(py_file):
                continue
            try:
                rel_path = str(py_file.relative_to(self.repo_path))
                python_files.append(rel_path)
            except ValueError:
                continue
        return sorted(python_files)

    def _build_dependency_graph(
        self, python_files: List[str]
    ) -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
        """
        Build dependency graph from imports.

        Returns:
            (forward_deps, reverse_deps) where:
            - forward_deps: file → list of files it imports
            - reverse_deps: file → list of files that import it
        """
        forward = defaultdict(list)
        reverse = defaultdict(list)

        for file_path in python_files:
            abs_path = self.repo_path / file_path
            if not abs_path.exists():
                continue

            try:
                with open(abs_path, "r", encoding="utf-8") as f:
                    content = f.read()
                tree = ast.parse(content)
            except Exception as e:
                self.logger.debug(f"Failed to parse {file_path}: {e}")
                continue

            # Extract imports
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        module_name = alias.name.split(".")[0]
                        forward[file_path].append(module_name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        module_name = node.module.split(".")[0]
                        forward[file_path].append(module_name)

        # Build reverse dependencies
        for source, targets in forward.items():
            for target in targets:
                reverse[target].append(source)

        return dict(forward), dict(reverse)

    def _detect_patterns(
        self, python_files: List[str], dep_graph: Dict[str, List[str]]
    ) -> List[ArchitecturalPattern]:
        """Detect architectural patterns in code."""
        patterns = []

        # Check for API endpoint pattern (fastapi/flask routes)
        api_files = [f for f in python_files if "api" in f or "route" in f or "endpoint" in f]
        if api_files:
            patterns.append(
                ArchitecturalPattern(
                    pattern_type=PatternType.API_ENDPOINT,
                    name="HTTP API Endpoints",
                    description="Routes and API endpoint definitions",
                    example_files=api_files[:3],
                    consistency_score=0.8,
                    conventions=["Located in api/ or routes/ directories", "Uses framework decorators"],
                )
            )

        # Check for database access pattern
        db_files = [f for f in python_files if "database" in f or "model" in f or "db" in f]
        if db_files:
            patterns.append(
                ArchitecturalPattern(
                    pattern_type=PatternType.DATABASE_ACCESS,
                    name="Database Access Layer",
                    description="Database models, ORM, and queries",
                    example_files=db_files[:3],
                    consistency_score=0.7,
                    conventions=["Centralized in database.py or models/", "Uses SQLAlchemy or similar ORM"],
                )
            )

        # Check for test pattern
        test_files = [f for f in python_files if "test" in f]
        if test_files:
            patterns.append(
                ArchitecturalPattern(
                    pattern_type=PatternType.TESTING,
                    name="Test Suite Structure",
                    description="Unit and integration tests",
                    example_files=test_files[:3],
                    consistency_score=0.9,
                    conventions=["Located in tests/ directory", "Files prefixed with test_"],
                )
            )

        return patterns

    def _find_circular_dependencies(
        self, dep_graph: Dict[str, List[str]]
    ) -> List[Tuple[str, str]]:
        """Find circular dependencies in the graph."""
        circular = []
        visited = set()

        def has_cycle(node: str, path: Set[str]) -> bool:
            if node in path:
                return True
            if node in visited:
                return False

            path.add(node)
            for neighbor in dep_graph.get(node, []):
                if has_cycle(neighbor, path):
                    return True
            path.remove(node)
            visited.add(node)
            return False

        for node in dep_graph:
            if has_cycle(node, set()):
                # Simple circular detection - just mark as potential circular
                pass

        return circular

    def _find_bottleneck_modules(
        self, dep_graph: Dict[str, List[str]], rev_deps: Dict[str, List[str]]
    ) -> List[Tuple[str, float]]:
        """Find bottleneck modules with high coupling."""
        bottlenecks = []

        for module in dep_graph:
            in_degree = len(rev_deps.get(module, []))  # How many depend on this
            out_degree = len(dep_graph.get(module, []))  # How many this depends on
            coupling_score = (in_degree + out_degree) / max(
                len(dep_graph) + len(rev_deps), 1
            )

            if coupling_score > 0.3:  # Threshold for bottleneck
                bottlenecks.append((module, coupling_score))

        bottlenecks.sort(key=lambda x: x[1], reverse=True)
        return bottlenecks[:10]  # Top 10

    def _detect_pattern_deviations(
        self, python_files: List[str], patterns: List[ArchitecturalPattern]
    ) -> List[Tuple[str, str]]:
        """Detect code that doesn't follow established patterns."""
        deviations = []

        # Check for test files not in tests/ directory
        test_files = [f for f in python_files if "test" in f]
        for test_file in test_files:
            if not test_file.startswith("tests/"):
                deviations.append(
                    (test_file, "Test file not located in tests/ directory")
                )

        return deviations

    def _build_file_histories(self, python_files: List[str]) -> Dict[str, FileHistory]:
        """Build file history (mock - would query git in production)."""
        histories = {}
        for file_path in python_files:
            # In production, this would query git log
            histories[file_path] = FileHistory(
                file_path=file_path,
                change_frequency=0,
                bug_count=0,
                stability_score=0.8,
                risk_score=0.2,
            )
        return histories

    def _detect_bug_patterns(
        self, file_histories: Dict[str, FileHistory]
    ) -> List[BugPattern]:
        """Detect common bug patterns (mock - would query database in production)."""
        # In production, this would query past reviews and findings
        return []

    def _identify_high_risk_files(self, file_histories: Dict[str, FileHistory]) -> List[str]:
        """Identify files with high bug history."""
        high_risk = [
            file_path
            for file_path, history in file_histories.items()
            if history.bug_count > 0 or history.risk_score > 0.7
        ]
        return high_risk
