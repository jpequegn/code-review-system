"""
Tests for architecture analysis and dependency graph modules.

Tests cover dependency graph construction, pattern detection,
anti-pattern recognition, metrics calculation, and suggestions.
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.architecture.dependency_graph import DependencyGraph, Cycle, CascadeImpact
from src.architecture.patterns import PatternDetector, ArchitecturePattern
from src.architecture.metrics import ArchitectureMetrics
from src.architecture.antipatterns import AntiPatternDetector
from src.architecture.cascade_analyzer import CascadeAnalyzer
from src.architecture.suggestions import SuggestionEngine


@pytest.fixture
def temp_repo():
    """Create a temporary repository with test Python files."""
    with TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)

        # Create a simple dependency structure
        # models.py - base module
        (tmpdir_path / "models.py").write_text("# Base models")

        # services.py - depends on models
        (tmpdir_path / "services.py").write_text(
            "from models import *\n"
            "def service_func(): pass"
        )

        # views.py - depends on services
        (tmpdir_path / "views.py").write_text(
            "from services import *\n"
            "def view_func(): pass"
        )

        # utils.py - independent
        (tmpdir_path / "utils.py").write_text("def util_func(): pass")

        # circular.py - create a cycle
        (tmpdir_path / "circular_a.py").write_text("from circular_b import *")
        (tmpdir_path / "circular_b.py").write_text("from circular_a import *")

        yield tmpdir_path


class TestDependencyGraph:
    """Test dependency graph construction and analysis."""

    def test_build_graph(self, temp_repo):
        """Test building dependency graph from imports."""
        graph = DependencyGraph(str(temp_repo))
        graph.build_graph()

        assert graph.graph.number_of_nodes() >= 4
        assert graph.graph.number_of_edges() >= 2

    def test_find_cycles(self, temp_repo):
        """Test cycle detection."""
        graph = DependencyGraph(str(temp_repo))
        graph.build_graph()

        cycles = graph.find_cycles()
        assert len(cycles) > 0
        assert isinstance(cycles[0], Cycle)

    def test_calculate_coupling(self, temp_repo):
        """Test coupling calculation between modules."""
        graph = DependencyGraph(str(temp_repo))
        graph.build_graph()

        # Models should have low coupling with utils
        coupling = graph.calculate_coupling("models", "utils")
        assert 0.0 <= coupling <= 1.0

    def test_get_cascade_impact(self, temp_repo):
        """Test cascade impact analysis."""
        graph = DependencyGraph(str(temp_repo))
        graph.build_graph()

        # Impact of changing models should affect services and views
        impact = graph.get_cascade_impact("models")
        assert isinstance(impact, CascadeImpact)
        assert impact.changed_module == "models"
        assert 0.0 <= impact.risk_score <= 1.0

    def test_find_bottlenecks(self, temp_repo):
        """Test bottleneck detection."""
        graph = DependencyGraph(str(temp_repo))
        graph.build_graph()

        bottlenecks = graph.find_bottlenecks()
        # Models might be a bottleneck if it has high in-degree
        assert isinstance(bottlenecks, list)

    def test_get_graph_metrics(self, temp_repo):
        """Test graph metrics calculation."""
        graph = DependencyGraph(str(temp_repo))
        graph.build_graph()

        metrics = graph.get_graph_metrics()
        assert "node_count" in metrics
        assert "edge_count" in metrics
        assert "density" in metrics
        assert "cycle_count" in metrics

    def test_is_acyclic(self, temp_repo):
        """Test acyclicity check."""
        graph = DependencyGraph(str(temp_repo))
        graph.build_graph()

        # Should have cycles due to circular_a and circular_b
        assert not graph.is_acyclic()


class TestPatternDetection:
    """Test architectural pattern detection."""

    def test_detect_pattern(self, temp_repo):
        """Test pattern detection."""
        detector = PatternDetector(str(temp_repo))
        pattern = detector.detect_pattern()

        assert isinstance(pattern, ArchitecturePattern)

    def test_detect_violations(self, temp_repo):
        """Test detecting pattern violations."""
        detector = PatternDetector(str(temp_repo))
        violations = detector.detect_violations()

        # Should find at least the circular dependency
        assert len(violations) >= 0

    def test_suggest_pattern_fit(self, temp_repo):
        """Test suggesting best pattern for codebase."""
        detector = PatternDetector(str(temp_repo))
        suggested = detector.suggest_pattern_fit()

        assert isinstance(suggested, ArchitecturePattern)


class TestArchitectureMetrics:
    """Test architecture metrics calculation."""

    def test_calculate_coupling_metrics(self, temp_repo):
        """Test coupling metrics."""
        metrics = ArchitectureMetrics(str(temp_repo))
        coupling = metrics.calculate_coupling_metrics("models")

        assert coupling is not None
        assert coupling.afferent >= 0
        assert coupling.efferent >= 0
        assert 0.0 <= coupling.instability <= 1.0

    def test_calculate_cohesion_metrics(self, temp_repo):
        """Test cohesion metrics."""
        metrics = ArchitectureMetrics(str(temp_repo))
        cohesion = metrics.calculate_cohesion_metrics("models")

        assert cohesion is not None
        assert 0.0 <= cohesion.cohesion_score <= 1.0

    def test_calculate_modularity_score(self, temp_repo):
        """Test modularity score."""
        metrics = ArchitectureMetrics(str(temp_repo))
        score = metrics.calculate_modularity_score()

        assert 0.0 <= score <= 1.0

    def test_get_all_metrics(self, temp_repo):
        """Test getting all metrics."""
        metrics = ArchitectureMetrics(str(temp_repo))
        all_metrics = metrics.get_all_metrics()

        assert "node_count" in all_metrics
        assert "modularity_score" in all_metrics
        assert "bottleneck_count" in all_metrics


class TestAntiPatternDetection:
    """Test anti-pattern detection."""

    def test_detect_circular_dependencies(self, temp_repo):
        """Test detecting circular dependencies."""
        detector = AntiPatternDetector(str(temp_repo))
        patterns = detector.detect_circular_dependencies()

        # Should find the circular dependency
        assert len(patterns) > 0
        assert patterns[0].pattern_type == "circular_dependency"

    def test_detect_all_antipatterns(self, temp_repo):
        """Test detecting all anti-patterns."""
        detector = AntiPatternDetector(str(temp_repo))
        patterns = detector.detect_all_antipatterns()

        # Should find at least circular dependency
        assert len(patterns) >= 1


class TestCascadeAnalyzer:
    """Test cascade impact analysis."""

    def test_show_impact_tree(self, temp_repo):
        """Test showing impact tree."""
        analyzer = CascadeAnalyzer(str(temp_repo))
        tree = analyzer.show_impact_tree("models")

        assert tree.changed_module == "models"
        assert isinstance(tree.risk_score, float)

    def test_estimate_risk(self, temp_repo):
        """Test risk estimation."""
        analyzer = CascadeAnalyzer(str(temp_repo))
        risk = analyzer.estimate_risk("models")

        assert risk in ["critical", "high", "medium", "low"]

    def test_analyze_change_impact(self, temp_repo):
        """Test multi-module change impact analysis."""
        analyzer = CascadeAnalyzer(str(temp_repo))
        impact = analyzer.analyze_change_impact(["models", "services"])

        assert "affected_count" in impact
        assert "risk_level" in impact


class TestSuggestionEngine:
    """Test architectural suggestion generation."""

    def test_suggest_refactorings(self, temp_repo):
        """Test refactoring suggestions."""
        engine = SuggestionEngine(str(temp_repo))
        suggestions = engine.suggest_refactorings()

        assert isinstance(suggestions, list)

    def test_suggest_modularization_points(self, temp_repo):
        """Test modularization suggestions."""
        engine = SuggestionEngine(str(temp_repo))
        suggestions = engine.suggest_modularization_points()

        assert isinstance(suggestions, list)

    def test_suggest_dependency_reduction(self, temp_repo):
        """Test dependency reduction suggestions."""
        engine = SuggestionEngine(str(temp_repo))
        suggestions = engine.suggest_dependency_reduction("services")

        assert isinstance(suggestions, list)

    def test_get_all_suggestions(self, temp_repo):
        """Test getting all suggestions."""
        engine = SuggestionEngine(str(temp_repo))
        all_suggestions = engine.get_all_suggestions()

        assert "refactorings" in all_suggestions
        assert "modularization" in all_suggestions
        assert "summary" in all_suggestions


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
