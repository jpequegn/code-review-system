"""
Query Profiling & Analysis - Identify database bottlenecks

Profiles query performance to identify:
- N+1 query patterns
- Slow queries (>100ms)
- Inefficient aggregations
- Missing indexes

Results guide optimization strategy and index creation.
"""

import time
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text, event
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from src.database import (
    Base,
    Review,
    Finding,
    FindingCategory,
    FindingSeverity,
    SuggestionFeedback,
    LearningMetrics,
    PatternMetrics,
)
from src.learning.insights import InsightsGenerator
from src.learning.suggestion_ranker import SuggestionRanker
from src.learning.pattern_learner import PatternLearner


class QueryProfiler:
    """Profiles database queries to identify bottlenecks."""

    def __init__(self, db: Session):
        self.db = db
        self.queries = []
        self.register_listener()

    def register_listener(self):
        """Register SQLAlchemy event listener to track queries."""
        @event.listens_for(self.db.get_bind(), "before_cursor_execute")
        def receive_before_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            conn.info.setdefault("query_start_time", []).append(time.time())

        @event.listens_for(self.db.get_bind(), "after_cursor_execute")
        def receive_after_cursor_execute(conn, cursor, statement, parameters, context, executemany):
            total_time = time.time() - conn.info["query_start_time"].pop(-1)
            self.queries.append({
                "statement": statement,
                "duration_ms": total_time * 1000,
                "timestamp": datetime.now(timezone.utc),
            })

    def get_slow_queries(self, threshold_ms: float = 50.0):
        """Get queries slower than threshold."""
        return [q for q in self.queries if q["duration_ms"] > threshold_ms]

    def get_query_summary(self):
        """Get query execution summary."""
        if not self.queries:
            return {}

        durations = [q["duration_ms"] for q in self.queries]
        return {
            "total_queries": len(self.queries),
            "total_time_ms": sum(durations),
            "avg_time_ms": sum(durations) / len(durations),
            "min_time_ms": min(durations),
            "max_time_ms": max(durations),
            "slow_queries": len(self.get_slow_queries()),
        }

    def reset(self):
        """Reset query tracking."""
        self.queries = []


def create_profiling_db():
    """Create test database with profiling enabled."""
    engine = create_engine(
        "sqlite:///:memory:",
        echo=False,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys = ON"))
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    return db, engine


def populate_test_data(db: Session, num_findings: int = 100):
    """Populate database with test data."""
    review = Review(
        pr_id=1,
        repo_url="https://github.com/test/repo",
        commit_sha="abc123",
        branch="main",
        created_at=datetime.now(timezone.utc),
    )
    db.add(review)
    db.commit()

    # Create findings
    findings = []
    for i in range(num_findings):
        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY if i % 2 == 0 else FindingCategory.PERFORMANCE,
            severity=FindingSeverity.HIGH,
            title=f"Finding {i % 10}",
            description=f"Test finding {i}",
            file_path=f"src/file_{i % 20}.py",
            line_number=i * 10,
            created_at=datetime.now(timezone.utc),
        )
        findings.append(finding)
    db.add_all(findings)
    db.commit()

    # Create feedback
    feedback_list = []
    for finding in findings:
        for j in range(5):
            feedback = SuggestionFeedback(
                finding_id=finding.id,
                feedback_type="helpful" if j % 2 == 0 else "false_positive",
                confidence=0.85,
                created_at=datetime.now(timezone.utc) - timedelta(days=j),
            )
            feedback_list.append(feedback)

    # Batch insert feedback
    for i in range(0, len(feedback_list), 100):
        db.add_all(feedback_list[i:i+100])
        db.commit()

    # Create learning metrics
    for category in [FindingCategory.SECURITY, FindingCategory.PERFORMANCE]:
        for severity in [FindingSeverity.CRITICAL, FindingSeverity.HIGH]:
            metrics = LearningMetrics(
                repo_url="https://github.com/test/repo",
                category=category,
                severity=severity,
                accuracy=85.0,
                precision=0.88,
                recall=0.92,
                avg_time_to_fix=2.5,
            )
            db.add(metrics)
    db.commit()

    return findings


# ============================================================================
# Profiling Tests
# ============================================================================

def test_insights_generator_profiling():
    """Profile InsightsGenerator queries."""
    db, engine = create_profiling_db()
    profiler = QueryProfiler(db)

    findings = populate_test_data(db, 100)
    repo_url = "https://github.com/test/repo"

    insights = InsightsGenerator(db)

    # Profile calculate_team_metrics
    profiler.reset()
    start = time.time()
    metrics = insights.calculate_team_metrics(repo_url=repo_url)
    duration = (time.time() - start) * 1000

    summary = profiler.get_query_summary()
    slow_queries = profiler.get_slow_queries(threshold_ms=20)

    print("\n" + "=" * 80)
    print("InsightsGenerator.calculate_team_metrics() Profiling")
    print("=" * 80)
    print(f"Total execution time: {duration:.2f}ms")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Query time breakdown:")
    print(f"  - Total query time: {summary['total_time_ms']:.2f}ms")
    print(f"  - Average per query: {summary['avg_time_ms']:.2f}ms")
    print(f"  - Min/Max: {summary['min_time_ms']:.2f}ms / {summary['max_time_ms']:.2f}ms")
    print(f"  - Slow queries (>20ms): {len(slow_queries)}")

    if slow_queries:
        print(f"\nSlow queries:")
        for i, q in enumerate(slow_queries[:5], 1):
            print(f"  {i}. {q['duration_ms']:.2f}ms")
            print(f"     {q['statement'][:100]}...")

    # Profile analyze_trends
    profiler.reset()
    start = time.time()
    trends = insights.analyze_trends(repo_url=repo_url)
    duration = (time.time() - start) * 1000

    summary = profiler.get_query_summary()
    slow_queries = profiler.get_slow_queries(threshold_ms=20)

    print("\n" + "=" * 80)
    print("InsightsGenerator.analyze_trends() Profiling")
    print("=" * 80)
    print(f"Total execution time: {duration:.2f}ms")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Query time breakdown:")
    print(f"  - Total query time: {summary['total_time_ms']:.2f}ms")
    print(f"  - Average per query: {summary['avg_time_ms']:.2f}ms")
    print(f"  - Min/Max: {summary['min_time_ms']:.2f}ms / {summary['max_time_ms']:.2f}ms")
    print(f"  - Slow queries (>20ms): {len(slow_queries)}")

    # Profile generate_learning_paths
    profiler.reset()
    start = time.time()
    paths = insights.generate_learning_paths(repo_url=repo_url)
    duration = (time.time() - start) * 1000

    summary = profiler.get_query_summary()
    slow_queries = profiler.get_slow_queries(threshold_ms=20)

    print("\n" + "=" * 80)
    print("InsightsGenerator.generate_learning_paths() Profiling")
    print("=" * 80)
    print(f"Total execution time: {duration:.2f}ms")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Query time breakdown:")
    print(f"  - Total query time: {summary['total_time_ms']:.2f}ms")
    print(f"  - Average per query: {summary['avg_time_ms']:.2f}ms")
    print(f"  - Min/Max: {summary['min_time_ms']:.2f}ms / {summary['max_time_ms']:.2f}ms")
    print(f"  - Slow queries (>20ms): {len(slow_queries)}")

    db.close()
    engine.dispose()


def test_suggestion_ranker_profiling():
    """Profile SuggestionRanker queries."""
    db, engine = create_profiling_db()
    profiler = QueryProfiler(db)

    findings = populate_test_data(db, 100)

    ranker = SuggestionRanker(db)

    # Profile rank_findings
    profiler.reset()
    start = time.time()
    ranked = ranker.rank_findings(findings)
    duration = (time.time() - start) * 1000

    summary = profiler.get_query_summary()
    slow_queries = profiler.get_slow_queries(threshold_ms=10)

    print("\n" + "=" * 80)
    print("SuggestionRanker.rank_findings() Profiling (100 findings)")
    print("=" * 80)
    print(f"Total execution time: {duration:.2f}ms")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Query time breakdown:")
    print(f"  - Total query time: {summary['total_time_ms']:.2f}ms")
    print(f"  - Average per query: {summary['avg_time_ms']:.2f}ms")
    print(f"  - Min/Max: {summary['min_time_ms']:.2f}ms / {summary['max_time_ms']:.2f}ms")
    print(f"  - Slow queries (>10ms): {len(slow_queries)}")
    print(f"  - Queries per finding: {summary['total_queries'] / len(findings):.1f}")

    if slow_queries:
        print(f"\nSlow queries (sample):")
        for i, q in enumerate(slow_queries[:3], 1):
            print(f"  {i}. {q['duration_ms']:.2f}ms - {q['statement'][:80]}...")

    db.close()
    engine.dispose()


def test_pattern_learner_profiling():
    """Profile PatternLearner queries."""
    db, engine = create_profiling_db()
    profiler = QueryProfiler(db)

    findings = populate_test_data(db, 100)

    learner = PatternLearner(db)

    # Profile detect_patterns
    profiler.reset()
    start = time.time()
    patterns = learner.detect_patterns()
    duration = (time.time() - start) * 1000

    summary = profiler.get_query_summary()

    print("\n" + "=" * 80)
    print("PatternLearner.detect_patterns() Profiling")
    print("=" * 80)
    print(f"Total execution time: {duration:.2f}ms")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Query time breakdown:")
    print(f"  - Total query time: {summary['total_time_ms']:.2f}ms")
    print(f"  - Average per query: {summary['avg_time_ms']:.2f}ms")
    print(f"  - Min/Max: {summary['min_time_ms']:.2f}ms / {summary['max_time_ms']:.2f}ms")
    print(f"\nPatterns detected: {len(patterns)}")
    print(f"Queries per pattern: {summary['total_queries'] / max(len(patterns), 1):.1f}")

    # Profile detect_anti_patterns
    profiler.reset()
    start = time.time()
    anti_patterns = learner.detect_anti_patterns()
    duration = (time.time() - start) * 1000

    summary = profiler.get_query_summary()

    print("\n" + "=" * 80)
    print("PatternLearner.detect_anti_patterns() Profiling")
    print("=" * 80)
    print(f"Total execution time: {duration:.2f}ms")
    print(f"Total queries: {summary['total_queries']}")
    print(f"Query time breakdown:")
    print(f"  - Total query time: {summary['total_time_ms']:.2f}ms")
    print(f"  - Average per query: {summary['avg_time_ms']:.2f}ms")

    db.close()
    engine.dispose()


if __name__ == "__main__":
    test_insights_generator_profiling()
    test_suggestion_ranker_profiling()
    test_pattern_learner_profiling()
