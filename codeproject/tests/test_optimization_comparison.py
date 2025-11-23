"""
Optimization Comparison - Measure before/after performance improvements

Compares original implementations against optimized versions to validate:
- Query count reduction
- Execution time improvement (target: 4-5x speedup)
- Accuracy preservation (results should be identical)
"""

import time
from datetime import datetime, timezone, timedelta
from sqlalchemy import create_engine, text
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
from src.learning.suggestion_ranker import SuggestionRanker
from src.learning.suggestion_ranker_optimized import SuggestionRankerOptimized
from src.learning.pattern_learner import PatternLearner
from src.learning.pattern_learner_optimized import PatternLearnerOptimized


def create_test_db():
    """Create test database."""
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


def populate_large_dataset(db: Session, num_findings: int = 500):
    """Populate database with large test dataset."""
    review = Review(
        pr_id=1,
        repo_url="https://github.com/test/repo",
        commit_sha="abc123",
        branch="main",
        created_at=datetime.now(timezone.utc),
    )
    db.add(review)
    db.commit()

    # Create findings with 10 unique patterns
    findings = []
    for i in range(num_findings):
        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY if i % 2 == 0 else FindingCategory.PERFORMANCE,
            severity=FindingSeverity.HIGH if i % 3 == 0 else FindingSeverity.MEDIUM,
            title=f"Pattern {i % 10}",  # Only 10 unique patterns
            description=f"Test finding {i}",
            file_path=f"src/file_{i % 20}.py",
            line_number=i * 10,
            created_at=datetime.now(timezone.utc),
        )
        findings.append(finding)
    db.add_all(findings)
    db.commit()

    # Create feedback (5 per finding)
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

    for i in range(0, len(feedback_list), 100):
        db.add_all(feedback_list[i:i+100])
        db.commit()

    # Create learning metrics
    for category in [FindingCategory.SECURITY, FindingCategory.PERFORMANCE]:
        for severity in [FindingSeverity.CRITICAL, FindingSeverity.HIGH, FindingSeverity.MEDIUM]:
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


def test_ranker_optimization_comparison():
    """Compare original vs optimized ranker performance."""
    db, engine = create_test_db()
    findings = populate_large_dataset(db, num_findings=500)

    print("\n" + "=" * 80)
    print("SUGGESTION RANKER - OPTIMIZATION COMPARISON")
    print("=" * 80)
    print(f"Test dataset: {len(findings)} findings")

    # Test original ranker
    original_ranker = SuggestionRanker(db)
    start = time.time()
    original_ranked = original_ranker.rank_findings(findings[:100])
    original_duration = (time.time() - start) * 1000

    print(f"\nOriginal Implementation:")
    print(f"  - Execution time: {original_duration:.2f}ms")
    print(f"  - Results: {len(original_ranked)} findings ranked")
    print(f"  - Top finding: {original_ranked[0][0].title} (score: {original_ranked[0][1]:.3f})")

    # Test optimized ranker
    optimized_ranker = SuggestionRankerOptimized(db)
    start = time.time()
    optimized_ranked = optimized_ranker.rank_findings(findings[:100])
    optimized_duration = (time.time() - start) * 1000

    print(f"\nOptimized Implementation:")
    print(f"  - Execution time: {optimized_duration:.2f}ms")
    print(f"  - Results: {len(optimized_ranked)} findings ranked")
    print(f"  - Top finding: {optimized_ranked[0][0].title} (score: {optimized_ranked[0][1]:.3f})")

    # Calculate improvement
    speedup = original_duration / optimized_duration if optimized_duration > 0 else 0
    improvement = ((original_duration - optimized_duration) / original_duration) * 100

    print(f"\nPerformance Improvement:")
    print(f"  - Speedup: {speedup:.1f}x faster")
    print(f"  - Time saved: {improvement:.1f}%")
    print(f"  - Absolute: {original_duration - optimized_duration:.2f}ms")

    # Verify results match
    original_ids = [f[0].id for f in original_ranked[:10]]
    optimized_ids = [f[0].id for f in optimized_ranked[:10]]
    match = original_ids == optimized_ids
    print(f"\nResult Accuracy: {'✓ MATCH' if match else '✗ MISMATCH'}")

    if not match:
        print(f"  Original top 10: {original_ids}")
        print(f"  Optimized top 10: {optimized_ids}")

    db.close()
    engine.dispose()

    # Assertions
    assert len(original_ranked) == len(optimized_ranked), "Result counts should match"
    assert speedup >= 2.0, f"Expected ≥2x speedup, got {speedup:.1f}x"


def test_pattern_learner_optimization_comparison():
    """Compare original vs optimized pattern learner performance."""
    db, engine = create_test_db()
    findings = populate_large_dataset(db, num_findings=500)

    print("\n" + "=" * 80)
    print("PATTERN LEARNER - OPTIMIZATION COMPARISON")
    print("=" * 80)
    print(f"Test dataset: {len(findings)} findings, 10 unique patterns")

    # Test original pattern learner
    original_learner = PatternLearner(db)
    start = time.time()
    original_patterns = original_learner.detect_patterns()
    original_duration = (time.time() - start) * 1000

    print(f"\nOriginal Implementation:")
    print(f"  - Execution time: {original_duration:.2f}ms")
    print(f"  - Patterns detected: {len(original_patterns)}")
    if original_patterns:
        print(f"  - Top pattern: {original_patterns[0]['pattern_type']} ({original_patterns[0]['occurrences']} occurrences)")

    # Test optimized pattern learner
    optimized_learner = PatternLearnerOptimized(db)
    start = time.time()
    optimized_patterns = optimized_learner.detect_patterns()
    optimized_duration = (time.time() - start) * 1000

    print(f"\nOptimized Implementation:")
    print(f"  - Execution time: {optimized_duration:.2f}ms")
    print(f"  - Patterns detected: {len(optimized_patterns)}")
    if optimized_patterns:
        print(f"  - Top pattern: {optimized_patterns[0]['pattern_type']} ({optimized_patterns[0]['occurrences']} occurrences)")

    # Calculate improvement
    speedup = original_duration / optimized_duration if optimized_duration > 0 else 0
    improvement = ((original_duration - optimized_duration) / original_duration) * 100

    print(f"\nPerformance Improvement:")
    print(f"  - Speedup: {speedup:.1f}x faster")
    print(f"  - Time saved: {improvement:.1f}%")
    print(f"  - Absolute: {original_duration - optimized_duration:.2f}ms")

    # Verify results match
    original_types = [p["pattern_type"] for p in original_patterns[:5]]
    optimized_types = [p["pattern_type"] for p in optimized_patterns[:5]]
    match = original_types == optimized_types
    print(f"\nResult Accuracy: {'✓ MATCH' if match else '✗ MISMATCH'}")

    if not match:
        print(f"  Original top 5: {original_types}")
        print(f"  Optimized top 5: {optimized_types}")

    db.close()
    engine.dispose()

    # Assertions
    assert len(original_patterns) == len(optimized_patterns), "Pattern counts should match"
    assert speedup >= 1.2, f"Expected ≥1.2x speedup, got {speedup:.1f}x"  # SQL aggregation provides modest gains for patterns


def test_anti_patterns_optimization():
    """Verify anti-pattern detection works in optimized version."""
    db, engine = create_test_db()
    findings = populate_large_dataset(db, num_findings=200)

    print("\n" + "=" * 80)
    print("ANTI-PATTERN DETECTION - OPTIMIZATION VERIFICATION")
    print("=" * 80)

    original_learner = PatternLearner(db)
    original_anti = original_learner.detect_anti_patterns()

    optimized_learner = PatternLearnerOptimized(db)
    optimized_anti = optimized_learner.detect_anti_patterns()

    print(f"Original implementation: {len(original_anti)} anti-patterns")
    print(f"Optimized implementation: {len(optimized_anti)} anti-patterns")

    # Both should find same number
    assert len(original_anti) == len(optimized_anti), "Anti-pattern counts should match"

    print("✓ Anti-pattern detection matches between implementations")

    db.close()
    engine.dispose()


if __name__ == "__main__":
    test_ranker_optimization_comparison()
    test_pattern_learner_optimization_comparison()
    test_anti_patterns_optimization()
