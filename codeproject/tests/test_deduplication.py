"""
Deduplication Service Tests

Tests for similarity detection, grouping, and diversity factor calculation.
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from src.database import (
    Base,
    Review,
    Finding,
    FindingCategory,
    FindingSeverity,
)
from src.learning.deduplication import DeduplicationService


# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def test_db() -> Session:
    """Create in-memory SQLite database for testing."""
    engine = create_engine("sqlite:///:memory:", echo=False)
    with engine.begin() as connection:
        connection.execute(text("PRAGMA foreign_keys = ON"))
    Base.metadata.create_all(engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    yield db
    db.close()


def create_review(db: Session, pr_id: int = 1) -> Review:
    """Helper to create a test review."""
    review = Review(
        pr_id=pr_id,
        repo_url="https://github.com/test/repo",
        commit_sha="abc123def456",
        branch="main",
        created_at=datetime.now(timezone.utc),
    )
    db.add(review)
    db.commit()
    return review


def create_finding(
    db: Session,
    review_id: int,
    title: str = "SQL Injection",
    description: str = "Test finding",
    severity: FindingSeverity = FindingSeverity.CRITICAL,
    category: FindingCategory = FindingCategory.SECURITY,
) -> Finding:
    """Helper to create a test finding."""
    finding = Finding(
        review_id=review_id,
        category=category,
        severity=severity,
        title=title,
        description=description,
        file_path="src/test.py",
        line_number=42,
        created_at=datetime.now(timezone.utc),
    )
    db.add(finding)
    db.commit()
    return finding


# ============================================================================
# String Similarity Tests
# ============================================================================


class TestStringSimilarity:
    """Test string similarity calculation."""

    def test_identical_strings(self, test_db: Session):
        """Identical strings have similarity 1.0."""
        service = DeduplicationService(test_db)
        similarity = service._string_similarity("SQL Injection", "SQL Injection")
        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_completely_different_strings(self, test_db: Session):
        """Completely different strings have low similarity."""
        service = DeduplicationService(test_db)
        similarity = service._string_similarity("SQL Injection", "Buffer Overflow")
        assert similarity < 0.5

    def test_case_insensitive(self, test_db: Session):
        """String similarity is case-insensitive."""
        service = DeduplicationService(test_db)
        sim1 = service._string_similarity("SQL Injection", "sql injection")
        assert sim1 == pytest.approx(1.0, rel=0.01)

    def test_partial_match(self, test_db: Session):
        """Partially matching strings have moderate similarity."""
        service = DeduplicationService(test_db)
        similarity = service._string_similarity(
            "SQL Injection in auth module", "SQL Injection in user service"
        )
        assert similarity > 0.6


# ============================================================================
# Similarity Detection Tests
# ============================================================================


class TestSimilarityDetection:
    """Test finding similarity calculation."""

    def test_identical_findings(self, test_db: Session):
        """Identical findings have similarity 1.0."""
        review = create_review(test_db)
        f1 = create_finding(
            test_db, review.id, title="SQL Injection", description="Unsafe query"
        )
        f2 = create_finding(
            test_db, review.id, title="SQL Injection", description="Unsafe query"
        )

        service = DeduplicationService(test_db)
        similarity = service.calculate_similarity(f1, f2)
        assert similarity == pytest.approx(1.0, rel=0.01)

    def test_different_category_reduces_similarity(self, test_db: Session):
        """Different category reduces similarity score."""
        review = create_review(test_db)
        f1 = create_finding(
            test_db,
            review.id,
            title="Issue",
            category=FindingCategory.SECURITY,
        )
        f2 = create_finding(
            test_db,
            review.id,
            title="Issue",
            category=FindingCategory.PERFORMANCE,
        )

        service = DeduplicationService(test_db)
        similarity = service.calculate_similarity(f1, f2)
        # Title match (0.4) + description match (0.3) = 0.7, missing 0.2 from category
        assert similarity <= 0.8

    def test_different_severity_affects_similarity(self, test_db: Session):
        """Different severity reduces similarity score."""
        review = create_review(test_db)
        f1 = create_finding(
            test_db,
            review.id,
            title="Issue",
            severity=FindingSeverity.CRITICAL,
        )
        f2 = create_finding(
            test_db,
            review.id,
            title="Issue",
            severity=FindingSeverity.LOW,
        )

        service = DeduplicationService(test_db)
        similarity = service.calculate_similarity(f1, f2)
        # Should still be high due to title match, but affected by severity mismatch
        assert 0.8 < similarity < 1.0

    def test_similar_titles(self, test_db: Session):
        """Similar titles increase overall similarity."""
        review = create_review(test_db)
        f1 = create_finding(
            test_db, review.id, title="SQL Injection in auth module"
        )
        f2 = create_finding(
            test_db, review.id, title="SQL Injection in database"
        )

        service = DeduplicationService(test_db)
        similarity = service.calculate_similarity(f1, f2)
        assert similarity > 0.7


# ============================================================================
# Finding Similar Findings Tests
# ============================================================================


class TestFindSimilarFindings:
    """Test finding similar findings."""

    def test_find_similar_findings(self, test_db: Session):
        """Find similar findings above threshold."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        f2 = create_finding(test_db, review.id, title="SQL Injection")
        f3 = create_finding(test_db, review.id, title="Buffer Overflow")

        service = DeduplicationService(test_db, similarity_threshold=0.7)
        similar = service.find_similar_findings(f1.id)

        # Should find f2 (similar), not f3 (different)
        assert len(similar) == 1
        assert similar[0][0].id == f2.id

    def test_similar_findings_sorted_by_similarity(self, test_db: Session):
        """Similar findings sorted by similarity descending."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        f2 = create_finding(test_db, review.id, title="SQL Injection")  # Exact match
        f3 = create_finding(
            test_db, review.id, title="SQL Injection in database"
        )  # Close match

        service = DeduplicationService(test_db, similarity_threshold=0.7)
        similar = service.find_similar_findings(f1.id)

        # Should be sorted: f2 (exact) > f3 (close)
        assert similar[0][0].id == f2.id
        assert similar[1][0].id == f3.id

    def test_find_similar_empty_when_threshold_not_met(self, test_db: Session):
        """No similar findings when all below threshold."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        create_finding(test_db, review.id, title="Buffer Overflow")

        service = DeduplicationService(test_db, similarity_threshold=0.9)
        similar = service.find_similar_findings(f1.id)

        assert similar == []


# ============================================================================
# Grouping Tests
# ============================================================================


class TestGrouping:
    """Test finding grouping by similarity."""

    def test_group_identical_findings(self, test_db: Session):
        """Identical findings are grouped together."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        f2 = create_finding(test_db, review.id, title="SQL Injection")
        f3 = create_finding(test_db, review.id, title="Buffer Overflow")

        service = DeduplicationService(test_db, similarity_threshold=0.9)
        groups = service.group_similar_findings()

        # Should have 2 groups: f1+f2, f3
        assert len(groups) == 2
        group1 = [f.id for f in groups[0]]
        assert set(group1) == {f1.id, f2.id} or set(group1) == {f3.id}

    def test_group_transitive_similarity(self, test_db: Session):
        """Transitive similarity: A~B, B~C → A, B, C in same group."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        f2 = create_finding(
            test_db, review.id, title="SQL Injection in auth"
        )  # Similar to f1 and f3
        f3 = create_finding(
            test_db, review.id, title="SQL Injection in database"
        )  # Similar to f2

        service = DeduplicationService(test_db, similarity_threshold=0.75)
        groups = service.group_similar_findings()

        # All three should be in same group
        all_in_one_group = any(
            set(f.id for f in group) == {f1.id, f2.id, f3.id} for group in groups
        )
        assert all_in_one_group

    def test_empty_database_returns_empty(self, test_db: Session):
        """Grouping empty database returns empty list."""
        service = DeduplicationService(test_db)
        groups = service.group_similar_findings()
        assert groups == []


# ============================================================================
# Diversity Factor Tests
# ============================================================================


class TestDiversityFactor:
    """Test diversity factor calculation."""

    def test_no_similar_shown_full_diversity(self, test_db: Session):
        """No similar shown findings → diversity factor 1.0."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        f2 = create_finding(test_db, review.id, title="Buffer Overflow")

        service = DeduplicationService(test_db)
        diversity = service.calculate_diversity_factor(f1.id, [f2.id])

        assert diversity == pytest.approx(1.0, rel=0.01)

    def test_identical_shown_reduces_diversity(self, test_db: Session):
        """Identical shown finding reduces diversity factor."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        f2 = create_finding(test_db, review.id, title="SQL Injection")

        service = DeduplicationService(test_db)
        diversity = service.calculate_diversity_factor(f1.id, [f2.id])

        # Identical → reduction of 1.0 * 0.5 = 0.5 → factor = 0.5
        assert diversity == pytest.approx(0.5, rel=0.01)

    def test_partially_similar_partial_reduction(self, test_db: Session):
        """Partially similar shown finding partially reduces diversity."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        f2 = create_finding(
            test_db, review.id, title="SQL Injection in auth"
        )  # ~80% similar

        service = DeduplicationService(test_db, similarity_threshold=0.7)
        diversity = service.calculate_diversity_factor(f1.id, [f2.id])

        # Similarity ~0.8, reduction = 0.8 * 0.5 = 0.4, factor = 0.6
        assert 0.5 < diversity < 1.0

    def test_multiple_similar_shown_max_reduction(self, test_db: Session):
        """Multiple similar shown → uses max similarity for reduction."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        f2 = create_finding(test_db, review.id, title="SQL Injection")  # Exact
        f3 = create_finding(
            test_db, review.id, title="SQL Injection in auth"
        )  # 80% similar

        service = DeduplicationService(test_db, similarity_threshold=0.7)
        diversity = service.calculate_diversity_factor(f1.id, [f2.id, f3.id])

        # Max similarity is 1.0 (f2), reduction = 1.0 * 0.5 = 0.5 → factor = 0.5
        assert diversity == pytest.approx(0.5, rel=0.01)

    def test_diversity_factor_capped_at_0_5(self, test_db: Session):
        """Diversity factor never goes below 0.5."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        f2 = create_finding(test_db, review.id, title="SQL Injection")

        service = DeduplicationService(test_db)
        diversity = service.calculate_diversity_factor(f1.id, [f2.id])

        assert diversity >= 0.5


# ============================================================================
# Deduplication Tests
# ============================================================================


class TestDeduplication:
    """Test finding deduplication."""

    def test_deduplicate_selects_diverse_findings(self, test_db: Session):
        """Deduplication selects diverse findings."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        f2 = create_finding(test_db, review.id, title="SQL Injection")  # Duplicate
        f3 = create_finding(test_db, review.id, title="Buffer Overflow")

        service = DeduplicationService(test_db, similarity_threshold=0.8)
        dedup = service.deduplicate_findings([f1, f2, f3], max_shown=2)

        # Should select f1 and f3 (diverse), not f1 and f2 (similar)
        assert len(dedup) == 2
        ids = {f.id for f in dedup}
        assert f1.id in ids or f2.id in ids  # One of the SQL injections
        assert f3.id in ids  # Buffer overflow

    def test_deduplicate_respects_max_shown(self, test_db: Session):
        """Deduplication respects max_shown limit."""
        review = create_review(test_db)

        findings = [
            create_finding(test_db, review.id, title=f"Issue {i}")
            for i in range(10)
        ]

        service = DeduplicationService(test_db)
        dedup = service.deduplicate_findings(findings, max_shown=3)

        assert len(dedup) == 3

    def test_deduplicate_empty_list(self, test_db: Session):
        """Deduplicating empty list returns empty."""
        service = DeduplicationService(test_db)
        dedup = service.deduplicate_findings([])
        assert dedup == []


# ============================================================================
# Reporting Tests
# ============================================================================


class TestReporting:
    """Test deduplication reporting."""

    def test_deduplication_report(self, test_db: Session):
        """Generate deduplication report."""
        review = create_review(test_db)

        f1 = create_finding(test_db, review.id, title="SQL Injection")
        f2 = create_finding(test_db, review.id, title="SQL Injection")
        f3 = create_finding(test_db, review.id, title="Buffer Overflow")

        service = DeduplicationService(test_db, similarity_threshold=0.8)
        report = service.get_deduplication_report()

        assert report["total_findings"] == 3
        assert report["total_groups"] >= 1
        assert "groups" in report

    def test_report_statistics(self, test_db: Session):
        """Report includes group statistics."""
        review = create_review(test_db)

        for i in range(5):
            create_finding(test_db, review.id, title="Issue")

        service = DeduplicationService(test_db)
        report = service.get_deduplication_report()

        assert report["total_findings"] == 5
        assert "avg_group_size" in report
        assert "max_group_size" in report
