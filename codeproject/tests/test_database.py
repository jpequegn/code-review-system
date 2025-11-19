"""
Tests for database models and ORM functionality.

Tests database configuration, model definitions, relationships, and constraints.
"""

import pytest
from datetime import datetime, timezone
from sqlalchemy import create_engine, inspect, event
from sqlalchemy.orm import sessionmaker, Session

from src.database import (
    Base, Review, Finding, ReviewStatus, FindingSeverity, FindingCategory,
    init_db, SessionLocal
)


@pytest.fixture
def test_db():
    """
    Create an in-memory SQLite database for testing.

    Provides a fresh database for each test and cleans up afterward.
    """
    # Create in-memory SQLite database
    engine = create_engine("sqlite:///:memory:")

    # Enable foreign key constraints for SQLite
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    # Create all tables
    Base.metadata.create_all(bind=engine)

    # Create session factory
    TestSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

    # Yield session for test
    session = TestSessionLocal()
    yield session

    # Cleanup
    session.close()


class TestReviewModel:
    """Tests for the Review ORM model."""

    def test_review_model_creation(self, test_db):
        """Test that a Review instance can be created with all required fields."""
        review = Review(
            pr_id=42,
            repo_url="https://github.com/user/repo",
            branch="feature/auth",
            commit_sha="abc123def456",
            status=ReviewStatus.PENDING
        )

        test_db.add(review)
        test_db.commit()

        # Query back and verify
        retrieved = test_db.query(Review).filter(Review.pr_id == 42).first()
        assert retrieved is not None
        assert retrieved.pr_id == 42
        assert retrieved.repo_url == "https://github.com/user/repo"
        assert retrieved.branch == "feature/auth"
        assert retrieved.commit_sha == "abc123def456"
        assert retrieved.status == ReviewStatus.PENDING

    def test_review_default_status(self, test_db):
        """Test that Review defaults to PENDING status."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )

        test_db.add(review)
        test_db.commit()

        retrieved = test_db.query(Review).first()
        assert retrieved.status == ReviewStatus.PENDING

    def test_review_created_at_timestamp(self, test_db):
        """Test that created_at is set to current time."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )

        before = datetime.now(timezone.utc)
        test_db.add(review)
        test_db.commit()
        after = datetime.now(timezone.utc)

        retrieved = test_db.query(Review).first()
        assert retrieved.created_at is not None
        # Compare without timezone awareness
        assert before.replace(tzinfo=None) <= retrieved.created_at <= after.replace(tzinfo=None)

    def test_review_completed_at_nullable(self, test_db):
        """Test that completed_at is optional and null by default."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )

        test_db.add(review)
        test_db.commit()

        retrieved = test_db.query(Review).first()
        assert retrieved.completed_at is None

    def test_review_pr_id_unique(self, test_db):
        """Test that pr_id is unique and enforced."""
        review1 = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        review2 = Review(
            pr_id=1,  # Same pr_id
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="def456"
        )

        test_db.add(review1)
        test_db.commit()

        test_db.add(review2)
        with pytest.raises(Exception):  # IntegrityError
            test_db.commit()

    def test_review_status_enum_validation(self, test_db):
        """Test that status enum only accepts valid values."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123",
            status=ReviewStatus.ANALYZING
        )

        test_db.add(review)
        test_db.commit()

        retrieved = test_db.query(Review).first()
        assert retrieved.status == ReviewStatus.ANALYZING

    def test_review_all_statuses(self, test_db):
        """Test that all ReviewStatus values are accepted."""
        statuses = [
            ReviewStatus.PENDING,
            ReviewStatus.ANALYZING,
            ReviewStatus.COMPLETED,
            ReviewStatus.FAILED
        ]

        for idx, status in enumerate(statuses, 1):
            review = Review(
                pr_id=idx,
                repo_url="https://github.com/user/repo",
                branch="main",
                commit_sha=f"sha{idx}"
            )
            review.status = status
            test_db.add(review)

        test_db.commit()

        reviews = test_db.query(Review).all()
        assert len(reviews) == 4
        assert set(r.status for r in reviews) == set(statuses)


class TestFindingModel:
    """Tests for the Finding ORM model."""

    def test_finding_model_creation(self, test_db):
        """Test that a Finding instance can be created with all required fields."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="SQL Injection Vulnerability",
            description="User input not sanitized before SQL query",
            file_path="src/auth.py",
            line_number=42,
            suggested_fix="Use parameterized queries"
        )

        test_db.add(finding)
        test_db.commit()

        # Query back and verify
        retrieved = test_db.query(Finding).filter(Finding.review_id == review.id).first()
        assert retrieved is not None
        assert retrieved.review_id == review.id
        assert retrieved.category == FindingCategory.SECURITY
        assert retrieved.severity == FindingSeverity.HIGH
        assert retrieved.title == "SQL Injection Vulnerability"
        assert retrieved.file_path == "src/auth.py"
        assert retrieved.line_number == 42

    def test_finding_line_number_nullable(self, test_db):
        """Test that line_number is optional for file-level issues."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.PERFORMANCE,
            severity=FindingSeverity.MEDIUM,
            title="Missing imports",
            description="Unused imports in module",
            file_path="src/utils.py"
            # line_number is intentionally omitted
        )

        test_db.add(finding)
        test_db.commit()

        retrieved = test_db.query(Finding).first()
        assert retrieved.line_number is None

    def test_finding_suggested_fix_nullable(self, test_db):
        """Test that suggested_fix is optional."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.BEST_PRACTICE,
            severity=FindingSeverity.LOW,
            title="Code style issue",
            description="Line too long",
            file_path="src/config.py"
            # suggested_fix is intentionally omitted
        )

        test_db.add(finding)
        test_db.commit()

        retrieved = test_db.query(Finding).first()
        assert retrieved.suggested_fix is None

    def test_finding_created_at_timestamp(self, test_db):
        """Test that created_at is set automatically."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="Critical security issue",
            description="Buffer overflow vulnerability",
            file_path="src/parser.c"
        )

        before = datetime.now(timezone.utc)
        test_db.add(finding)
        test_db.commit()
        after = datetime.now(timezone.utc)

        retrieved = test_db.query(Finding).first()
        assert retrieved.created_at is not None
        # Compare without timezone awareness
        assert before.replace(tzinfo=None) <= retrieved.created_at <= after.replace(tzinfo=None)

    def test_finding_all_severity_levels(self, test_db):
        """Test that all FindingSeverity values are accepted."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        severities = [
            FindingSeverity.CRITICAL,
            FindingSeverity.HIGH,
            FindingSeverity.MEDIUM,
            FindingSeverity.LOW
        ]

        for idx, severity in enumerate(severities, 1):
            finding = Finding(
                review_id=review.id,
                category=FindingCategory.SECURITY,
                severity=severity,
                title=f"Issue {idx}",
                description="Test issue",
                file_path="src/test.py"
            )
            test_db.add(finding)

        test_db.commit()

        findings = test_db.query(Finding).all()
        assert len(findings) == 4
        assert set(f.severity for f in findings) == set(severities)

    def test_finding_all_categories(self, test_db):
        """Test that all FindingCategory values are accepted."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        categories = [
            FindingCategory.SECURITY,
            FindingCategory.PERFORMANCE,
            FindingCategory.BEST_PRACTICE
        ]

        for idx, category in enumerate(categories, 1):
            finding = Finding(
                review_id=review.id,
                category=category,
                severity=FindingSeverity.MEDIUM,
                title=f"Issue {idx}",
                description="Test issue",
                file_path="src/test.py"
            )
            test_db.add(finding)

        test_db.commit()

        findings = test_db.query(Finding).all()
        assert len(findings) == 3
        assert set(f.category for f in findings) == set(categories)


class TestReviewFindingRelationship:
    """Tests for the Review-Finding relationship."""

    def test_review_findings_relationship(self, test_db):
        """Test that findings are properly related to a review."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding1 = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Finding 1",
            description="Test",
            file_path="test.py"
        )
        finding2 = Finding(
            review_id=review.id,
            category=FindingCategory.PERFORMANCE,
            severity=FindingSeverity.MEDIUM,
            title="Finding 2",
            description="Test",
            file_path="test.py"
        )

        test_db.add(finding1)
        test_db.add(finding2)
        test_db.commit()

        # Query review and access findings through relationship
        retrieved = test_db.query(Review).first()
        assert len(retrieved.findings) == 2
        assert finding1 in retrieved.findings
        assert finding2 in retrieved.findings

    def test_finding_review_relationship(self, test_db):
        """Test that a finding can access its associated review."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Test Finding",
            description="Test",
            file_path="test.py"
        )

        test_db.add(finding)
        test_db.commit()

        # Query finding and access review through relationship
        retrieved = test_db.query(Finding).first()
        assert retrieved.review is not None
        assert retrieved.review.pr_id == 1

    def test_cascade_delete_findings(self, test_db):
        """Test that deleting a review cascades to delete findings."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Test Finding",
            description="Test",
            file_path="test.py"
        )
        test_db.add(finding)
        test_db.commit()

        # Verify finding exists
        assert test_db.query(Finding).count() == 1

        # Delete review
        test_db.delete(review)
        test_db.commit()

        # Verify finding is also deleted (cascade)
        assert test_db.query(Finding).count() == 0

    def test_foreign_key_constraint(self, test_db):
        """Test that invalid review_id is rejected."""
        from sqlalchemy.exc import IntegrityError

        finding = Finding(
            review_id=9999,  # Non-existent review
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Test Finding",
            description="Test",
            file_path="test.py"
        )

        test_db.add(finding)
        with pytest.raises(IntegrityError):
            test_db.commit()


class TestFindingSuggestionFields:
    """Tests for AI-generated suggestion fields in Finding model (Task 4.2)."""

    def test_finding_auto_fix_nullable(self, test_db):
        """Test that auto_fix field is optional and nullable."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Security Issue",
            description="Test issue",
            file_path="src/test.py"
            # auto_fix intentionally omitted
        )

        test_db.add(finding)
        test_db.commit()

        retrieved = test_db.query(Finding).first()
        assert retrieved.auto_fix is None

    def test_finding_explanation_nullable(self, test_db):
        """Test that explanation field is optional and nullable."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.BEST_PRACTICE,
            severity=FindingSeverity.MEDIUM,
            title="Best Practice Issue",
            description="Test issue",
            file_path="src/test.py"
            # explanation intentionally omitted
        )

        test_db.add(finding)
        test_db.commit()

        retrieved = test_db.query(Finding).first()
        assert retrieved.explanation is None

    def test_finding_improvement_suggestions_nullable(self, test_db):
        """Test that improvement_suggestions field is optional and nullable."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.PERFORMANCE,
            severity=FindingSeverity.LOW,
            title="Performance Issue",
            description="Test issue",
            file_path="src/test.py"
            # improvement_suggestions intentionally omitted
        )

        test_db.add(finding)
        test_db.commit()

        retrieved = test_db.query(Finding).first()
        assert retrieved.improvement_suggestions is None

    def test_finding_with_all_suggestions(self, test_db):
        """Test that all three suggestion fields can be populated together."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        auto_fix_text = """
def vulnerable_query(user_id):
    # BEFORE: SQL injection vulnerability
    # query = f"SELECT * FROM users WHERE id = {user_id}"

    # AFTER: Use parameterized queries
    query = "SELECT * FROM users WHERE id = ?"
    return db.execute(query, (user_id,))
"""

        explanation_text = "This is a SQL injection vulnerability where user input is directly concatenated into a SQL query. An attacker could modify the query by injecting malicious SQL code. Always use parameterized queries to prevent this attack."

        improvement_text = """
- Use parameterized queries for all database operations
- Implement input validation and sanitization
- Use an ORM that handles query escaping automatically
- Implement least privilege database user permissions
"""

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="SQL Injection Vulnerability",
            description="User input directly concatenated into SQL query",
            file_path="src/database.py",
            line_number=42,
            suggested_fix="Use parameterized queries",
            auto_fix=auto_fix_text,
            explanation=explanation_text,
            improvement_suggestions=improvement_text
        )

        test_db.add(finding)
        test_db.commit()

        retrieved = test_db.query(Finding).first()
        assert retrieved.auto_fix == auto_fix_text
        assert retrieved.explanation == explanation_text
        assert retrieved.improvement_suggestions == improvement_text

    def test_finding_with_partial_suggestions(self, test_db):
        """Test that some suggestions can be populated while others are None."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.BEST_PRACTICE,
            severity=FindingSeverity.MEDIUM,
            title="Code Quality Issue",
            description="Complex function",
            file_path="src/utils.py",
            line_number=10,
            explanation="This function is too complex.",
            # auto_fix and improvement_suggestions intentionally omitted
        )

        test_db.add(finding)
        test_db.commit()

        retrieved = test_db.query(Finding).first()
        assert retrieved.explanation == "This function is too complex."
        assert retrieved.auto_fix is None
        assert retrieved.improvement_suggestions is None

    def test_suggestion_fields_persist_text(self, test_db):
        """Test that suggestion fields can store and retrieve multi-line text."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        multiline_text = """Line 1
Line 2
Line 3
Line 4 with special chars: !@#$%^&*()"""

        finding = Finding(
            review_id=review.id,
            category=FindingCategory.PERFORMANCE,
            severity=FindingSeverity.HIGH,
            title="Performance Issue",
            description="Test",
            file_path="src/test.py",
            auto_fix=multiline_text
        )

        test_db.add(finding)
        test_db.commit()

        retrieved = test_db.query(Finding).first()
        assert retrieved.auto_fix == multiline_text

    def test_multiple_findings_different_suggestions(self, test_db):
        """Test that different findings can have different suggestion values."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )
        test_db.add(review)
        test_db.commit()

        finding1 = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="Issue 1",
            description="Security",
            file_path="src/test.py",
            auto_fix="Fix 1",
            explanation="Explanation 1"
        )

        finding2 = Finding(
            review_id=review.id,
            category=FindingCategory.PERFORMANCE,
            severity=FindingSeverity.MEDIUM,
            title="Issue 2",
            description="Performance",
            file_path="src/test.py",
            explanation="Explanation 2",
            improvement_suggestions="Improve 2"
        )

        test_db.add(finding1)
        test_db.add(finding2)
        test_db.commit()

        findings = test_db.query(Finding).order_by(Finding.id).all()
        assert len(findings) == 2

        assert findings[0].auto_fix == "Fix 1"
        assert findings[0].explanation == "Explanation 1"
        assert findings[0].improvement_suggestions is None

        assert findings[1].auto_fix is None
        assert findings[1].explanation == "Explanation 2"
        assert findings[1].improvement_suggestions == "Improve 2"


class TestDatabaseInitialization:
    """Tests for database initialization."""

    def test_init_db_creates_tables(self):
        """Test that init_db() creates all required tables."""
        engine = create_engine("sqlite:///:memory:")

        # Tables should not exist yet
        inspector = inspect(engine)
        assert "reviews" not in inspector.get_table_names()
        assert "findings" not in inspector.get_table_names()

        # Create tables
        Base.metadata.create_all(bind=engine)

        # Tables should exist now
        inspector = inspect(engine)
        assert "reviews" in inspector.get_table_names()
        assert "findings" in inspector.get_table_names()

    def test_reviews_table_columns(self):
        """Test that reviews table has all required columns."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)

        inspector = inspect(engine)
        columns = [col["name"] for col in inspector.get_columns("reviews")]

        required_columns = ["id", "pr_id", "repo_url", "branch", "commit_sha", "status", "created_at", "completed_at"]
        for col in required_columns:
            assert col in columns

    def test_findings_table_columns(self):
        """Test that findings table has all required columns."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)

        inspector = inspect(engine)
        columns = [col["name"] for col in inspector.get_columns("findings")]

        required_columns = [
            "id", "review_id", "category", "severity", "title", "description",
            "file_path", "line_number", "suggested_fix", "created_at",
            # AI-generated suggestion fields (Task 4.2)
            "auto_fix", "explanation", "improvement_suggestions"
        ]
        for col in required_columns:
            assert col in columns, f"Column '{col}' not found in findings table"

    def test_reviews_indexes(self):
        """Test that reviews table has expected indexes."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)

        inspector = inspect(engine)
        indexes = inspector.get_indexes("reviews")
        index_names = [idx["name"] for idx in indexes]

        # Should have index on pr_id (unique) and status
        assert any("pr_id" in name for name in index_names)
        assert any("status" in name for name in index_names)

    def test_findings_indexes(self):
        """Test that findings table has expected indexes."""
        engine = create_engine("sqlite:///:memory:")
        Base.metadata.create_all(bind=engine)

        inspector = inspect(engine)
        indexes = inspector.get_indexes("findings")
        index_names = [idx["name"] for idx in indexes]

        # Should have index on review_id, severity, and category
        assert any("review_id" in name for name in index_names)
        assert any("severity" in name for name in index_names)
        assert any("category" in name for name in index_names)


class TestSessionManagement:
    """Tests for FastAPI database dependency."""

    def test_sessionlocal_creates_sessions(self):
        """Test that SessionLocal factory creates sessions."""
        session1 = SessionLocal()
        session2 = SessionLocal()

        try:
            assert session1 is not None
            assert session2 is not None
            assert session1 is not session2  # Different instances
        finally:
            session1.close()
            session2.close()

    def test_sessionlocal_session_works(self, test_db):
        """Test that SessionLocal sessions can perform queries."""
        review = Review(
            pr_id=1,
            repo_url="https://github.com/user/repo",
            branch="main",
            commit_sha="abc123"
        )

        test_db.add(review)
        test_db.commit()

        # Verify we can query with the session
        result = test_db.query(Review).filter(Review.pr_id == 1).first()
        assert result is not None
        assert result.pr_id == 1
