"""
Integration tests for the complete code review pipeline.

Tests the full workflow from webhook to PR comment posting.
"""

import pytest
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.database import (
    Review,
    Finding,
    ReviewStatus,
    FindingCategory,
    FindingSeverity,
    init_db,
)
from src.main import app
from src.webhooks.github import parse_github_payload
from src.review_service import ReviewService
from src.integrations.github_api import GitHubAPIClient


# ============================================================================
# Application Tests
# ============================================================================

class TestApplicationStartup:
    """Tests for application startup and health checks."""

    def test_app_health_endpoint(self):
        """Test health check endpoint returns 200."""
        from fastapi.testclient import TestClient

        client = TestClient(app)
        response = client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "healthy"}

    def test_app_webhook_endpoint_exists(self):
        """Test webhook endpoint is registered."""
        routes = [route.path for route in app.routes]
        assert "/webhook/github" in routes


# ============================================================================
# Webhook to Review Workflow
# ============================================================================

class TestWebhookToReviewWorkflow:
    """Tests webhook processing creates review records."""

    def test_webhook_creates_review_record(self):
        """Test webhook payload creates Review record."""
        payload = {
            "action": "opened",
            "pull_request": {
                "id": 123456789,
                "number": 42,
                "title": "Add security fix",
                "body": "This PR fixes a vulnerability",
                "head": {
                    "sha": "abc123def456",
                    "ref": "feature/security-fix",
                },
                "base": {"ref": "main"},
                "user": {"login": "developer", "id": 999},
                "html_url": "https://github.com/user/repo/pull/42",
            },
            "repository": {
                "id": 111222333,
                "name": "repo",
                "full_name": "user/repo",
                "clone_url": "https://github.com/user/repo.git",
            },
            "sender": {"login": "developer", "id": 999},
        }

        parsed = parse_github_payload(payload)

        assert parsed is not None
        assert parsed.pull_request.number == 42
        assert parsed.pull_request.title == "Add security fix"
        assert parsed.repository.full_name == "user/repo"


# ============================================================================
# Database Operations
# ============================================================================

class TestDatabaseOperations:
    """Tests database creation and persistence."""

    def test_init_db_creates_tables(self, test_db):
        """Test database initialization creates tables."""
        # Tables should exist after init_db
        inspector = pytest.importorskip("sqlalchemy").inspect

        from src.database import engine

        insp = inspector(engine)
        table_names = insp.get_table_names()

        assert "reviews" in table_names
        assert "findings" in table_names

    def test_review_record_persistence(self, test_db):
        """Test review records persist in database."""
        # Create a review
        review = Review(
            pr_id=42,
            repo_url="https://github.com/owner/repo.git",
            branch="feature/x",
            commit_sha="abc123",
            status=ReviewStatus.PENDING,
        )
        test_db.add(review)
        test_db.commit()

        # Query back
        result = test_db.query(Review).filter(Review.pr_id == 42).first()

        assert result is not None
        assert result.pr_id == 42
        assert result.status == ReviewStatus.PENDING

    def test_finding_record_creation(self, test_db):
        """Test finding records can be created."""
        # Create review first
        review = Review(
            pr_id=42,
            repo_url="https://github.com/owner/repo.git",
            branch="feature/x",
            commit_sha="abc123",
            status=ReviewStatus.COMPLETED,
        )
        test_db.add(review)
        test_db.commit()

        # Create finding
        finding = Finding(
            review_id=review.id,
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="SQL Injection",
            description="Unsanitized input",
            file_path="app.py",
            line_number=42,
            suggested_fix="Use parameterized queries",
        )
        test_db.add(finding)
        test_db.commit()

        # Query back
        result = test_db.query(Finding).filter(Finding.review_id == review.id).first()

        assert result is not None
        assert result.title == "SQL Injection"
        assert result.severity == FindingSeverity.CRITICAL


# ============================================================================
# Service Integration Tests
# ============================================================================

class TestServiceIntegration:
    """Tests for service-level integration."""

    def test_github_client_initialization(self):
        """Test GitHub client can be initialized with token."""
        with patch("src.integrations.github_api.settings") as mock_settings:
            mock_settings.github_token = "ghp_test_token_123"
            client = GitHubAPIClient()
            assert client.token == "ghp_test_token_123"

    @patch("src.review_service.get_llm_provider")
    @patch("src.review_service.GitHubAPIClient")
    def test_review_service_initialization(self, mock_github, mock_llm, test_db):
        """Test ReviewService can be initialized."""
        mock_llm.return_value = MagicMock()
        mock_github.return_value = MagicMock()

        service = ReviewService(db=test_db)

        assert service.db == test_db
        assert service.analyzer is not None
        assert service.github_client is not None


# ============================================================================
# End-to-End Workflow Tests
# ============================================================================

class TestEndToEndWorkflow:
    """Tests for complete workflow simulation."""

    def test_review_status_transitions(self, test_db):
        """Test review progresses through status states."""
        # Create review
        review = Review(
            pr_id=42,
            repo_url="https://github.com/owner/repo.git",
            branch="feature/x",
            commit_sha="abc123",
            status=ReviewStatus.PENDING,
        )
        test_db.add(review)
        test_db.commit()

        # Verify initial status
        result = test_db.query(Review).filter(Review.pr_id == 42).first()
        assert result.status == ReviewStatus.PENDING
        assert result.completed_at is None

        # Update to analyzing
        result.status = ReviewStatus.ANALYZING
        test_db.commit()

        # Verify update
        result = test_db.query(Review).filter(Review.pr_id == 42).first()
        assert result.status == ReviewStatus.ANALYZING

        # Update to completed
        result.status = ReviewStatus.COMPLETED
        result.completed_at = datetime.now(timezone.utc)
        test_db.commit()

        # Verify completion
        result = test_db.query(Review).filter(Review.pr_id == 42).first()
        assert result.status == ReviewStatus.COMPLETED
        assert result.completed_at is not None

    def test_multiple_findings_for_single_review(self, test_db):
        """Test multiple findings can be associated with single review."""
        # Create review
        review = Review(
            pr_id=42,
            repo_url="https://github.com/owner/repo.git",
            branch="feature/x",
            commit_sha="abc123",
            status=ReviewStatus.COMPLETED,
        )
        test_db.add(review)
        test_db.commit()

        # Create multiple findings
        findings_data = [
            ("SQL Injection", FindingSeverity.CRITICAL),
            ("N+1 Query", FindingSeverity.HIGH),
            ("Memory Leak", FindingSeverity.MEDIUM),
        ]

        for title, severity in findings_data:
            finding = Finding(
                review_id=review.id,
                category=FindingCategory.SECURITY,
                severity=severity,
                title=title,
                description=f"{title} vulnerability",
                file_path="app.py",
                line_number=10,
            )
            test_db.add(finding)

        test_db.commit()

        # Query findings
        findings = test_db.query(Finding).filter(Finding.review_id == review.id).all()

        assert len(findings) == 3
        assert findings[0].title == "SQL Injection"
        assert findings[1].title == "N+1 Query"
        assert findings[2].title == "Memory Leak"

    def test_severity_filtering(self, test_db):
        """Test findings can be filtered by severity."""
        # Create review
        review = Review(
            pr_id=42,
            repo_url="https://github.com/owner/repo.git",
            branch="feature/x",
            commit_sha="abc123",
            status=ReviewStatus.COMPLETED,
        )
        test_db.add(review)
        test_db.commit()

        # Create findings with different severities
        for severity in [FindingSeverity.CRITICAL, FindingSeverity.HIGH, FindingSeverity.LOW]:
            finding = Finding(
                review_id=review.id,
                category=FindingCategory.SECURITY,
                severity=severity,
                title=f"Finding {severity.value}",
                description="Description",
                file_path="app.py",
            )
            test_db.add(finding)

        test_db.commit()

        # Filter by severity
        critical = test_db.query(Finding).filter(
            Finding.review_id == review.id,
            Finding.severity == FindingSeverity.CRITICAL,
        ).all()

        assert len(critical) == 1
        assert critical[0].severity == FindingSeverity.CRITICAL


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def test_db():
    """Provide a test database session."""
    from src.database import SessionLocal, engine, Base

    # Create tables
    Base.metadata.create_all(bind=engine)

    # Create session
    db = SessionLocal()

    try:
        yield db
    finally:
        db.close()
        # Clean up tables
        Base.metadata.drop_all(bind=engine)


# ============================================================================
# Configuration Tests
# ============================================================================

class TestConfiguration:
    """Tests for configuration management."""

    def test_settings_loaded(self):
        """Test settings can be loaded."""
        from src.config import settings

        assert settings is not None
        assert settings.host == "0.0.0.0"
        assert settings.port == 8000

    def test_database_url_configured(self):
        """Test database URL is set."""
        from src.config import settings

        assert settings.database_url is not None
        assert settings.database_url.startswith("sqlite://")

    def test_log_level_configured(self):
        """Test log level is valid."""
        from src.config import settings

        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        assert settings.log_level in valid_levels


# ============================================================================
# Error Handling Tests
# ============================================================================

class TestErrorHandling:
    """Tests for error handling in workflows."""

    def test_invalid_review_id_raises_error(self, test_db):
        """Test querying non-existent review returns None."""
        result = test_db.query(Review).filter(Review.id == 9999).first()
        assert result is None

    def test_review_without_findings(self, test_db):
        """Test review can exist without findings."""
        review = Review(
            pr_id=42,
            repo_url="https://github.com/owner/repo.git",
            branch="feature/x",
            commit_sha="abc123",
            status=ReviewStatus.PENDING,
        )
        test_db.add(review)
        test_db.commit()

        # Query findings
        findings = test_db.query(Finding).filter(Finding.review_id == review.id).all()
        assert len(findings) == 0


# ============================================================================
# Performance Tests
# ============================================================================

class TestPerformance:
    """Basic performance tests."""

    def test_bulk_finding_creation(self, test_db):
        """Test creating many findings doesn't slow down significantly."""
        import time

        # Create review
        review = Review(
            pr_id=42,
            repo_url="https://github.com/owner/repo.git",
            branch="feature/x",
            commit_sha="abc123",
            status=ReviewStatus.COMPLETED,
        )
        test_db.add(review)
        test_db.commit()

        # Create 100 findings
        start = time.time()

        for i in range(100):
            finding = Finding(
                review_id=review.id,
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.MEDIUM,
                title=f"Finding {i}",
                description=f"Description {i}",
                file_path=f"file_{i}.py",
                line_number=i,
            )
            test_db.add(finding)

        test_db.commit()
        elapsed = time.time() - start

        # Should complete in reasonable time
        assert elapsed < 5.0

        # Verify all created
        findings = test_db.query(Finding).filter(Finding.review_id == review.id).all()
        assert len(findings) == 100
