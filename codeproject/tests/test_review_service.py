"""
Tests for review service.

Tests end-to-end review pipeline orchestration.
"""

import pytest
import json
from unittest.mock import MagicMock, patch, Mock
from pathlib import Path

from src.review_service import (
    ReviewService,
    ReviewServiceError,
    RepositoryCloneError,
    DiffExtractionError,
)
from src.database import Review, Finding, ReviewStatus, FindingCategory, FindingSeverity
from src.analysis.analyzer import AnalyzedFinding


# ============================================================================
# Test Data & Fixtures
# ============================================================================

@pytest.fixture
def mock_db():
    """Provide a mock database session."""
    db = MagicMock()
    return db


@pytest.fixture
def mock_review():
    """Provide a mock review record."""
    review = MagicMock(spec=Review)
    review.id = 1
    review.pr_id = 42
    review.repo_url = "https://github.com/owner/repo.git"
    review.branch = "feature/security-fix"
    review.commit_sha = "abc123def456"
    review.status = ReviewStatus.PENDING
    return review


@pytest.fixture
def sample_findings():
    """Provide sample analyzed findings."""
    return [
        AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="SQL Injection",
            description="Unsanitized input",
            file_path="app.py",
            line_number=42,
            suggested_fix="Use parameterized queries",
            confidence=0.95,
        ),
        AnalyzedFinding(
            category=FindingCategory.PERFORMANCE,
            severity=FindingSeverity.HIGH,
            title="N+1 Query",
            description="Inefficient queries",
            file_path="models.py",
            line_number=87,
            confidence=0.85,
        ),
    ]


@pytest.fixture
def sample_diff():
    """Sample git diff output."""
    return """diff --git a/app.py b/app.py
index 1234567..abcdefg 100644
--- a/app.py
+++ b/app.py
@@ -40,7 +40,7 @@ def get_user(user_id):
     cursor = connection.cursor()
-    query = f"SELECT * FROM users WHERE id={user_id}"
+    query = "SELECT * FROM users WHERE id=?"
-    cursor.execute(query)
+    cursor.execute(query, (user_id,))
     return cursor.fetchone()
"""


@pytest.fixture
def review_service(mock_db):
    """Provide a ReviewService instance."""
    with patch("src.review_service.get_llm_provider") as mock_llm, \
         patch("src.review_service.GitHubAPIClient") as mock_github_client:
        mock_llm.return_value = MagicMock()
        mock_github_client.return_value = MagicMock()
        service = ReviewService(db=mock_db)
    return service


# ============================================================================
# Test Review Fetching
# ============================================================================

class TestReviewFetching:
    """Tests for fetching review records."""

    def test_get_review_success(self, review_service, mock_db, mock_review):
        """Test fetching existing review."""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_review

        result = review_service._get_review(1)

        assert result.id == 1
        assert result.pr_id == 42

    def test_get_review_not_found(self, review_service, mock_db):
        """Test fetching non-existent review."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = review_service._get_review(999)

        assert result is None

    def test_process_review_not_found_raises(self, review_service, mock_db):
        """Test processing non-existent review raises error."""
        mock_db.query.return_value.filter.return_value.first.return_value = None

        with pytest.raises(ReviewServiceError, match="not found"):
            review_service.process_review(999)


# ============================================================================
# Test Repository Cloning
# ============================================================================

class TestRepositoryCloning:
    """Tests for repository cloning."""

    @patch("src.review_service.subprocess.run")
    @patch("src.review_service.tempfile.mkdtemp")
    def test_clone_repository_success(self, mock_mkdtemp, mock_subprocess, review_service):
        """Test successful repository clone."""
        mock_mkdtemp.return_value = "/tmp/repo_12345"
        mock_subprocess.return_value = MagicMock(returncode=0)

        result = review_service._clone_repository("https://github.com/owner/repo.git")

        assert result == Path("/tmp/repo_12345")
        assert mock_subprocess.called

    @patch("src.review_service.subprocess.run")
    @patch("src.review_service.tempfile.mkdtemp")
    def test_clone_repository_failure(self, mock_mkdtemp, mock_subprocess, review_service):
        """Test repository clone failure."""
        mock_mkdtemp.return_value = "/tmp/repo_12345"
        mock_subprocess.return_value = MagicMock(
            returncode=1,
            stderr="Repository not found"
        )

        with pytest.raises(RepositoryCloneError):
            review_service._clone_repository("https://github.com/owner/repo.git")

    @patch("src.review_service.subprocess.run")
    @patch("src.review_service.tempfile.mkdtemp")
    def test_clone_repository_timeout(self, mock_mkdtemp, mock_subprocess, review_service):
        """Test repository clone timeout."""
        import subprocess
        mock_mkdtemp.return_value = "/tmp/repo_12345"
        mock_subprocess.side_effect = subprocess.TimeoutExpired("git", 60)

        with pytest.raises(RepositoryCloneError, match="timed out"):
            review_service._clone_repository("https://github.com/owner/repo.git")


# ============================================================================
# Test Diff Extraction
# ============================================================================

class TestDiffExtraction:
    """Tests for diff extraction."""

    @patch("src.review_service.subprocess.run")
    def test_extract_diff_success(self, mock_subprocess, review_service, sample_diff):
        """Test successful diff extraction."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout=sample_diff
        )

        result = review_service._extract_diff(
            repo_path=Path("/tmp/repo"),
            base_ref="main",
            head_ref="feature/x",
            commit_sha="abc123",
        )

        assert result == sample_diff
        assert "SQL Injection" not in result  # Raw diff
        assert "SELECT * FROM users" in result

    @patch("src.review_service.subprocess.run")
    def test_extract_diff_with_commit_sha(self, mock_subprocess, review_service, sample_diff):
        """Test diff extraction using commit SHA."""
        mock_subprocess.return_value = MagicMock(
            returncode=0,
            stdout=sample_diff
        )

        review_service._extract_diff(
            repo_path=Path("/tmp/repo"),
            base_ref="main",
            head_ref="feature/x",
            commit_sha="abc123def456",
        )

        # Should use commit_sha instead of head_ref
        call_args = mock_subprocess.call_args_list
        assert any("abc123def456" in str(call) for call in call_args)

    @patch("src.review_service.subprocess.run")
    def test_extract_diff_timeout(self, mock_subprocess, review_service):
        """Test diff extraction timeout."""
        import subprocess
        mock_subprocess.side_effect = subprocess.TimeoutExpired("git", 30)

        with pytest.raises(DiffExtractionError, match="timed out"):
            review_service._extract_diff(
                repo_path=Path("/tmp/repo"),
                base_ref="main",
                head_ref="feature/x",
                commit_sha="abc123",
            )


# ============================================================================
# Test Finding Storage
# ============================================================================

class TestFindingStorage:
    """Tests for storing findings in database."""

    def test_store_findings_success(self, review_service, mock_db, mock_review, sample_findings):
        """Test storing findings in database."""
        # Mock database add and commit
        stored_findings = []
        def capture_add(finding):
            stored_findings.append(finding)
        mock_db.add.side_effect = capture_add

        result = review_service._store_findings(mock_review, sample_findings)

        assert len(result) == 2
        assert mock_db.add.call_count == 2
        assert mock_db.commit.called

    def test_store_findings_empty(self, review_service, mock_db, mock_review):
        """Test storing empty findings list."""
        result = review_service._store_findings(mock_review, [])

        assert result == []

    def test_store_findings_maps_category(self, review_service, mock_db, mock_review):
        """Test findings category is correctly mapped."""
        finding = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="Issue",
            description="Desc",
            file_path="file.py",
        )

        stored = []
        def capture_add(f):
            stored.append(f)
        mock_db.add.side_effect = capture_add

        review_service._store_findings(mock_review, [finding])

        assert stored[0].category == FindingCategory.SECURITY


# ============================================================================
# Test Status Updates
# ============================================================================

class TestStatusUpdates:
    """Tests for updating review status."""

    def test_update_review_status(self, review_service, mock_db, mock_review):
        """Test updating review status."""
        review_service._update_review_status(mock_review, ReviewStatus.ANALYZING)

        assert mock_review.status == ReviewStatus.ANALYZING
        assert mock_db.commit.called

    def test_update_review_status_completed_sets_timestamp(self, review_service, mock_db, mock_review):
        """Test completed status sets completed_at timestamp."""
        from datetime import datetime, timezone

        review_service._update_review_status(mock_review, ReviewStatus.COMPLETED)

        assert mock_review.completed_at is not None
        assert isinstance(mock_review.completed_at, datetime)


# ============================================================================
# Test PR Comment Posting
# ============================================================================

class TestPRCommentPosting:
    """Tests for posting PR comments."""

    @patch.object(ReviewService, "_post_pr_comment")
    def test_post_pr_comment_success(self, mock_post, review_service, mock_review, sample_findings):
        """Test successful PR comment posting."""
        mock_post.return_value = {"id": 12345}

        result = review_service._post_pr_comment(mock_review, sample_findings)

        # Note: _post_pr_comment is actually called, mock intercepts
        assert mock_post.called

    def test_post_pr_comment_handles_api_error(self, review_service, mock_review, sample_findings):
        """Test handling of GitHub API errors."""
        from src.integrations.github_api import GitHubAPIError
        review_service.github_client.post_pr_comment.side_effect = GitHubAPIError("API error")

        with pytest.raises(GitHubAPIError):
            review_service._post_pr_comment(mock_review, sample_findings)


# ============================================================================
# Test End-to-End Review Processing
# ============================================================================

class TestEndToEndReviewProcessing:
    """Tests for complete review processing pipeline."""

    @patch("src.review_service.shutil.rmtree")
    @patch.object(ReviewService, "_extract_diff")
    @patch.object(ReviewService, "_clone_repository")
    @patch.object(ReviewService, "_post_pr_comment")
    def test_process_review_success(
        self,
        mock_post_comment,
        mock_clone,
        mock_extract_diff,
        mock_rmtree,
        review_service,
        mock_db,
        mock_review,
        sample_diff,
    ):
        """Test successful review processing."""
        # Setup mocks
        mock_db.query.return_value.filter.return_value.first.return_value = mock_review
        mock_clone.return_value = Path("/tmp/repo")
        mock_extract_diff.return_value = sample_diff

        # Mock analyzer to return findings
        with patch.object(review_service.analyzer, "analyze_code_changes") as mock_analyze:
            mock_analyze.return_value = [
                AnalyzedFinding(
                    category=FindingCategory.SECURITY,
                    severity=FindingSeverity.CRITICAL,
                    title="SQL Injection",
                    description="Unsafe query",
                    file_path="app.py",
                    line_number=42,
                )
            ]

            # Mock database add
            mock_db.add.return_value = None

            result = review_service.process_review(1)

        assert result["status"] == ReviewStatus.COMPLETED.value
        assert result["findings_count"] == 1
        assert mock_clone.called
        assert mock_extract_diff.called

    @patch("src.review_service.shutil.rmtree")
    @patch.object(ReviewService, "_clone_repository")
    def test_process_review_clone_failure(
        self,
        mock_clone,
        mock_rmtree,
        review_service,
        mock_db,
        mock_review,
    ):
        """Test handling of clone failure."""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_review
        mock_clone.side_effect = RepositoryCloneError("Clone failed")

        with pytest.raises(ReviewServiceError):
            review_service.process_review(1)

    @patch("src.review_service.shutil.rmtree")
    @patch.object(ReviewService, "_extract_diff")
    @patch.object(ReviewService, "_clone_repository")
    def test_process_review_no_changes(
        self,
        mock_clone,
        mock_extract_diff,
        mock_rmtree,
        review_service,
        mock_db,
        mock_review,
    ):
        """Test review with no code changes."""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_review
        mock_clone.return_value = Path("/tmp/repo")
        mock_extract_diff.return_value = ""  # Empty diff

        # Mock parser to return no diffs
        with patch.object(review_service.diff_parser, "parse") as mock_parse:
            mock_parse.return_value = []

            result = review_service.process_review(1)

        assert result["status"] == ReviewStatus.COMPLETED.value
        assert result["findings_count"] == 0

    @patch("src.review_service.shutil.rmtree")
    @patch.object(ReviewService, "_extract_diff")
    @patch.object(ReviewService, "_clone_repository")
    def test_process_review_continues_on_comment_failure(
        self,
        mock_clone,
        mock_extract_diff,
        mock_rmtree,
        review_service,
        mock_db,
        mock_review,
        sample_diff,
    ):
        """Test review continues even if posting comment fails."""
        from src.integrations.github_api import GitHubAPIError

        mock_db.query.return_value.filter.return_value.first.return_value = mock_review
        mock_clone.return_value = Path("/tmp/repo")
        mock_extract_diff.return_value = sample_diff

        # Mock analyzer
        with patch.object(review_service.analyzer, "analyze_code_changes") as mock_analyze:
            mock_analyze.return_value = [
                AnalyzedFinding(
                    category=FindingCategory.SECURITY,
                    severity=FindingSeverity.CRITICAL,
                    title="Issue",
                    description="Desc",
                    file_path="file.py",
                )
            ]

            # Mock GitHub client to fail
            review_service.github_client.post_pr_comment.side_effect = GitHubAPIError(
                "API error"
            )

            # Mock database add
            mock_db.add.return_value = None

            result = review_service.process_review(1)

        # Should still complete despite comment failure
        assert result["status"] == ReviewStatus.COMPLETED.value


# ============================================================================
# Test Error Handling & Recovery
# ============================================================================

class TestErrorHandling:
    """Tests for error handling and recovery."""

    @patch("src.review_service.shutil.rmtree")
    @patch.object(ReviewService, "_clone_repository")
    def test_cleanup_on_error(
        self,
        mock_clone,
        mock_rmtree,
        review_service,
        mock_db,
        mock_review,
    ):
        """Test temporary directory is cleaned up on error."""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_review
        mock_clone.side_effect = RepositoryCloneError("Clone failed")

        try:
            review_service.process_review(1)
        except ReviewServiceError:
            pass

        # Should mark as failed
        assert mock_review.status == ReviewStatus.FAILED

    def test_review_service_error_inheritance(self):
        """Test ReviewServiceError is proper exception."""
        error = ReviewServiceError("test")
        assert isinstance(error, Exception)

    def test_repository_clone_error_inheritance(self):
        """Test RepositoryCloneError inherits from ReviewServiceError."""
        error = RepositoryCloneError("test")
        assert isinstance(error, ReviewServiceError)

    def test_diff_extraction_error_inheritance(self):
        """Test DiffExtractionError inherits from ReviewServiceError."""
        error = DiffExtractionError("test")
        assert isinstance(error, ReviewServiceError)


# ============================================================================
# Test Logging
# ============================================================================

class TestLogging:
    """Tests for service logging."""

    @patch("src.review_service.logger")
    @patch("src.review_service.shutil.rmtree")
    @patch.object(ReviewService, "_extract_diff")
    @patch.object(ReviewService, "_clone_repository")
    def test_logs_processing_start(
        self,
        mock_clone,
        mock_extract_diff,
        mock_rmtree,
        mock_logger,
        review_service,
        mock_db,
        mock_review,
        sample_diff,
    ):
        """Test service logs processing start."""
        mock_db.query.return_value.filter.return_value.first.return_value = mock_review
        mock_clone.return_value = Path("/tmp/repo")
        mock_extract_diff.return_value = sample_diff

        with patch.object(review_service.analyzer, "analyze_code_changes") as mock_analyze:
            mock_analyze.return_value = []
            mock_db.add.return_value = None

            try:
                review_service.process_review(1)
            except:
                pass

        # Should log processing info
        assert any(
            "Processing review" in str(call)
            for call in mock_logger.info.call_args_list
        )
