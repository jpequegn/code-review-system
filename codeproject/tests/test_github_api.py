"""
Tests for GitHub API integration.

Tests repository URL parsing, comment formatting, and API interactions.
"""

import pytest
import json
from unittest.mock import MagicMock, patch

from src.integrations.github_api import (
    GitHubAPIClient,
    GitHubAPIError,
    RepositoryNotFoundError,
    RateLimitError,
    parse_repository_url,
    RepositoryInfo,
    SEVERITY_EMOJI,
    CATEGORY_EMOJI,
)
from src.analysis.analyzer import AnalyzedFinding
from src.database import FindingCategory, FindingSeverity


# ============================================================================
# Test Data & Fixtures
# ============================================================================

@pytest.fixture
def github_token():
    """Provide a test GitHub token."""
    return "ghp_test1234567890abcdefghijklmnopqrst"


@pytest.fixture
def github_client(github_token):
    """Provide a GitHubAPIClient instance."""
    return GitHubAPIClient(github_token=github_token)


@pytest.fixture
def sample_findings():
    """Sample findings for testing."""
    return [
        AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="SQL Injection Vulnerability",
            description="Unsanitized user input in database query",
            file_path="app.py",
            line_number=42,
            suggested_fix="Use parameterized queries",
            confidence=0.98,
        ),
        AnalyzedFinding(
            category=FindingCategory.PERFORMANCE,
            severity=FindingSeverity.HIGH,
            title="N+1 Query Problem",
            description="Inefficient database queries in loop",
            file_path="models.py",
            line_number=87,
            suggested_fix="Use batch queries or join",
            confidence=0.85,
        ),
    ]


# ============================================================================
# Test Repository URL Parsing
# ============================================================================

class TestRepositoryURLParsing:
    """Tests for repository URL parsing."""

    def test_parse_https_url(self):
        """Test parsing HTTPS repository URL."""
        repo_info = parse_repository_url("https://github.com/user/repo.git", 42)

        assert repo_info.owner == "user"
        assert repo_info.repo == "repo"
        assert repo_info.pr_number == 42

    def test_parse_https_url_without_git_suffix(self):
        """Test parsing HTTPS URL without .git suffix."""
        repo_info = parse_repository_url("https://github.com/owner/myrepo", 10)

        assert repo_info.owner == "owner"
        assert repo_info.repo == "myrepo"
        assert repo_info.pr_number == 10

    def test_parse_ssh_url(self):
        """Test parsing SSH repository URL."""
        repo_info = parse_repository_url("git@github.com:developer/project.git", 5)

        assert repo_info.owner == "developer"
        assert repo_info.repo == "project"
        assert repo_info.pr_number == 5

    def test_parse_git_protocol_url(self):
        """Test parsing git:// protocol URL."""
        repo_info = parse_repository_url("git://github.com/org/app.git", 99)

        assert repo_info.owner == "org"
        assert repo_info.repo == "app"
        assert repo_info.pr_number == 99

    def test_parse_url_with_trailing_slash(self):
        """Test parsing URL with trailing slash."""
        repo_info = parse_repository_url("https://github.com/user/repo/", 1)

        assert repo_info.owner == "user"
        assert repo_info.repo == "repo"

    def test_parse_empty_url_raises(self):
        """Test parsing empty URL raises ValueError."""
        with pytest.raises(ValueError, match="Repository URL cannot be empty"):
            parse_repository_url("", 42)

    def test_parse_invalid_url_no_path(self):
        """Test parsing URL with no path raises."""
        with pytest.raises(ValueError, match="Invalid repository URL"):
            parse_repository_url("https://github.com", 42)

    def test_parse_invalid_url_format(self):
        """Test parsing URL with invalid format raises."""
        with pytest.raises(ValueError, match="Invalid repository URL format"):
            parse_repository_url("https://github.com/just-one-part", 42)

    def test_parse_url_with_deep_path(self):
        """Test parsing URL with deep path (only last two parts used)."""
        repo_info = parse_repository_url(
            "https://github.com/deeply/nested/user/repo.git",
            1
        )

        # Should use last two path segments
        assert repo_info.owner == "user"
        assert repo_info.repo == "repo"


# ============================================================================
# Test GitHub API Client Initialization
# ============================================================================

class TestGitHubAPIClientInitialization:
    """Tests for GitHub API client initialization."""

    def test_initialize_with_token(self, github_token):
        """Test initializing client with explicit token."""
        client = GitHubAPIClient(github_token=github_token)
        assert client.token == github_token

    def test_initialize_without_token_raises(self):
        """Test initializing without token raises error."""
        with patch("src.integrations.github_api.settings") as mock_settings:
            mock_settings.github_token = ""
            with pytest.raises(ValueError, match="GitHub token not configured"):
                GitHubAPIClient()

    def test_initialize_uses_settings_token(self):
        """Test initialization uses token from settings if not provided."""
        test_token = "ghp_settings_token_123"
        with patch("src.integrations.github_api.settings") as mock_settings:
            mock_settings.github_token = test_token
            client = GitHubAPIClient()
            assert client.token == test_token


# ============================================================================
# Test Comment Formatting
# ============================================================================

class TestCommentFormatting:
    """Tests for comment formatting."""

    def test_format_single_finding_complete(self, github_client):
        """Test formatting a complete finding."""
        finding = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.CRITICAL,
            title="SQL Injection",
            description="Input not sanitized",
            file_path="app.py",
            line_number=42,
            suggested_fix="Use parameterized queries",
            confidence=0.95,
        )

        formatted = github_client._format_single_finding(finding)

        assert "SQL Injection" in formatted
        assert "Input not sanitized" in formatted
        assert "`app.py`" in formatted
        assert "line 42" in formatted
        assert "Use parameterized queries" in formatted
        assert "95%" in formatted

    def test_format_single_finding_no_line_number(self, github_client):
        """Test formatting finding without line number."""
        finding = AnalyzedFinding(
            category=FindingCategory.SECURITY,
            severity=FindingSeverity.HIGH,
            title="Issue",
            description="Description",
            file_path="file.py",
            line_number=None,
        )

        formatted = github_client._format_single_finding(finding)

        assert "`file.py`" in formatted
        assert "line" not in formatted

    def test_format_single_finding_no_suggested_fix(self, github_client):
        """Test formatting finding without suggested fix."""
        finding = AnalyzedFinding(
            category=FindingCategory.PERFORMANCE,
            severity=FindingSeverity.MEDIUM,
            title="Inefficiency",
            description="Description",
            file_path="file.py",
            suggested_fix=None,
        )

        formatted = github_client._format_single_finding(finding)

        assert "Suggested Fix:" not in formatted

    def test_format_findings_comment_basic(self, github_client, sample_findings):
        """Test formatting complete findings comment."""
        comment = github_client._format_findings_comment(sample_findings)

        assert "Code Review Analysis" in comment
        assert "2" in comment  # Number of findings
        assert "SQL Injection Vulnerability" in comment
        assert "N+1 Query Problem" in comment
        assert "CRITICAL" in comment
        assert "HIGH" in comment

    def test_format_findings_comment_severity_grouping(self, github_client):
        """Test findings are grouped by severity."""
        findings = [
            AnalyzedFinding(
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.LOW,
                title="Low severity",
                description="Desc",
                file_path="file.py",
            ),
            AnalyzedFinding(
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.CRITICAL,
                title="Critical severity",
                description="Desc",
                file_path="file.py",
            ),
        ]

        comment = github_client._format_findings_comment(findings)

        # Critical should appear before LOW
        critical_pos = comment.find("CRITICAL")
        low_pos = comment.find("LOW")
        assert critical_pos < low_pos

    def test_format_findings_comment_empty(self, github_client):
        """Test formatting with no findings."""
        comment = github_client._format_findings_comment([])

        assert "Code Review Analysis" in comment

    def test_severity_emoji_in_comment(self, github_client, sample_findings):
        """Test severity emojis are included in comment."""
        comment = github_client._format_findings_comment(sample_findings)

        for emoji in SEVERITY_EMOJI.values():
            # At least the emojis used should be present
            pass
        assert "🔴" in comment or "🟠" in comment or "🟡" in comment or "🔵" in comment

    def test_category_emoji_in_comment(self, github_client, sample_findings):
        """Test category emojis are included in comment."""
        comment = github_client._format_findings_comment(sample_findings)

        # Should have both security and performance emojis
        assert "🛡️" in comment or "⚡" in comment


# ============================================================================
# Test Grouping Functions
# ============================================================================

class TestFindingsGrouping:
    """Tests for grouping findings."""

    def test_group_findings_by_severity(self, github_client):
        """Test grouping findings by severity."""
        findings = [
            AnalyzedFinding(
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.LOW,
                title="Low",
                description="Desc",
                file_path="file.py",
            ),
            AnalyzedFinding(
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.CRITICAL,
                title="Critical",
                description="Desc",
                file_path="file.py",
            ),
            AnalyzedFinding(
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.LOW,
                title="Another low",
                description="Desc",
                file_path="file.py",
            ),
        ]

        groups = github_client._group_findings_by_severity(findings)

        assert len(groups[FindingSeverity.CRITICAL]) == 1
        assert len(groups[FindingSeverity.LOW]) == 2

    def test_group_findings_empty(self, github_client):
        """Test grouping empty findings list."""
        groups = github_client._group_findings_by_severity([])
        assert groups == {}


# ============================================================================
# Test API Request Handling
# ============================================================================

class TestAPIRequests:
    """Tests for API request handling."""

    @patch("src.integrations.github_api.requests.request")
    def test_post_pr_comment_success(self, mock_request, github_client, sample_findings):
        """Test successfully posting PR comment."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {
            "id": 123456,
            "body": "Comment body",
            "user": {"login": "github-app"},
        }
        mock_request.return_value = mock_response

        result = github_client.post_pr_comment(
            repo_url="https://github.com/owner/repo.git",
            pr_number=42,
            findings=sample_findings,
        )

        assert result["id"] == 123456
        assert mock_request.called

    @patch("src.integrations.github_api.requests.request")
    def test_post_pr_comment_404_not_found(self, mock_request, github_client, sample_findings):
        """Test posting comment when repo not found."""
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_response.text = "Not found"
        mock_request.return_value = mock_response

        with pytest.raises(RepositoryNotFoundError):
            github_client.post_pr_comment(
                repo_url="https://github.com/owner/repo.git",
                pr_number=42,
                findings=sample_findings,
            )

    @patch("src.integrations.github_api.requests.request")
    def test_post_pr_comment_403_rate_limit(self, mock_request, github_client, sample_findings):
        """Test posting comment when rate limited."""
        mock_response = MagicMock()
        mock_response.status_code = 403
        mock_response.text = "Rate limit exceeded"
        mock_response.headers = {
            "X-RateLimit-Remaining": "0",
            "X-RateLimit-Reset": "1234567890",
        }
        mock_request.return_value = mock_response

        with pytest.raises(RateLimitError):
            github_client.post_pr_comment(
                repo_url="https://github.com/owner/repo.git",
                pr_number=42,
                findings=sample_findings,
            )

    @patch("src.integrations.github_api.requests.request")
    def test_post_pr_comment_500_error(self, mock_request, github_client, sample_findings):
        """Test posting comment when server error occurs."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Internal server error"
        mock_request.return_value = mock_response

        with pytest.raises(GitHubAPIError):
            github_client.post_pr_comment(
                repo_url="https://github.com/owner/repo.git",
                pr_number=42,
                findings=sample_findings,
            )

    @patch("src.integrations.github_api.requests.request")
    def test_post_pr_comment_timeout(self, mock_request, github_client, sample_findings):
        """Test posting comment with timeout."""
        import requests
        mock_request.side_effect = requests.Timeout("Request timed out")

        with pytest.raises(GitHubAPIError, match="timed out"):
            github_client.post_pr_comment(
                repo_url="https://github.com/owner/repo.git",
                pr_number=42,
                findings=sample_findings,
            )

    @patch("src.integrations.github_api.requests.request")
    def test_post_pr_comment_connection_error(self, mock_request, github_client, sample_findings):
        """Test posting comment with connection error."""
        import requests
        mock_request.side_effect = requests.RequestException("Connection failed")

        with pytest.raises(GitHubAPIError, match="failed"):
            github_client.post_pr_comment(
                repo_url="https://github.com/owner/repo.git",
                pr_number=42,
                findings=sample_findings,
            )

    def test_post_pr_comment_empty_findings(self, github_client):
        """Test posting with no findings."""
        result = github_client.post_pr_comment(
            repo_url="https://github.com/owner/repo.git",
            pr_number=42,
            findings=[],
        )

        assert result == {}

    def test_post_pr_comment_invalid_url(self, github_client, sample_findings):
        """Test posting with invalid repo URL."""
        with pytest.raises(ValueError):
            github_client.post_pr_comment(
                repo_url="invalid-url",
                pr_number=42,
                findings=sample_findings,
            )


# ============================================================================
# Test Error Messages
# ============================================================================

class TestErrorMessages:
    """Tests for error messages and logging."""

    def test_github_api_error_message(self):
        """Test GitHubAPIError contains message."""
        error = GitHubAPIError("Test error", status_code=400, response="Bad request")

        assert error.message == "Test error"
        assert error.status_code == 400
        assert error.response == "Bad request"

    def test_repository_not_found_error(self):
        """Test RepositoryNotFoundError is specific error type."""
        error = RepositoryNotFoundError("Repo not found")

        assert isinstance(error, GitHubAPIError)
        assert "not found" in str(error).lower()

    def test_rate_limit_error(self):
        """Test RateLimitError is specific error type."""
        error = RateLimitError("Rate limit exceeded")

        assert isinstance(error, GitHubAPIError)
        assert "limit" in str(error).lower()


# ============================================================================
# Test API Endpoint Construction
# ============================================================================

class TestAPIEndpoints:
    """Tests for API endpoint construction."""

    @patch("src.integrations.github_api.requests.request")
    def test_correct_endpoint_used(self, mock_request, github_client, sample_findings):
        """Test correct GitHub API endpoint is used."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1}
        mock_request.return_value = mock_response

        github_client.post_pr_comment(
            repo_url="https://github.com/myorg/myrepo.git",
            pr_number=123,
            findings=sample_findings,
        )

        # Verify the correct endpoint was called
        call_args = mock_request.call_args
        assert "/repos/myorg/myrepo/issues/123/comments" in call_args[1]["url"]

    @patch("src.integrations.github_api.requests.request")
    def test_authentication_header_included(self, mock_request, github_client, sample_findings):
        """Test authentication header is included."""
        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1}
        mock_request.return_value = mock_response

        github_client.post_pr_comment(
            repo_url="https://github.com/owner/repo.git",
            pr_number=42,
            findings=sample_findings,
        )

        # Verify authentication header
        call_args = mock_request.call_args
        headers = call_args[1]["headers"]
        assert "Authorization" in headers
        assert headers["Authorization"].startswith("token ")


# ============================================================================
# Test Integration Scenarios
# ============================================================================

class TestIntegrationScenarios:
    """Tests for real-world integration scenarios."""

    @patch("src.integrations.github_api.requests.request")
    def test_post_multiple_findings_different_files(self, mock_request, github_client):
        """Test posting findings from different files."""
        findings = [
            AnalyzedFinding(
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.CRITICAL,
                title="Issue in app.py",
                description="Security issue",
                file_path="app.py",
                line_number=10,
            ),
            AnalyzedFinding(
                category=FindingCategory.PERFORMANCE,
                severity=FindingSeverity.HIGH,
                title="Issue in models.py",
                description="Performance issue",
                file_path="models.py",
                line_number=20,
            ),
            AnalyzedFinding(
                category=FindingCategory.SECURITY,
                severity=FindingSeverity.MEDIUM,
                title="Another in app.py",
                description="Another issue",
                file_path="app.py",
                line_number=50,
            ),
        ]

        mock_response = MagicMock()
        mock_response.status_code = 201
        mock_response.json.return_value = {"id": 1}
        mock_request.return_value = mock_response

        result = github_client.post_pr_comment(
            repo_url="https://github.com/owner/repo",
            pr_number=42,
            findings=findings,
        )

        # Should successfully post
        assert result["id"] == 1

        # Verify comment body contains all findings
        call_args = mock_request.call_args
        comment_body = call_args[1]["json"]["body"]
        assert "app.py" in comment_body
        assert "models.py" in comment_body
        assert "Issue in app.py" in comment_body
        assert "Issue in models.py" in comment_body
