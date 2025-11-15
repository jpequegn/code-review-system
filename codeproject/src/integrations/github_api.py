"""
GitHub API Integration

Provides GitHub API client for posting findings as PR comments.
Handles authentication, comment formatting, and error handling.
"""

import logging
from typing import Optional, Dict, Any
from urllib.parse import urlparse

import requests

from src.config import settings
from src.analysis.analyzer import AnalyzedFinding
from src.database import FindingSeverity

logger = logging.getLogger(__name__)


# ============================================================================
# Severity Indicators
# ============================================================================

SEVERITY_EMOJI = {
    FindingSeverity.CRITICAL: "🔴",
    FindingSeverity.HIGH: "🟠",
    FindingSeverity.MEDIUM: "🟡",
    FindingSeverity.LOW: "🔵",
}

CATEGORY_EMOJI = {
    "security": "🛡️",
    "performance": "⚡",
    "best_practice": "✨",
}


# ============================================================================
# Custom Exceptions
# ============================================================================


class GitHubAPIError(Exception):
    """Raised when GitHub API request fails."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[str] = None,
    ):
        self.message = message
        self.status_code = status_code
        self.response = response
        super().__init__(self.message)


class RepositoryNotFoundError(GitHubAPIError):
    """Raised when repository cannot be found."""

    pass


class RateLimitError(GitHubAPIError):
    """Raised when GitHub API rate limit is exceeded."""

    pass


# ============================================================================
# Repository URL Parser
# ============================================================================


class RepositoryInfo:
    """Parsed GitHub repository information."""

    def __init__(self, owner: str, repo: str, pr_number: int):
        self.owner = owner
        self.repo = repo
        self.pr_number = pr_number

    def __repr__(self) -> str:
        return f"RepositoryInfo({self.owner}/{self.repo} PR#{self.pr_number})"


def parse_repository_url(repo_url: str, pr_number: int) -> RepositoryInfo:
    """
    Parse GitHub repository URL and extract owner/repo.

    Args:
        repo_url: Repository URL (https://github.com/owner/repo.git or similar)
        pr_number: Pull request number

    Returns:
        RepositoryInfo with owner, repo, and PR number

    Raises:
        ValueError: If URL format is invalid
    """
    if not repo_url:
        raise ValueError("Repository URL cannot be empty")

    # Remove .git suffix if present
    url = repo_url.rstrip("/")
    if url.endswith(".git"):
        url = url[:-4]

    # Handle SSH URLs (git@github.com:owner/repo)
    if url.startswith("git@") and ":" in url:
        # Extract the part after the colon
        ssh_path = url.split(":", 1)[1]
        parts = ssh_path.split("/")
        if len(parts) >= 2:
            owner = parts[-2]
            repo = parts[-1]
            if owner and repo:
                return RepositoryInfo(owner=owner, repo=repo, pr_number=pr_number)

    # Parse URL
    parsed = urlparse(url)

    # Extract from path (works for both https and git+ssh URLs)
    path = parsed.path.strip("/")
    if not path:
        raise ValueError(f"Invalid repository URL: {repo_url}")

    parts = path.split("/")
    if len(parts) < 2:
        raise ValueError(f"Invalid repository URL format: {repo_url}")

    owner = parts[-2]
    repo = parts[-1]

    if not owner or not repo:
        raise ValueError(f"Could not extract owner/repo from URL: {repo_url}")

    return RepositoryInfo(owner=owner, repo=repo, pr_number=pr_number)


# ============================================================================
# GitHub API Client
# ============================================================================


class GitHubAPIClient:
    """
    Client for GitHub API v3.

    Provides methods to post PR comments with findings.
    """

    BASE_URL = "https://api.github.com"

    def __init__(self, github_token: Optional[str] = None):
        """
        Initialize GitHub API client.

        Args:
            github_token: GitHub personal access token (uses config if not provided)

        Raises:
            ValueError: If no GitHub token is configured
        """
        self.token = github_token or settings.github_token
        if not self.token:
            raise ValueError(
                "GitHub token not configured. Set GITHUB_TOKEN environment variable "
                "or pass github_token parameter."
            )

    def _make_request(
        self,
        method: str,
        endpoint: str,
        data: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Make authenticated HTTP request to GitHub API.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint path (e.g., "/repos/owner/repo/issues/1/comments")
            data: Request body data (for POST/PATCH)
            **kwargs: Additional requests parameters

        Returns:
            JSON response from API

        Raises:
            GitHubAPIError: If request fails
            RateLimitError: If rate limit exceeded
            RepositoryNotFoundError: If repository not found
        """
        url = f"{self.BASE_URL}{endpoint}"

        headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github.v3+json",
            "User-Agent": "Code-Review-System/1.0",
        }

        try:
            response = requests.request(
                method=method, url=url, json=data, headers=headers, timeout=10, **kwargs
            )

            # Handle rate limiting
            if response.status_code == 403:
                remaining = response.headers.get("X-RateLimit-Remaining", "unknown")
                reset_time = response.headers.get("X-RateLimit-Reset", "unknown")
                logger.warning(
                    f"GitHub API rate limit exceeded. Remaining: {remaining}, Reset: {reset_time}"
                )
                raise RateLimitError(
                    f"GitHub API rate limit exceeded. Remaining: {remaining}",
                    status_code=403,
                    response=response.text,
                )

            # Handle 404 (repository not found)
            if response.status_code == 404:
                logger.error(f"Repository not found: {endpoint}")
                raise RepositoryNotFoundError(
                    f"Repository or resource not found: {endpoint}",
                    status_code=404,
                    response=response.text,
                )

            # Handle other errors
            if response.status_code >= 400:
                logger.error(
                    f"GitHub API error: {response.status_code} {response.text}"
                )
                raise GitHubAPIError(
                    f"GitHub API request failed: {response.status_code}",
                    status_code=response.status_code,
                    response=response.text,
                )

            return response.json()

        except requests.Timeout:
            logger.error("GitHub API request timed out")
            raise GitHubAPIError("GitHub API request timed out")
        except requests.RequestException as e:
            logger.error(f"GitHub API request failed: {str(e)}")
            raise GitHubAPIError(f"GitHub API request failed: {str(e)}")

    def post_pr_comment(
        self,
        repo_url: str,
        pr_number: int,
        findings: list[AnalyzedFinding],
    ) -> Dict[str, Any]:
        """
        Post PR comment with findings summary.

        Args:
            repo_url: Repository URL
            pr_number: Pull request number
            findings: List of AnalyzedFinding objects

        Returns:
            API response with comment details

        Raises:
            GitHubAPIError: If posting comment fails
            ValueError: If URL format is invalid
        """
        if not findings:
            logger.info("No findings to post")
            return {}

        # Parse repository URL
        try:
            repo_info = parse_repository_url(repo_url, pr_number)
        except ValueError as e:
            logger.error(f"Failed to parse repository URL: {str(e)}")
            raise

        # Format comment
        comment_body = self._format_findings_comment(findings)

        # Post comment
        endpoint = (
            f"/repos/{repo_info.owner}/{repo_info.repo}/issues/{pr_number}/comments"
        )

        try:
            response = self._make_request(
                method="POST",
                endpoint=endpoint,
                data={"body": comment_body},
            )
            logger.info(
                f"Posted comment to {repo_info.owner}/{repo_info.repo}#{pr_number}: "
                f"{len(findings)} findings"
            )
            return response
        except GitHubAPIError as e:
            logger.error(
                f"Failed to post comment to {repo_info.owner}/{repo_info.repo}#{pr_number}: "
                f"{str(e)}"
            )
            raise

    def _format_findings_comment(self, findings: list[AnalyzedFinding]) -> str:
        """
        Format findings as markdown comment for GitHub PR.

        Args:
            findings: List of AnalyzedFinding objects

        Returns:
            Formatted markdown comment body
        """
        lines = []

        # Header
        lines.append("## 🔍 Code Review Analysis")
        lines.append("")
        lines.append(f"Found **{len(findings)}** issue(s) in this pull request.")
        lines.append("")

        # Findings by severity
        severity_groups = self._group_findings_by_severity(findings)

        for severity in [
            FindingSeverity.CRITICAL,
            FindingSeverity.HIGH,
            FindingSeverity.MEDIUM,
            FindingSeverity.LOW,
        ]:
            if severity not in severity_groups:
                continue

            group_findings = severity_groups[severity]
            emoji = SEVERITY_EMOJI.get(severity, "•")
            lines.append(f"### {emoji} {severity.value.upper()}")
            lines.append("")

            for finding in group_findings:
                lines.append(self._format_single_finding(finding))
                lines.append("")

        # Footer
        lines.append("---")
        lines.append("_Analysis powered by LLM Code Reviewer_")

        return "\n".join(lines)

    def _format_single_finding(self, finding: AnalyzedFinding) -> str:
        """
        Format a single finding as markdown.

        Args:
            finding: AnalyzedFinding object

        Returns:
            Formatted markdown string
        """
        lines = []

        # Title with category emoji
        category_emoji = CATEGORY_EMOJI.get(
            (
                finding.category.value
                if hasattr(finding.category, "value")
                else finding.category
            ),
            "•",
        )
        lines.append(f"#### {category_emoji} {finding.title}")
        lines.append("")

        # Description
        lines.append(f"**Description:** {finding.description}")
        lines.append("")

        # Location
        location_parts = [f"`{finding.file_path}`"]
        if finding.line_number:
            location_parts.append(f"line {finding.line_number}")
        lines.append(f"**Location:** {', '.join(location_parts)}")
        lines.append("")

        # Suggested fix (if provided)
        if finding.suggested_fix:
            lines.append("**Suggested Fix:**")
            lines.append("")
            lines.append("```python")
            lines.append(finding.suggested_fix)
            lines.append("```")
            lines.append("")

        # Confidence
        if finding.confidence:
            confidence_pct = int(finding.confidence * 100)
            lines.append(f"_Confidence: {confidence_pct}%_")

        return "\n".join(lines)

    def _group_findings_by_severity(
        self, findings: list[AnalyzedFinding]
    ) -> Dict[FindingSeverity, list[AnalyzedFinding]]:
        """
        Group findings by severity.

        Args:
            findings: List of AnalyzedFinding objects

        Returns:
            Dictionary mapping FindingSeverity to list of findings
        """
        groups: Dict[FindingSeverity, list[AnalyzedFinding]] = {}

        for finding in findings:
            severity = finding.severity
            if severity not in groups:
                groups[severity] = []
            groups[severity].append(finding)

        return groups
