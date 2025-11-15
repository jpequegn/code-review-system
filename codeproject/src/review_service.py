"""
Review Service - End-to-End Pipeline Orchestration

Coordinates the complete code review workflow:
1. Repository cloning
2. Diff extraction
3. Code analysis
4. Finding storage
5. PR commenting
6. Status tracking
"""

import logging
import shutil
import tempfile
import subprocess
from pathlib import Path
from typing import Optional, List
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.config import settings
from src.database import Review, Finding, ReviewStatus, FindingCategory, FindingSeverity
from src.analysis.diff_parser import DiffParser
from src.analysis.analyzer import CodeAnalyzer, AnalyzedFinding
from src.integrations.github_api import GitHubAPIClient, GitHubAPIError
from src.llm.provider import get_llm_provider

logger = logging.getLogger(__name__)


# ============================================================================
# Custom Exceptions
# ============================================================================

class ReviewServiceError(Exception):
    """Base exception for review service errors."""

    pass


class RepositoryCloneError(ReviewServiceError):
    """Raised when repository cloning fails."""

    pass


class DiffExtractionError(ReviewServiceError):
    """Raised when diff extraction fails."""

    pass


class AnalysisError(ReviewServiceError):
    """Raised when analysis fails."""

    pass


# ============================================================================
# Review Service
# ============================================================================

class ReviewService:
    """
    Orchestrates the complete code review pipeline.

    Responsibilities:
    - Clone repository
    - Extract diffs
    - Run analysis
    - Store findings
    - Post comments
    - Handle errors
    """

    def __init__(self, db: Session):
        """
        Initialize ReviewService.

        Args:
            db: Database session
        """
        self.db = db
        self.analyzer = CodeAnalyzer(llm_provider=get_llm_provider())
        self.github_client = GitHubAPIClient()
        self.diff_parser = DiffParser()

    def process_review(self, review_id: int) -> dict:
        """
        Process a code review end-to-end.

        Args:
            review_id: Review record ID

        Returns:
            Dictionary with review results and status

        Raises:
            ReviewServiceError: If processing fails
        """
        # Fetch review from database
        review = self._get_review(review_id)
        if not review:
            raise ReviewServiceError(f"Review {review_id} not found")

        logger.info(f"Processing review {review_id}: PR#{review.pr_id}")

        try:
            # Mark as analyzing
            self._update_review_status(review, ReviewStatus.ANALYZING)

            # Clone repository
            repo_path = self._clone_repository(review.repo_url)
            try:
                # Extract diff
                diff_text = self._extract_diff(
                    repo_path=repo_path,
                    base_ref="main",
                    head_ref=review.branch,
                    commit_sha=review.commit_sha,
                )

                # Parse diff
                file_diffs = self.diff_parser.parse(diff_text)
                if not file_diffs:
                    logger.info(f"No code changes found in review {review_id}")
                    self._update_review_status(review, ReviewStatus.COMPLETED)
                    return {
                        "review_id": review_id,
                        "status": ReviewStatus.COMPLETED.value,
                        "findings_count": 0,
                    }

                # Analyze code
                findings = self.analyzer.analyze_code_changes(file_diffs)

                # Store findings in database
                finding_records = self._store_findings(review, findings)

                # Post PR comment if findings exist
                if finding_records:
                    try:
                        self._post_pr_comment(review, findings)
                    except GitHubAPIError as e:
                        logger.warning(f"Failed to post PR comment: {str(e)}")
                        # Don't fail the review, but log the error

                # Mark as completed
                self._update_review_status(review, ReviewStatus.COMPLETED)

                logger.info(
                    f"Review {review_id} completed: {len(finding_records)} findings"
                )

                return {
                    "review_id": review_id,
                    "status": ReviewStatus.COMPLETED.value,
                    "findings_count": len(finding_records),
                    "findings": [
                        {
                            "id": f.id,
                            "severity": f.severity.value,
                            "title": f.title,
                            "file_path": f.file_path,
                        }
                        for f in finding_records
                    ],
                }

            finally:
                # Clean up temporary directory
                if repo_path.exists():
                    shutil.rmtree(repo_path, ignore_errors=True)
                    logger.debug(f"Cleaned up temporary directory: {repo_path}")

        except Exception as e:
            logger.error(f"Review {review_id} failed: {str(e)}", exc_info=True)
            self._update_review_status(review, ReviewStatus.FAILED)
            raise ReviewServiceError(f"Failed to process review {review_id}: {str(e)}")

    def _get_review(self, review_id: int) -> Optional[Review]:
        """
        Fetch review record from database.

        Args:
            review_id: Review record ID

        Returns:
            Review object or None if not found
        """
        return self.db.query(Review).filter(Review.id == review_id).first()

    def _update_review_status(
        self,
        review: Review,
        status: ReviewStatus,
    ) -> None:
        """
        Update review status in database.

        Args:
            review: Review object
            status: New status
        """
        review.status = status
        if status == ReviewStatus.COMPLETED:
            review.completed_at = datetime.now(timezone.utc)
        self.db.commit()
        logger.debug(f"Review {review.id} status updated to {status.value}")

    def _clone_repository(self, repo_url: str) -> Path:
        """
        Clone repository to temporary directory.

        Args:
            repo_url: Repository URL

        Returns:
            Path to cloned repository

        Raises:
            RepositoryCloneError: If clone fails
        """
        try:
            # Create temporary directory
            temp_dir = Path(tempfile.mkdtemp(prefix="code_review_"))
            logger.debug(f"Cloning repository to {temp_dir}")

            # Clone repository
            result = subprocess.run(
                [
                    "git",
                    "clone",
                    "--depth",
                    "1",
                    repo_url,
                    str(temp_dir),
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )

            if result.returncode != 0:
                raise RepositoryCloneError(
                    f"Failed to clone repository: {result.stderr}"
                )

            logger.info(f"Repository cloned: {repo_url}")
            return temp_dir

        except subprocess.TimeoutExpired:
            raise RepositoryCloneError("Repository clone timed out (>60s)")
        except Exception as e:
            raise RepositoryCloneError(f"Repository clone failed: {str(e)}")

    def _extract_diff(
        self,
        repo_path: Path,
        base_ref: str,
        head_ref: str,
        commit_sha: str,
    ) -> str:
        """
        Extract diff between branches.

        Args:
            repo_path: Path to repository
            base_ref: Base branch name (e.g., 'main')
            head_ref: Head branch name (e.g., 'feature/x')
            commit_sha: Specific commit SHA to use (if provided)

        Returns:
            Unified diff text

        Raises:
            DiffExtractionError: If diff extraction fails
        """
        try:
            # Use specific commit if provided, otherwise use branch
            ref = commit_sha if commit_sha else head_ref

            # Fetch all branches first
            subprocess.run(
                ["git", "fetch", "--all"],
                cwd=repo_path,
                capture_output=True,
                timeout=30,
                check=True,
            )

            # Generate diff
            result = subprocess.run(
                [
                    "git",
                    "diff",
                    f"origin/{base_ref}...{ref}",
                ],
                cwd=repo_path,
                capture_output=True,
                text=True,
                timeout=30,
            )

            if result.returncode != 0:
                # Try alternative diff format
                logger.warning(
                    f"Diff failed with origin/{base_ref}...{ref}, trying {ref}..HEAD"
                )
                result = subprocess.run(
                    [
                        "git",
                        "diff",
                        f"{base_ref}..{ref}",
                    ],
                    cwd=repo_path,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )

            logger.info(f"Diff extracted: {len(result.stdout)} bytes")
            return result.stdout

        except subprocess.TimeoutExpired:
            raise DiffExtractionError("Diff extraction timed out (>30s)")
        except Exception as e:
            raise DiffExtractionError(f"Diff extraction failed: {str(e)}")

    def _store_findings(
        self,
        review: Review,
        findings: List[AnalyzedFinding],
    ) -> List[Finding]:
        """
        Store analyzed findings in database.

        Args:
            review: Review object
            findings: List of AnalyzedFinding objects

        Returns:
            List of stored Finding records
        """
        stored_findings = []

        for finding in findings:
            # Map category
            category = FindingCategory[finding.category.name]

            # Map severity
            severity = finding.severity

            # Create Finding record
            db_finding = Finding(
                review_id=review.id,
                category=category,
                severity=severity,
                title=finding.title,
                description=finding.description,
                file_path=finding.file_path,
                line_number=finding.line_number,
                suggested_fix=finding.suggested_fix,
            )

            self.db.add(db_finding)
            stored_findings.append(db_finding)

        self.db.commit()
        logger.info(f"Stored {len(stored_findings)} findings for review {review.id}")

        return stored_findings

    def _post_pr_comment(
        self,
        review: Review,
        findings: List[AnalyzedFinding],
    ) -> dict:
        """
        Post PR comment with findings.

        Args:
            review: Review object
            findings: List of AnalyzedFinding objects

        Returns:
            API response from GitHub

        Raises:
            GitHubAPIError: If posting fails
        """
        try:
            response = self.github_client.post_pr_comment(
                repo_url=review.repo_url,
                pr_number=review.pr_id,
                findings=findings,
            )
            logger.info(f"Posted comment to PR #{review.pr_id}")
            return response
        except GitHubAPIError as e:
            logger.error(f"Failed to post comment: {str(e)}")
            raise
