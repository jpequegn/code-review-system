"""
GitHub Webhook Handler

Handles GitHub webhook events with signature verification.
Supports PR opened and synchronize (push) events.
"""

import hmac
import hashlib
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

from src.database import Review, ReviewStatus

logger = logging.getLogger(__name__)


# ============================================================================
# GitHub Webhook Payload Models
# ============================================================================


@dataclass
class GitHubUser:
    """GitHub user information."""

    login: str
    id: int


@dataclass
class GitHubRepository:
    """GitHub repository information."""

    id: int
    name: str
    full_name: str
    clone_url: str


@dataclass
class GitHubPullRequest:
    """GitHub pull request information."""

    id: int
    number: int
    title: str
    body: Optional[str]
    head_sha: str
    head_ref: str
    base_ref: str
    user: GitHubUser
    url: str


@dataclass
class GitHubWebhookPayload:
    """GitHub webhook payload."""

    action: str  # opened, synchronize, closed, etc.
    pull_request: GitHubPullRequest
    repository: GitHubRepository
    sender: GitHubUser


# ============================================================================
# Webhook Signature Verification
# ============================================================================


def verify_github_signature(
    payload: bytes, signature_header: str, webhook_secret: str
) -> bool:
    """
    Verify GitHub webhook signature.

    GitHub sends X-Hub-Signature-256 header with format:
    sha256=<hex_digest>

    Args:
        payload: Raw request body bytes
        signature_header: Value of X-Hub-Signature-256 header
        webhook_secret: Secret configured in GitHub webhook settings

    Returns:
        True if signature is valid, False otherwise
    """
    # Extract algorithm and signature from header
    try:
        algorithm, signature = signature_header.split("=", 1)
    except ValueError:
        logger.warning("Invalid signature header format")
        return False

    if algorithm != "sha256":
        logger.warning(f"Unexpected signature algorithm: {algorithm}")
        return False

    # Compute expected signature
    expected_signature = hmac.new(
        webhook_secret.encode(), payload, hashlib.sha256
    ).hexdigest()

    # Use constant-time comparison to prevent timing attacks
    return hmac.compare_digest(signature, expected_signature)


# ============================================================================
# Webhook Payload Parsing
# ============================================================================


def parse_github_payload(
    payload_dict: Dict[str, Any],
) -> Optional[GitHubWebhookPayload]:
    """
    Parse GitHub webhook payload from JSON.

    Args:
        payload_dict: Parsed JSON webhook payload

    Returns:
        GitHubWebhookPayload if valid, None otherwise

    Raises:
        KeyError: If required fields are missing
        ValueError: If data cannot be parsed
    """
    try:
        # Extract action
        action = payload_dict.get("action")
        if not action:
            logger.warning("Webhook missing 'action' field")
            return None

        # Extract PR information
        pr_dict = payload_dict.get("pull_request", {})
        if not pr_dict:
            logger.warning("Webhook missing 'pull_request' field")
            return None

        pr = GitHubPullRequest(
            id=pr_dict["id"],
            number=pr_dict["number"],
            title=pr_dict["title"],
            body=pr_dict.get("body"),
            head_sha=pr_dict["head"]["sha"],
            head_ref=pr_dict["head"]["ref"],
            base_ref=pr_dict["base"]["ref"],
            user=GitHubUser(login=pr_dict["user"]["login"], id=pr_dict["user"]["id"]),
            url=pr_dict["html_url"],
        )

        # Extract repository information
        repo_dict = payload_dict.get("repository", {})
        repo = GitHubRepository(
            id=repo_dict["id"],
            name=repo_dict["name"],
            full_name=repo_dict["full_name"],
            clone_url=repo_dict["clone_url"],
        )

        # Extract sender information
        sender_dict = payload_dict.get("sender", {})
        sender = GitHubUser(login=sender_dict["login"], id=sender_dict["id"])

        return GitHubWebhookPayload(
            action=action,
            pull_request=pr,
            repository=repo,
            sender=sender,
        )

    except (KeyError, TypeError) as e:
        logger.error(f"Failed to parse GitHub webhook payload: {str(e)}")
        raise ValueError(f"Invalid payload structure: {str(e)}")


# ============================================================================
# Webhook Event Handler
# ============================================================================


def handle_github_webhook(payload: GitHubWebhookPayload, db_session) -> Dict[str, Any]:
    """
    Handle GitHub webhook event.

    Creates or updates Review record for PR opened/synchronize events.
    Ignores other PR actions.

    Args:
        payload: Parsed GitHub webhook payload
        db_session: Database session for persistence

    Returns:
        Dictionary with review ID and status

    Raises:
        ValueError: If event action is not supported
    """
    # Only handle opened and synchronize (code push) actions
    if payload.action not in ["opened", "synchronize"]:
        logger.info(
            f"Ignoring PR action '{payload.action}' for PR #{payload.pull_request.number}"
        )
        return {"ignored": True, "action": payload.action}

    logger.info(
        f"Processing PR #{payload.pull_request.number} "
        f"({payload.action}) in {payload.repository.full_name}"
    )

    pr = payload.pull_request
    repo = payload.repository

    # Check if review already exists for this PR
    existing_review = db_session.query(Review).filter(Review.pr_id == pr.number).first()

    if existing_review:
        # Update existing review for synchronize event
        logger.debug(f"Updating existing review for PR #{pr.number}")
        existing_review.commit_sha = pr.head_sha
        existing_review.branch = pr.head_ref
        existing_review.status = ReviewStatus.PENDING  # Reset to pending for new push
        db_session.commit()
        review = existing_review
    else:
        # Create new review
        logger.debug(f"Creating new review for PR #{pr.number}")
        review = Review(
            pr_id=pr.number,
            repo_url=repo.clone_url,
            branch=pr.head_ref,
            commit_sha=pr.head_sha,
            status=ReviewStatus.PENDING,
        )
        db_session.add(review)
        db_session.commit()
        db_session.refresh(review)

    return {
        "review_id": review.id,
        "pr_id": review.pr_id,
        "action": payload.action,
        "status": review.status.value,
    }
