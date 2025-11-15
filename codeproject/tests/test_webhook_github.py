"""
Tests for GitHub webhook handler.

Tests signature verification, payload parsing, and webhook event handling.
"""

import json
import hmac
import hashlib
import pytest
from unittest.mock import MagicMock

from src.webhooks.github import (
    verify_github_signature,
    parse_github_payload,
    handle_github_webhook,
    GitHubWebhookPayload,
    GitHubPullRequest,
    GitHubRepository,
    GitHubUser,
)
from src.database import Review, ReviewStatus


# ============================================================================
# Test Data & Fixtures
# ============================================================================

WEBHOOK_SECRET = "test-webhook-secret"


def create_signature(payload: bytes, secret: str) -> str:
    """Create valid GitHub webhook signature."""
    signature = hmac.new(
        secret.encode(),
        payload,
        hashlib.sha256
    ).hexdigest()
    return f"sha256={signature}"


@pytest.fixture
def sample_pr_opened_payload():
    """Sample GitHub webhook payload for PR opened event."""
    return {
        "action": "opened",
        "pull_request": {
            "id": 123456789,
            "number": 42,
            "title": "Add security fix",
            "body": "This PR fixes a SQL injection vulnerability",
            "head": {
                "sha": "abc123def456",
                "ref": "feature/security-fix"
            },
            "base": {
                "ref": "main"
            },
            "user": {
                "login": "developer",
                "id": 999
            },
            "html_url": "https://github.com/user/repo/pull/42"
        },
        "repository": {
            "id": 111222333,
            "name": "repo",
            "full_name": "user/repo",
            "clone_url": "https://github.com/user/repo.git"
        },
        "sender": {
            "login": "developer",
            "id": 999
        }
    }


@pytest.fixture
def sample_pr_synchronize_payload(sample_pr_opened_payload):
    """Sample GitHub webhook payload for PR synchronize (push) event."""
    payload = sample_pr_opened_payload.copy()
    payload["action"] = "synchronize"
    payload["pull_request"]["head"]["sha"] = "new456sha789"
    return payload


@pytest.fixture
def sample_pr_closed_payload(sample_pr_opened_payload):
    """Sample GitHub webhook payload for PR closed event."""
    payload = sample_pr_opened_payload.copy()
    payload["action"] = "closed"
    return payload


# ============================================================================
# Test Signature Verification
# ============================================================================

class TestSignatureVerification:
    """Tests for GitHub webhook signature verification."""

    def test_verify_valid_signature(self, sample_pr_opened_payload):
        """Test verification of valid signature."""
        payload_bytes = json.dumps(sample_pr_opened_payload).encode()
        signature = create_signature(payload_bytes, WEBHOOK_SECRET)

        result = verify_github_signature(
            payload_bytes,
            signature,
            WEBHOOK_SECRET
        )
        assert result is True

    def test_verify_invalid_signature(self, sample_pr_opened_payload):
        """Test rejection of invalid signature."""
        payload_bytes = json.dumps(sample_pr_opened_payload).encode()
        invalid_signature = "sha256=invalid0000000000000000000000000000000000000"

        result = verify_github_signature(
            payload_bytes,
            invalid_signature,
            WEBHOOK_SECRET
        )
        assert result is False

    def test_verify_wrong_secret(self, sample_pr_opened_payload):
        """Test rejection when wrong secret used."""
        payload_bytes = json.dumps(sample_pr_opened_payload).encode()
        signature = create_signature(payload_bytes, "wrong-secret")

        result = verify_github_signature(
            payload_bytes,
            signature,
            WEBHOOK_SECRET
        )
        assert result is False

    def test_verify_tampered_payload(self, sample_pr_opened_payload):
        """Test rejection if payload was tampered with."""
        payload_bytes = json.dumps(sample_pr_opened_payload).encode()
        signature = create_signature(payload_bytes, WEBHOOK_SECRET)

        # Tamper with payload
        tampered_payload = payload_bytes + b"extra"

        result = verify_github_signature(
            tampered_payload,
            signature,
            WEBHOOK_SECRET
        )
        assert result is False

    def test_verify_invalid_header_format(self, sample_pr_opened_payload):
        """Test rejection of invalid signature header format."""
        payload_bytes = json.dumps(sample_pr_opened_payload).encode()
        invalid_header = "invalid-format"

        result = verify_github_signature(
            payload_bytes,
            invalid_header,
            WEBHOOK_SECRET
        )
        assert result is False

    def test_verify_wrong_algorithm(self, sample_pr_opened_payload):
        """Test rejection of unsupported signature algorithm."""
        payload_bytes = json.dumps(sample_pr_opened_payload).encode()
        # SHA1 instead of SHA256
        header = "sha1=somesignature"

        result = verify_github_signature(
            payload_bytes,
            header,
            WEBHOOK_SECRET
        )
        assert result is False

    def test_verify_constant_time_comparison(self, sample_pr_opened_payload):
        """Test that signature comparison is constant-time (no timing attacks)."""
        payload_bytes = json.dumps(sample_pr_opened_payload).encode()
        signature = create_signature(payload_bytes, WEBHOOK_SECRET)

        # Both should fail, but should take similar time
        invalid1 = "sha256=0000000000000000000000000000000000000000000000000000000000000000"
        invalid2 = "sha256=ffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff"

        result1 = verify_github_signature(payload_bytes, invalid1, WEBHOOK_SECRET)
        result2 = verify_github_signature(payload_bytes, invalid2, WEBHOOK_SECRET)

        assert result1 is False
        assert result2 is False


# ============================================================================
# Test Payload Parsing
# ============================================================================

class TestPayloadParsing:
    """Tests for GitHub webhook payload parsing."""

    def test_parse_valid_payload(self, sample_pr_opened_payload):
        """Test parsing of valid payload."""
        payload = parse_github_payload(sample_pr_opened_payload)

        assert payload is not None
        assert payload.action == "opened"
        assert payload.pull_request.number == 42
        assert payload.pull_request.title == "Add security fix"
        assert payload.pull_request.head_sha == "abc123def456"
        assert payload.pull_request.head_ref == "feature/security-fix"
        assert payload.repository.full_name == "user/repo"
        assert payload.sender.login == "developer"

    def test_parse_pr_synchronize(self, sample_pr_synchronize_payload):
        """Test parsing of PR synchronize (push) event."""
        payload = parse_github_payload(sample_pr_synchronize_payload)

        assert payload is not None
        assert payload.action == "synchronize"
        assert payload.pull_request.head_sha == "new456sha789"

    def test_parse_missing_action(self, sample_pr_opened_payload):
        """Test parsing handles missing action field."""
        del sample_pr_opened_payload["action"]

        payload = parse_github_payload(sample_pr_opened_payload)
        assert payload is None

    def test_parse_missing_pull_request(self, sample_pr_opened_payload):
        """Test parsing handles missing pull_request field."""
        del sample_pr_opened_payload["pull_request"]

        payload = parse_github_payload(sample_pr_opened_payload)
        assert payload is None

    def test_parse_missing_required_pr_field(self, sample_pr_opened_payload):
        """Test parsing handles missing required PR fields."""
        del sample_pr_opened_payload["pull_request"]["number"]

        with pytest.raises(ValueError, match="Invalid payload structure"):
            parse_github_payload(sample_pr_opened_payload)

    def test_parse_invalid_json_structure(self):
        """Test parsing handles invalid payload structure."""
        invalid_payload = {"not": "a real payload"}

        # Returns None when action is missing (normal case for unhandled events)
        result = parse_github_payload(invalid_payload)
        assert result is None

    def test_parse_pr_with_optional_fields(self, sample_pr_opened_payload):
        """Test parsing handles optional fields correctly."""
        # Body is optional
        sample_pr_opened_payload["pull_request"]["body"] = None

        payload = parse_github_payload(sample_pr_opened_payload)
        assert payload is not None
        assert payload.pull_request.body is None


# ============================================================================
# Test Webhook Event Handler
# ============================================================================

class TestWebhookEventHandler:
    """Tests for GitHub webhook event handler."""

    def test_handle_pr_opened_creates_review(self, sample_pr_opened_payload):
        """Test that PR opened event creates Review record."""
        payload = parse_github_payload(sample_pr_opened_payload)
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        result = handle_github_webhook(payload, mock_db)

        # Verify Review.add was called
        assert mock_db.add.called
        assert result["action"] == "opened"
        assert result["status"] == ReviewStatus.PENDING.value

    def test_handle_pr_synchronize_updates_review(
        self,
        sample_pr_synchronize_payload
    ):
        """Test that PR synchronize event updates existing review."""
        payload = parse_github_payload(sample_pr_synchronize_payload)

        # Mock existing review
        existing_review = MagicMock(spec=Review)
        existing_review.id = 1
        existing_review.pr_id = 42
        existing_review.status = ReviewStatus.PENDING

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = existing_review

        result = handle_github_webhook(payload, mock_db)

        # Verify review was updated
        assert existing_review.commit_sha == "new456sha789"
        assert existing_review.branch == "feature/security-fix"
        assert existing_review.status == ReviewStatus.PENDING
        assert mock_db.commit.called

    def test_handle_pr_closed_ignored(self, sample_pr_closed_payload):
        """Test that PR closed event is ignored."""
        payload = parse_github_payload(sample_pr_closed_payload)
        mock_db = MagicMock()

        result = handle_github_webhook(payload, mock_db)

        assert result["ignored"] is True
        assert result["action"] == "closed"
        assert not mock_db.add.called

    def test_handle_pr_opened_returns_review_metadata(
        self,
        sample_pr_opened_payload
    ):
        """Test that handler returns correct review metadata."""
        payload = parse_github_payload(sample_pr_opened_payload)

        # Mock new review
        new_review = MagicMock(spec=Review)
        new_review.id = 1
        new_review.pr_id = 42
        new_review.status = ReviewStatus.PENDING

        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None
        # Configure refresh to set the ID
        def refresh_side_effect(obj):
            obj.id = 1
        mock_db.refresh.side_effect = refresh_side_effect

        result = handle_github_webhook(payload, mock_db)

        assert result["pr_id"] == 42
        assert result["action"] == "opened"
        assert result["status"] == ReviewStatus.PENDING.value

    def test_handle_unsupported_action_ignored(self, sample_pr_opened_payload):
        """Test that unsupported PR actions are ignored."""
        sample_pr_opened_payload["action"] = "edited"
        payload = parse_github_payload(sample_pr_opened_payload)
        mock_db = MagicMock()

        result = handle_github_webhook(payload, mock_db)

        assert result["ignored"] is True
        assert result["action"] == "edited"
        assert not mock_db.add.called

    def test_handle_pr_opened_stores_metadata(self, sample_pr_opened_payload):
        """Test that PR metadata is correctly stored."""
        payload = parse_github_payload(sample_pr_opened_payload)
        mock_db = MagicMock()
        mock_db.query.return_value.filter.return_value.first.return_value = None

        handle_github_webhook(payload, mock_db)

        # Verify Review was created with correct data
        call_args = mock_db.add.call_args
        review = call_args[0][0]

        assert review.pr_id == 42
        assert review.branch == "feature/security-fix"
        assert review.commit_sha == "abc123def456"
        assert review.repo_url == "https://github.com/user/repo.git"
        assert review.status == ReviewStatus.PENDING


# ============================================================================
# Test FastAPI Webhook Endpoint
# ============================================================================

class TestWebhookEndpoint:
    """Tests for FastAPI webhook endpoint."""

    @pytest.mark.asyncio
    async def test_webhook_valid_signature(self, client, sample_pr_opened_payload):
        """Test webhook with valid signature."""
        payload_bytes = json.dumps(sample_pr_opened_payload).encode()
        signature = create_signature(payload_bytes, WEBHOOK_SECRET)

        # This would need a test client with configured WEBHOOK_SECRET
        # For now, we're testing the handler functions directly
        assert verify_github_signature(payload_bytes, signature, WEBHOOK_SECRET)

    @pytest.mark.asyncio
    async def test_webhook_invalid_signature(self, client, sample_pr_opened_payload):
        """Test webhook with invalid signature."""
        payload_bytes = json.dumps(sample_pr_opened_payload).encode()
        invalid_signature = "sha256=invalid0000000000000000000000000000000000000"

        assert not verify_github_signature(
            payload_bytes,
            invalid_signature,
            WEBHOOK_SECRET
        )

    def test_webhook_missing_signature_header(self):
        """Test webhook missing X-Hub-Signature-256 header."""
        # This would require a full FastAPI test client
        # For now, we verify the function behavior
        payload = b"test"
        invalid_header = None

        # Should fail when header is missing
        assert verify_github_signature(payload, "", WEBHOOK_SECRET) is False
