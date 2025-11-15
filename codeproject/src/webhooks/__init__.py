"""
Webhooks Package

Provides webhook handlers for various git platforms (GitHub, GitLab, etc.)
"""

from src.webhooks.github import (
    verify_github_signature,
    handle_github_webhook,
    GitHubWebhookPayload,
)

__all__ = [
    "verify_github_signature",
    "handle_github_webhook",
    "GitHubWebhookPayload",
]
