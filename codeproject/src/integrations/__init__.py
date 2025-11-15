"""
Integrations Package

Provides integrations with external services (GitHub, GitLab, etc.)
"""

from src.integrations.github_api import (
    GitHubAPIClient,
    RepositoryNotFoundError,
    GitHubAPIError,
)

__all__ = [
    "GitHubAPIClient",
    "RepositoryNotFoundError",
    "GitHubAPIError",
]
