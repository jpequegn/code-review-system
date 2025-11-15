"""
Pytest configuration and shared fixtures.
"""

import pytest
from fastapi.testclient import TestClient
from src.main import app


@pytest.fixture
def client():
    """
    Provide a test client for FastAPI.

    Returns:
        TestClient: Test client for making requests to the app
    """
    return TestClient(app)
