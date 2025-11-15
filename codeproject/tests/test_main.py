"""
Tests for the main FastAPI application.
"""

import pytest


def test_health_endpoint(client):
    """
    Test that the health endpoint returns 200 with correct status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_health_endpoint_json_format(client):
    """
    Test that the health endpoint returns valid JSON.
    """
    response = client.get("/health")
    assert response.headers["content-type"] == "application/json"
    data = response.json()
    assert "status" in data
    assert data["status"] == "healthy"


def test_app_starts_without_errors():
    """
    Test that the FastAPI app initializes without errors.
    """
    from src.main import app
    assert app is not None
    assert app.title == "Code Security & Performance Reviewer"
