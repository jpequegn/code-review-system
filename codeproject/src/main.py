"""
Code Security & Performance Reviewer - FastAPI Application

Main entry point for the LLM-powered code review service.
"""

import logging
from fastapi import FastAPI, Request, HTTPException, Depends, status
from sqlalchemy.orm import Session

from src.config import settings
from src.database import get_db
from src.webhooks.github import (
    verify_github_signature,
    handle_github_webhook,
    parse_github_payload,
)
from src.api.insights_routes import router as insights_router

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Code Security & Performance Reviewer",
    description="LLM-powered CI/CD code review analyzing security vulnerabilities and performance issues",
    version="0.1.0",
)

# Include insights API routes
app.include_router(insights_router)


@app.get("/health")
async def health():
    """
    Health check endpoint.

    Returns:
        dict: Status indicator showing the service is healthy
    """
    return {"status": "healthy"}


@app.post("/webhook/github")
async def github_webhook(
    request: Request,
    db: Session = Depends(get_db),
):
    """
    GitHub webhook endpoint.

    Receives webhook events from GitHub for pull request changes.
    Verifies webhook signature and creates Review records for processing.

    Security:
    - Verifies HMAC-SHA256 signature from X-Hub-Signature-256 header
    - Rejects requests with invalid signatures (401 Unauthorized)

    Events handled:
    - pull_request opened: New pull request
    - pull_request synchronize: Code pushed to existing PR

    Args:
        request: FastAPI Request object
        db: Database session dependency

    Returns:
        dict: Review ID and metadata if successful
        401: If signature verification fails
        400: If webhook payload is invalid
    """
    # Get raw body for signature verification
    body = await request.body()

    # Get signature header
    signature_header = request.headers.get("X-Hub-Signature-256")
    if not signature_header:
        logger.warning("Webhook missing X-Hub-Signature-256 header")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing webhook signature header",
        )

    # Verify webhook secret is configured
    if not settings.webhook_secret:
        logger.error("WEBHOOK_SECRET not configured")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Webhook secret not configured",
        )

    # Verify signature
    if not verify_github_signature(body, signature_header, settings.webhook_secret):
        logger.warning(f"Invalid webhook signature: {signature_header[:20]}...")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid webhook signature"
        )

    # Parse payload
    try:
        payload_dict = await request.json()
    except Exception as e:
        logger.error(f"Failed to parse webhook JSON: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid JSON payload"
        )

    # Parse GitHub payload
    try:
        payload = parse_github_payload(payload_dict)
        if not payload:
            # Not a relevant event (e.g., PR closed)
            return {"ignored": True}
    except ValueError as e:
        logger.error(f"Failed to parse GitHub payload: {str(e)}")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    # Handle webhook event
    try:
        result = handle_github_webhook(payload, db)
        return result
    except Exception as e:
        logger.error(f"Error handling webhook: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing webhook",
        )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
