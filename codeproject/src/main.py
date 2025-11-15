"""
Code Security & Performance Reviewer - FastAPI Application

Main entry point for the LLM-powered code review service.
"""

from fastapi import FastAPI
from fastapi.responses import JSONResponse

# Create FastAPI application
app = FastAPI(
    title="Code Security & Performance Reviewer",
    description="LLM-powered CI/CD code review analyzing security vulnerabilities and performance issues",
    version="0.1.0"
)


@app.get("/health")
async def health():
    """
    Health check endpoint.

    Returns:
        dict: Status indicator showing the service is healthy
    """
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
