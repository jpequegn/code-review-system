# Security & Performance LLM Code Review System - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build a self-hosted CI/CD service that analyzes code changes for security vulnerabilities and performance/scalability issues using pluggable LLM backends.

**Architecture:** Webhook-based service (FastAPI) that receives Git events, extracts code diffs, routes to LLM analyzer (Claude API or local models), correlates findings, posts PR comments, and stores audit trail. Pluggable LLM backends via abstract provider interface. SQLite for MVP, PostgreSQL for scale.

**Tech Stack:**
- **Language**: Python 3.11+
- **Framework**: FastAPI (async HTTP webhooks)
- **Database**: SQLite (MVP), PostgreSQL option
- **LLM Integration**: langchain + anthropic SDK (Claude) + ollama (local)
- **Testing**: pytest + pytest-asyncio
- **Containerization**: Docker + docker-compose
- **VCS Integration**: GitPython + requests (webhook posting)

---

## Phase 1: MVP (Webhook + Analysis + PR Comments)

### Task 1: Project Setup & Scaffolding

**Files:**
- Create: `codeproject/` (new project directory)
- Create: `codeproject/pyproject.toml`
- Create: `codeproject/.env.example`
- Create: `codeproject/Dockerfile`
- Create: `codeproject/docker-compose.yml`
- Create: `codeproject/src/main.py`
- Create: `codeproject/tests/conftest.py`

**Step 1: Create project directory structure**

```bash
mkdir -p codeproject/{src,tests,docs}
cd codeproject
```

**Step 2: Create pyproject.toml**

```toml
[build-system]
requires = ["setuptools>=68.0", "wheel"]
build-backend = "setuptools.build_meta"

[project]
name = "code-security-performance-reviewer"
version = "0.1.0"
description = "LLM-powered CI/CD code review for security and performance"
requires-python = ">=3.11"
dependencies = [
    "fastapi==0.104.1",
    "uvicorn[standard]==0.24.0",
    "pydantic==2.5.0",
    "pydantic-settings==2.1.0",
    "anthropic==0.7.13",
    "langchain==0.1.0",
    "langchain-anthropic==0.1.0",
    "gitpython==3.1.40",
    "requests==2.31.0",
    "sqlalchemy==2.0.23",
    "pytest==7.4.3",
    "pytest-asyncio==0.21.1",
    "pytest-cov==4.1.0",
    "httpx==0.25.2",
]

[project.optional-dependencies]
dev = ["black==23.12.0", "ruff==0.1.9", "mypy==1.7.1"]
```

**Step 3: Create .env.example**

```bash
cat > .env.example << 'EOF'
# LLM Provider Configuration
LLM_PROVIDER=claude  # Options: claude, ollama
CLAUDE_API_KEY=sk-...
OLLAMA_BASE_URL=http://localhost:11434

# Git Integration
WEBHOOK_SECRET=your-webhook-secret
GITHUB_TOKEN=ghp_...
GITLAB_TOKEN=...

# Database
DATABASE_URL=sqlite:///./codeproject.db

# Server
HOST=0.0.0.0
PORT=8000
EOF
```

**Step 4: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .

# Copy source
COPY src/ src/
COPY tests/ tests/

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

**Step 5: Create docker-compose.yml**

```yaml
version: '3.8'

services:
  reviewer:
    build: .
    ports:
      - "8000:8000"
    env_file: .env
    environment:
      DATABASE_URL: sqlite:///./codeproject.db
    volumes:
      - ./codeproject.db:/app/codeproject.db
    restart: unless-stopped

  # Optional: local Ollama for LLM
  ollama:
    image: ollama/ollama
    ports:
      - "11434:11434"
    volumes:
      - ollama_data:/root/.ollama
    restart: unless-stopped

volumes:
  ollama_data:
```

**Step 6: Create src/main.py (minimal FastAPI app)**

```python
from fastapi import FastAPI
from fastapi.responses import JSONResponse

app = FastAPI(
    title="Code Security & Performance Reviewer",
    description="LLM-powered CI/CD code review",
    version="0.1.0"
)

@app.get("/health")
async def health():
    return {"status": "healthy"}
```

**Step 7: Create tests/conftest.py**

```python
import pytest
from fastapi.testclient import TestClient
from src.main import app

@pytest.fixture
def client():
    return TestClient(app)
```

**Step 8: Create basic README.md**

```markdown
# Code Security & Performance Reviewer

LLM-powered CI/CD code review service analyzing security vulnerabilities and performance/scalability issues.

## Quick Start

\`\`\`bash
cp .env.example .env
# Edit .env with your API keys
docker-compose up
\`\`\`

## API

- POST /webhook/github - GitHub webhook endpoint
- GET /health - Health check

See docs/plans/ for implementation details.
\`\`\`
```

**Step 9: Initialize git and commit**

```bash
git init
git add .
git commit -m "init: project scaffolding and dependencies"
```

---

### Task 2: Configuration & Environment Management

**Files:**
- Create: `src/config.py`
- Create: `tests/test_config.py`

**Step 1: Write failing test for config**

```python
# tests/test_config.py
import os
from src.config import Settings

def test_settings_loads_from_env():
    os.environ["LLM_PROVIDER"] = "claude"
    os.environ["CLAUDE_API_KEY"] = "sk-test123"
    os.environ["WEBHOOK_SECRET"] = "secret123"

    settings = Settings()
    assert settings.llm_provider == "claude"
    assert settings.claude_api_key == "sk-test123"
    assert settings.webhook_secret == "secret123"

def test_settings_defaults():
    os.environ.pop("DATABASE_URL", None)
    settings = Settings()
    assert settings.database_url == "sqlite:///./codeproject.db"
```

**Step 2: Run test to verify it fails**

```bash
pytest tests/test_config.py::test_settings_loads_from_env -v
# Expected: FAIL - "No module named 'src.config'"
```

**Step 3: Implement Settings class**

```python
# src/config.py
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    # LLM Configuration
    llm_provider: Literal["claude", "ollama"] = "claude"
    claude_api_key: str = ""
    ollama_base_url: str = "http://localhost:11434"

    # Git Integration
    webhook_secret: str = ""
    github_token: str = ""
    gitlab_token: str = ""

    # Database
    database_url: str = "sqlite:///./codeproject.db"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    class Config:
        env_file = ".env"
        case_sensitive = False

# Global settings instance
settings = Settings()
```

**Step 4: Run tests to verify they pass**

```bash
pytest tests/test_config.py -v
# Expected: PASS (2 passed)
```

**Step 5: Commit**

```bash
git add src/config.py tests/test_config.py
git commit -m "feat: configuration management with pydantic"
```

---

[See full implementation plan in code repository docs/plans/ directory]

---

## Summary

This plan breaks implementation into bite-sized tasks following TDD:
1. Setup scaffolding & dependencies
2. Config management
3. Database schema
4. LLM abstraction layer
5. GitHub webhook handler
6. Diff parsing
7. Analysis engine
8. PR comment posting
9. End-to-end integration
10. Docker deployment

Each task: Write failing test → Run & verify fail → Implement → Verify pass → Commit.

**Ready to execute?**
