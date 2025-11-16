# Code Review System - Usage Guide

Complete guide to using the LLM-powered code review system with practical examples.

## Table of Contents

1. [Setup & Installation](#setup--installation)
2. [Quick Examples](#quick-examples)
3. [Running the Service](#running-the-service)
4. [GitHub Webhook Setup](#github-webhook-setup)
5. [Testing the System](#testing-the-system)
6. [Troubleshooting](#troubleshooting)
7. [Advanced Usage](#advanced-usage)

## Setup & Installation

### Step 1: Prerequisites

Ensure you have:
- Python 3.11 or later
- Git
- A GitHub account with repository access
- API credentials (Claude API key or Ollama)

Check Python version:
```bash
python3 --version
# Should output: Python 3.11.x or higher
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/jpequegn/code-review-system.git
cd code-review-system
```

### Step 3: Install the Package

```bash
cd codeproject

# Install in development mode (recommended)
pip install -e .

# Or install production mode
pip install .
```

### Step 4: Configure Environment

```bash
# Copy example configuration
cp .env.example .env

# Edit the configuration
nano .env  # or use your preferred editor
```

### Step 5: Configuration Details

Edit `.env` with your settings:

```bash
# === LLM Configuration ===

# Option A: Use Claude (Anthropic)
LLM_PROVIDER=claude
CLAUDE_API_KEY=sk-ant-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Option B: Use Ollama (local models)
# LLM_PROVIDER=ollama
# OLLAMA_BASE_URL=http://localhost:11434

# === GitHub Configuration ===

# GitHub token for posting PR comments
# Generate at: https://github.com/settings/tokens
# Permissions needed: repo (all), if private repo
GITHUB_TOKEN=ghp_xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# Webhook secret for verifying GitHub requests
# Can be any random string you create
WEBHOOK_SECRET=your-secret-webhook-password-here

# === Server Configuration ===

HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR

# === Database Configuration ===

# Default SQLite is fine for MVP
DATABASE_URL=sqlite:///./codeproject.db

# For production, use PostgreSQL
# DATABASE_URL=postgresql://user:password@localhost/dbname
```

## Quick Examples

### Example 1: Using Claude

**Configuration:**
```bash
LLM_PROVIDER=claude
CLAUDE_API_KEY=sk-ant-...
GITHUB_TOKEN=ghp_...
WEBHOOK_SECRET=my-webhook-secret
```

**Usage:**
```bash
python -m uvicorn src.main:app --reload
# Server runs at http://localhost:8000
```

### Example 2: Using Ollama Locally

**Prerequisites:**
```bash
# Download and run Ollama
# From https://ollama.ai
ollama pull llama2  # or mistral, neural-chat, etc.
ollama serve
```

**Configuration:**
```bash
LLM_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
GITHUB_TOKEN=ghp_...
WEBHOOK_SECRET=my-webhook-secret
```

**Usage:**
```bash
python -m uvicorn src.main:app --reload
# Server runs at http://localhost:8000
```

### Example 3: Using Docker Compose

**Configuration:**
```bash
cp .env.example .env
# Edit .env with your values
```

**Run:**
```bash
# With Claude
docker-compose up

# Or with Ollama (uncomment profiles in docker-compose.yml first)
docker-compose --profile ollama up
```

## Running the Service

### Development Mode (Recommended for Testing)

```bash
cd codeproject

# Terminal 1: Start the server
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000

# You should see:
# INFO:     Uvicorn running on http://0.0.0.0:8000
# INFO:     Application startup complete
```

### Test Server is Running

**Terminal 2:**
```bash
# Health check
curl http://localhost:8000/health

# Response:
# {"status":"healthy"}
```

### Production Mode (Docker)

```bash
cd codeproject

# Start services
docker-compose up -d

# Check logs
docker-compose logs -f reviewer

# Stop services
docker-compose down
```

### Production Mode (Direct)

```bash
cd codeproject

# Start with gunicorn (production ASGI server)
pip install gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.main:app --bind 0.0.0.0:8000
```

## GitHub Webhook Setup

### Step 1: Generate Webhook Secret

Create a random secret (or use one you configure):

```bash
# macOS/Linux
openssl rand -hex 32
# Output: a1b2c3d4e5f6... (32 characters)

# Windows (PowerShell)
-join ((1..32 | ForEach-Object { [char](Get-Random -Minimum 65 -Maximum 90) }) +
       (1..32 | ForEach-Object { [char](Get-Random -Minimum 48 -Maximum 57) })) |
       % { $_ }
```

Add this to `.env`:
```bash
WEBHOOK_SECRET=a1b2c3d4e5f6...
```

### Step 2: Configure GitHub Webhook

1. Go to your GitHub repository
2. Navigate to **Settings** → **Webhooks** → **Add webhook**

**Fill in the form:**

| Field | Value |
|-------|-------|
| Payload URL | `https://your-domain.com/webhook/github` |
| Content type | `application/json` |
| Secret | (paste your WEBHOOK_SECRET value) |
| Which events? | **Pull requests** |
| Active | ✓ (check this box) |

**For Local Testing:**

If you're testing locally, you can use a tunnel:

```bash
# Using ngrok (free tier available at https://ngrok.com)
ngrok http 8000

# Copy the HTTPS URL (e.g., https://a1b2c3d4.ngrok.io)
# Use this as your Payload URL: https://a1b2c3d4.ngrok.io/webhook/github
```

### Step 3: Test the Webhook

1. Go back to your webhook settings
2. Scroll to "Recent Deliveries"
3. Click the test delivery
4. Check that it was received (status 202)

## Testing the System

### Test 1: Verify Server Health

```bash
curl -s http://localhost:8000/health | jq
# Output:
# {
#   "status": "healthy"
# }
```

### Test 2: Verify LLM Provider

```bash
cd codeproject

python << 'EOF'
from src.llm.provider import get_llm_provider

# Get configured provider
provider = get_llm_provider()
print(f"Provider: {provider.__class__.__name__}")

# Test it works
result = provider.analyze_security("print(user_input)")
print(f"Result: {result}")
EOF
```

### Test 3: Test Webhook Signature Verification

```bash
# Create a test payload
python << 'EOF'
import json
import hmac
import hashlib
import os

# Sample payload
payload = {
    "action": "opened",
    "pull_request": {
        "id": 123,
        "number": 42,
        "title": "Test PR",
        "head": {"sha": "abc123", "ref": "feature/test"},
        "base": {"ref": "main"},
        "user": {"login": "testuser"},
        "html_url": "https://github.com/user/repo/pull/42"
    },
    "repository": {
        "name": "test-repo",
        "full_name": "user/test-repo",
        "clone_url": "https://github.com/user/test-repo.git"
    }
}

# Get your WEBHOOK_SECRET from .env
secret = os.getenv("WEBHOOK_SECRET", "test-secret")
payload_json = json.dumps(payload)

# Create signature
signature = hmac.new(
    secret.encode(),
    payload_json.encode(),
    hashlib.sha256
).hexdigest()

print(f"Payload: {payload_json[:100]}...")
print(f"Signature: sha256={signature}")

# Use this to test your endpoint:
# curl -X POST http://localhost:8000/webhook/github \
#   -H "Content-Type: application/json" \
#   -H "X-GitHub-Event: pull_request" \
#   -H "X-Hub-Signature-256: sha256={signature}" \
#   -d '{payload}'
EOF
```

### Test 4: Run the Test Suite

```bash
cd codeproject

# Run all tests
python -m pytest tests/ -v

# Run with coverage report
python -m pytest tests/ -v --cov=src --cov-report=term-missing

# View HTML coverage report
python -m pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

### Test 5: Test End-to-End with Real Repository

```bash
cd codeproject

python << 'EOF'
from src.review_service import ReviewService
from src.database import SessionLocal, Review, ReviewStatus, Finding

# Create database session
db = SessionLocal()

# Create a test review for a real repository
review = Review(
    pr_id=999,
    repo_url="https://github.com/python/cpython.git",
    branch="main",
    commit_sha="abc123def456",
    status=ReviewStatus.PENDING
)
db.add(review)
db.commit()
print(f"Created review: {review.id}")

# Process the review
try:
    service = ReviewService(db=db)
    result = service.process_review(review.id)
    print(f"Result: {result}")

    # Check findings
    findings = db.query(Finding).filter(Finding.review_id == review.id).all()
    print(f"\nFound {len(findings)} issues:")
    for f in findings:
        print(f"  - {f.severity.value}: {f.title}")
        print(f"    Location: {f.file_path}:{f.line_number}")
        if f.suggested_fix:
            print(f"    Fix: {f.suggested_fix[:100]}...")
except Exception as e:
    print(f"Error: {e}")
    db.rollback()
finally:
    db.close()
EOF
```

## Troubleshooting

### Problem: "Python 3.11+ required"

```bash
# Check your Python version
python3 --version

# If you have multiple Python versions:
python3.11 -m pip install -e .
python3.11 -m uvicorn src.main:app --reload
```

### Problem: "No module named 'src'"

```bash
# Solution: Install package in development mode
cd codeproject
pip install -e .

# Or ensure you're in the correct directory
pwd  # Should end with /codeproject
```

### Problem: "GitHub token not configured"

```bash
# Check .env file exists
ls -la .env

# Check GITHUB_TOKEN is set
grep GITHUB_TOKEN .env

# If empty, generate a token:
# 1. Go to https://github.com/settings/tokens
# 2. Click "Generate new token"
# 3. Copy the token to .env
```

### Problem: "Webhook signature verification failed"

```bash
# Check that WEBHOOK_SECRET in .env matches GitHub webhook secret
# 1. Go to GitHub: Settings > Webhooks
# 2. Click your webhook
# 3. Check that "Secret" matches WEBHOOK_SECRET in .env

# If they don't match, regenerate:
openssl rand -hex 32  # Generate new secret
# Update both .env and GitHub webhook settings
```

### Problem: "LLM connection timeout"

**For Claude:**
```bash
# Check your API key is valid
# 1. Go to https://console.anthropic.com
# 2. Check API key hasn't been revoked
# 3. Verify CLAUDE_API_KEY in .env

# Test the connection
python << 'EOF'
from anthropic import Anthropic
client = Anthropic()
print("Claude API: OK")
EOF
```

**For Ollama:**
```bash
# Check Ollama is running
curl http://localhost:11434/api/tags

# If failed, start Ollama
ollama serve

# Check model is available
ollama list
```

### Problem: "Port 8000 already in use"

```bash
# Find what's using port 8000
lsof -i :8000

# Kill the process (macOS/Linux)
kill -9 <PID>

# Or use a different port
python -m uvicorn src.main:app --port 8001
```

### Problem: Docker build fails

```bash
# Check Docker daemon is running
docker ps

# If failed, start Docker
# macOS: open /Applications/Docker.app
# Linux: sudo systemctl start docker

# Try building again
docker-compose build --no-cache
```

## Advanced Usage

### Custom LLM Provider

To add support for a new LLM (e.g., GPT-4, Cohere):

1. **Create provider class** (`src/llm/gpt4.py`):

```python
from src.llm.provider import LLMProvider

class GPT4Provider(LLMProvider):
    def __init__(self, api_key: str):
        self.api_key = api_key
        # Initialize OpenAI client

    def analyze_security(self, code_snippet: str) -> dict:
        # Implement security analysis
        pass

    def analyze_performance(self, code_snippet: str) -> dict:
        # Implement performance analysis
        pass
```

2. **Register in provider factory** (`src/llm/provider.py`):

```python
def get_llm_provider() -> LLMProvider:
    if settings.llm_provider == "gpt4":
        return GPT4Provider(settings.openai_api_key)
    # ... existing providers
```

3. **Add tests** (`tests/test_llm_provider.py`)

4. **Update documentation** and `.env.example`

### Running with PostgreSQL

For production deployments:

1. **Install PostgreSQL**

```bash
# macOS
brew install postgresql

# Ubuntu
sudo apt-get install postgresql

# Start server
brew services start postgresql  # macOS
```

2. **Create database**

```bash
createdb code_review
```

3. **Update `.env`**

```bash
DATABASE_URL=postgresql://user:password@localhost/code_review
```

4. **Run migrations** (if applicable)

```bash
python -m alembic upgrade head
```

### Monitoring and Logging

**View logs in development:**

```bash
# Server logs are printed to console
# Check for analysis errors or webhook issues
```

**Check database:**

```bash
# Connect to SQLite
sqlite3 codeproject.db

# View reviews
SELECT id, pr_id, status, created_at FROM reviews;

# View findings
SELECT id, review_id, severity, title, file_path FROM findings;

# Exit
.quit
```

**Monitor with Docker:**

```bash
# View container logs
docker-compose logs -f reviewer

# View specific errors
docker-compose logs reviewer | grep ERROR

# Check resource usage
docker stats code-reviewer
```

### Batch Processing Multiple Repositories

If you want to analyze multiple repositories:

```python
# batch_review.py
from src.review_service import ReviewService
from src.database import SessionLocal, Review, ReviewStatus

repositories = [
    "https://github.com/user/repo1.git",
    "https://github.com/user/repo2.git",
    "https://github.com/user/repo3.git",
]

db = SessionLocal()
service = ReviewService(db=db)

for repo_url in repositories:
    review = Review(
        pr_id=1,  # Set appropriate PR ID
        repo_url=repo_url,
        branch="main",
        commit_sha="latest",
        status=ReviewStatus.PENDING
    )
    db.add(review)
    db.commit()

    try:
        result = service.process_review(review.id)
        print(f"✓ {repo_url}: {result['findings_count']} findings")
    except Exception as e:
        print(f"✗ {repo_url}: {e}")

db.close()
```

Run it:
```bash
python batch_review.py
```

### Performance Tuning

**For large repositories:**

1. **Use shallow clones** (already configured)
2. **Increase LLM timeout** in `.env`:
   ```bash
   LLM_TIMEOUT=60  # Seconds
   ```

3. **Increase code snippet size limit**:
   ```python
   # In src/analysis/analyzer.py
   MAX_CODE_SNIPPET_LENGTH = 100000  # characters
   ```

4. **Run multiple instances** behind load balancer:
   ```bash
   # Terminal 1
   python -m uvicorn src.main:app --port 8000

   # Terminal 2
   python -m uvicorn src.main:app --port 8001

   # Terminal 3
   python -m uvicorn src.main:app --port 8002

   # Use nginx/haproxy to load balance
   ```

---

**Need help?** Check the main [README.md](README.md) or open a GitHub issue.
