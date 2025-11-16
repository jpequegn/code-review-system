# Code Review System - Phase 1 MVP

**LLM-powered CI/CD code review analyzing security vulnerabilities and performance issues in pull requests.**

## Overview

This is a production-ready code review service that integrates with GitHub via webhooks to automatically analyze code changes using AI models. It detects security vulnerabilities and performance issues, then posts detailed findings as PR comments.

### Key Features

✅ **Security Analysis**
- SQL injection and other injection vulnerabilities
- Authentication/authorization flaws
- Hardcoded secrets and credentials
- Unsafe cryptographic patterns
- Input validation issues

✅ **Performance Analysis**
- Algorithm inefficiencies (O(n²) loops, unnecessary recursion)
- Resource leaks (unclosed files, memory issues)
- Architectural problems (N+1 queries, blocking operations)
- Scalability anti-patterns

✅ **GitHub Integration**
- Webhook-based automatic reviews on PR creation/update
- Formatted PR comments with severity indicators
- HMAC-SHA256 signature verification
- Support for private repositories

✅ **Flexible LLM Backend**
- Claude API (via Anthropic SDK)
- Local models (via Ollama)
- Easy provider switching via configuration

✅ **Production Ready**
- SQLite audit trail with full review history
- 248+ comprehensive tests (91% coverage)
- Docker containerization with health checks
- Proper error handling and logging
- Constant-time cryptographic comparisons

## Quick Start

### Prerequisites

- Python 3.11+
- Git
- GitHub account with repository access
- API key for Claude (or Ollama instance for local models)

### Installation

**Option 1: Local Development**

```bash
# Clone and navigate to codeproject
git clone https://github.com/jpequegn/code-review-system.git
cd code-review-system/codeproject

# Install package in development mode
pip install -e .

# Install optional dev tools
pip install -e ".[dev]"

# Create environment file
cp .env.example .env
# Edit .env with your API keys
```

**Option 2: Docker**

```bash
cd code-review-system/codeproject
cp .env.example .env
# Edit .env with your API keys
docker-compose up -d
```

### Configuration

Create `.env` file (or use docker-compose.yml for Docker):

```bash
# LLM Configuration
LLM_PROVIDER=claude  # or 'ollama' for local models
CLAUDE_API_KEY=sk-...
OLLAMA_BASE_URL=http://localhost:11434  # if using Ollama

# GitHub Configuration
GITHUB_TOKEN=ghp_...  # for posting comments
WEBHOOK_SECRET=your-webhook-secret  # for signature verification

# Server Configuration
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=INFO

# Database
DATABASE_URL=sqlite:///./codeproject.db
```

See `.env.example` for all available options.

## Usage

### 1. Running the Service

**Development Mode**

```bash
cd codeproject
python -m uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
```

The server starts at `http://localhost:8000`

**Production Mode (Docker)**

```bash
cd codeproject
docker-compose up -d
```

**Verify it's running**

```bash
curl http://localhost:8000/health
# Response: {"status": "healthy"}
```

### 2. Setting Up GitHub Webhook

1. Go to your GitHub repository → Settings → Webhooks → Add webhook
2. **Payload URL**: `https://your-domain.com/webhook/github`
3. **Content type**: `application/json`
4. **Secret**: Use the same value as `WEBHOOK_SECRET` in `.env`
5. **Events**: Select "Pull requests" (or just "push" if preferred)
6. **Active**: ✓ Check this box

### 3. Triggering Reviews

Once the webhook is configured, reviews happen automatically:

1. Create a PR in your GitHub repository
2. GitHub sends webhook event to your service
3. Service clones the repo
4. Service analyzes the code changes
5. Service posts PR comment with findings

### 4. Example PR Comment

```
## 🔍 Code Review Analysis

Found **2** issue(s) in this pull request.

### 🔴 CRITICAL

#### 🛡️ SQL Injection Vulnerability
**Description:** User input not parameterized in SQL query

**Location:** `app.py`, line 42

**Suggested Fix:**
```python
# Before
cursor.execute(f"SELECT * FROM users WHERE id = {user_id}")

# After
cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))
```

_Confidence: 95%_

---
_Analysis powered by LLM Code Reviewer_
```

## API Reference

### Health Check

```bash
GET /health
```

Returns: `{"status": "healthy"}`

### GitHub Webhook Endpoint

```bash
POST /webhook/github
```

**Headers Required:**
- `X-GitHub-Event: pull_request`
- `X-Hub-Signature-256: sha256=...`
- `Content-Type: application/json`

**Supported Events:**
- `opened` - Triggered when PR is created
- `synchronize` - Triggered when PR is updated with new commits
- `closed` - Ignored (no analysis needed)

## Architecture

### System Design

```
┌─────────────────┐
│  GitHub Event   │
└────────┬────────┘
         │ (webhook)
         ▼
┌─────────────────────────────┐
│   Webhook Handler           │
│ - Verify HMAC signature     │
│ - Parse PR metadata         │
│ - Create Review record      │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Review Service             │
│ - Clone repository          │
│ - Extract diff              │
│ - Analyze code              │
│ - Store findings            │
│ - Post PR comment           │
└────────┬────────────────────┘
         │
    ┌────┴─────┐
    ▼          ▼
┌────────┐  ┌──────────┐
│Database│  │GitHub PR │
│Audit   │  │Comments  │
└────────┘  └──────────┘
```

### Core Components

**1. Webhook Handler** (`src/webhooks/github.py`)
- Verifies GitHub webhook signatures using HMAC-SHA256
- Parses pull request metadata
- Creates Review database records
- Returns 202 Accepted for async processing

**2. Diff Parser** (`src/analysis/diff_parser.py`)
- Parses unified diff format from `git diff`
- Extracts file changes with line numbers
- Filters code files (supports 15+ languages)
- Skips test files and documentation

**3. Code Analyzer** (`src/analysis/analyzer.py`)
- Sends code changes to LLM for analysis
- Runs security and performance checks in parallel
- Deduplicates findings by (category, file, line, title)
- Sorts by severity (CRITICAL → HIGH → MEDIUM → LOW)
- Validates LLM responses with JSON schema

**4. LLM Providers** (`src/llm/`)
- **ClaudeProvider**: Uses Anthropic API with streaming
- **OllamaProvider**: Uses local Ollama instance
- Abstract interface for easy provider switching
- Timeout handling and error recovery

**5. GitHub API Client** (`src/integrations/github_api.py`)
- Posts formatted PR comments
- Handles GitHub rate limiting
- Supports HTTPS and SSH repository URLs
- Graceful error handling

**6. Review Service** (`src/review_service.py`)
- Orchestrates the complete pipeline
- Clones repository with shallow clone (--depth 1)
- Extracts diffs between main and feature branch
- Stores findings in SQLite database
- Updates review status (PENDING → ANALYZING → COMPLETED)
- Handles errors and cleanup

**7. Database Models** (`src/database.py`)
- Review: Tracks PR reviews with status
- Finding: Stores individual issues found
- ReviewStatus: PENDING, ANALYZING, COMPLETED, FAILED
- FindingSeverity: CRITICAL, HIGH, MEDIUM, LOW
- FindingCategory: SECURITY, PERFORMANCE

## Testing

### Run All Tests

```bash
cd codeproject
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=src --cov-report=html
open htmlcov/index.html
```

### Test Coverage

- **Overall**: 91% coverage
- **Components**:
  - github_api.py: 99%
  - webhooks/github.py: 100%
  - llm/claude.py: 100%
  - analysis/analyzer.py: 87%
  - All others: 93-96%

### Test Categories

**Integration Tests** (17 tests)
- Application startup and health checks
- Webhook to Review workflow
- Database operations and persistence
- Service integration and initialization
- End-to-end review processing
- Configuration management
- Error handling
- Performance tests

**Unit Tests** (231 tests)
- LLM provider functionality (33 tests)
- Webhook security and parsing (23 tests)
- Diff parser edge cases (36 tests)
- Code analyzer deduplication (33 tests)
- GitHub API integration (36 tests)
- Review service orchestration (25 tests)
- Database models (18 tests)
- Configuration (18 tests)

### Running Specific Tests

```bash
# Test only webhook security
pytest tests/test_webhook_github.py::TestSignatureVerification -v

# Test only GitHub API integration
pytest tests/test_github_api.py -v

# Test only analyzer
pytest tests/test_analyzer.py -v

# Test only integration
pytest tests/test_integration.py -v
```

## Development Guide

### Code Quality

**Run Formatters**

```bash
# Format code with Black
python -m black src/ tests/

# Check formatting
python -m black --check src/ tests/
```

**Run Linters**

```bash
# Check with ruff
python -m ruff check src/ tests/

# Fix issues automatically
python -m ruff check src/ tests/ --fix
```

**Type Checking** (optional)

```bash
mypy src/
```

### Adding a New Feature

1. **Write tests first** (test-driven development)
   ```bash
   # Add test in tests/test_*.py
   ```

2. **Implement feature** in corresponding module

3. **Run tests**
   ```bash
   pytest tests/ -v
   ```

4. **Format and lint**
   ```bash
   black src/
   ruff check src/ --fix
   ```

5. **Verify coverage**
   ```bash
   pytest tests/ --cov=src --cov-report=term-missing
   ```

### Project Structure

```
codeproject/
├── src/                          # Source code
│   ├── __init__.py
│   ├── main.py                   # FastAPI app, endpoints
│   ├── config.py                 # Configuration (pydantic-settings)
│   ├── database.py               # ORM models (SQLAlchemy)
│   │
│   ├── webhooks/
│   │   └── github.py             # GitHub webhook handler
│   │
│   ├── analysis/
│   │   ├── diff_parser.py        # Git diff parsing
│   │   └── analyzer.py           # LLM-based code analysis
│   │
│   ├── llm/
│   │   ├── provider.py           # Abstract LLM interface
│   │   ├── claude.py             # Claude/Anthropic provider
│   │   └── ollama.py             # Ollama/local models provider
│   │
│   ├── integrations/
│   │   └── github_api.py         # GitHub API client
│   │
│   └── review_service.py         # Orchestration service
│
├── tests/                        # Test suite (248 tests)
│   ├── test_*.py                 # Unit tests by component
│   ├── test_integration.py       # Integration tests
│   ├── conftest.py               # Pytest fixtures
│   └── ...
│
├── pyproject.toml               # Package configuration
├── Dockerfile                   # Docker image definition
├── docker-compose.yml           # Docker Compose setup
├── .env.example                 # Environment variables template
├── .dockerignore                # Docker build exclusions
└── README.md                    # This file
```

### Adding Support for a New LLM Provider

1. **Create provider class** in `src/llm/newprovider.py`:

```python
from src.llm.provider import LLMProvider

class NewProvider(LLMProvider):
    def __init__(self, config):
        self.config = config

    def analyze_security(self, code_snippet: str) -> dict:
        # Your implementation
        pass

    def analyze_performance(self, code_snippet: str) -> dict:
        # Your implementation
        pass
```

2. **Register in factory** (`src/llm/provider.py`):

```python
def get_llm_provider() -> LLMProvider:
    if settings.llm_provider == "newprovider":
        return NewProvider(settings)
    # ... existing providers
```

3. **Add tests** in `tests/test_llm_provider.py`

4. **Update documentation** and `.env.example`

## Troubleshooting

### Docker Daemon Not Running

```bash
# Error: Cannot connect to the Docker daemon
# Solution: Start Docker Desktop or Docker service

# On macOS
open /Applications/Docker.app

# On Linux
sudo systemctl start docker
```

### Import Errors

```bash
# Error: ModuleNotFoundError: No module named 'src'
# Solution: Install package in development mode

cd codeproject
pip install -e .
```

### GitHub Token Issues

```bash
# Error: "GitHub token not configured"
# Solution: Set GITHUB_TOKEN in .env

export GITHUB_TOKEN=ghp_your_token
# Or add to .env file
```

### LLM Provider Connection Issues

```bash
# Error: "Claude API timeout" or "Ollama connection refused"
# Solution: Check credentials and service status

# Test Claude
python -c "from anthropic import Anthropic; print('Claude OK')"

# Test Ollama
curl http://localhost:11434/api/tags
```

### Webhook Secret Mismatch

```bash
# Error: "Webhook signature verification failed"
# Solution: Verify WEBHOOK_SECRET matches GitHub settings

# Get your webhook secret from GitHub Settings > Webhooks
# Update .env: WEBHOOK_SECRET=your-actual-secret
```

## Performance Metrics

Based on the test suite:

| Operation | Time | Notes |
|-----------|------|-------|
| Repository clone (shallow) | ~2-5s | Depends on network and repo size |
| Diff extraction | ~1s | Git diff operation |
| Code analysis | ~2-5s | LLM API call + parsing |
| Database operations | ~10-50ms | SQLite writes |
| Full pipeline | ~10-15s | End-to-end for typical PR |

Bulk operations tested:
- 100 findings creation: <5 seconds
- 248 test suite execution: <1.5 seconds

## Security Considerations

✅ **Implemented Security**

1. **Webhook Verification**
   - HMAC-SHA256 signature verification
   - Constant-time comparison (prevents timing attacks)
   - Validates X-Hub-Signature-256 header

2. **Credential Management**
   - API keys stored in environment variables (not in code)
   - Never logged or exposed in error messages
   - `.env` file excluded from version control

3. **Database Security**
   - SQLAlchemy ORM prevents SQL injection
   - Foreign key constraints enforced
   - No direct SQL queries

4. **Repository Access**
   - GitHub tokens with minimal required permissions
   - Shallow clones limit data transfer
   - Automatic cleanup of temporary directories

### Recommendations for Production

1. **Use PostgreSQL** instead of SQLite for scalability
2. **Run in private VPC** with restricted network access
3. **Rotate API keys** regularly
4. **Monitor webhook failures** and retry logic
5. **Rate limit** webhook processing per repository
6. **Use HTTPS** for all webhook communications
7. **Implement authentication** for health/status endpoints if exposed

## License

MIT

## Support

For issues or questions:

1. Check the [design document](../docs/plans/2025-11-15-security-performance-code-review-design.md)
2. Review [implementation plan](../docs/plans/2025-11-15-security-performance-code-review-implementation.md)
3. Open a GitHub issue

---

**Status**: Phase 1 MVP Complete ✅
- 248 tests passing (91% coverage)
- All 10 core features implemented
- Production-ready with Docker support
- Comprehensive documentation included
