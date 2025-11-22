# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Code Review System** - LLM-powered CI/CD code review analyzing security vulnerabilities and performance issues in pull requests.

**Tech Stack**: Python 3.11+ | FastAPI | SQLAlchemy | SQLite/PostgreSQL | Claude API/Ollama/OpenRouter

**Key Architecture**:
- Webhook handler receives GitHub PR events
- Parallel security & performance analysis via LLM
- Database persistence of findings and review history
- PR comment posting with formatted results

## Common Development Commands

### Setup & Installation

```bash
cd codeproject
python3 -m venv venv
source venv/bin/activate
pip install -e .
pip install -e ".[dev]"  # Development tools (black, ruff, mypy)
cp .env.example .env
# Edit .env with API keys
```

### Running the Service

```bash
# Development server with auto-reload
python -m uvicorn src.main:app --reload

# Production server
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000

# Docker Compose
docker-compose up -d
```

### Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src

# Run single test file
pytest tests/test_analyzer.py -v

# Run single test
pytest tests/test_analyzer.py::test_analyzer_initialization -v

# Run with coverage report
pytest tests/ --cov=src --cov-report=html  # Creates htmlcov/index.html
```

### Code Quality

```bash
# Format code
black codeproject/src

# Lint
ruff check src

# Type checking (optional)
mypy src
```

### Database & Migrations

```bash
# View database schema
sqlite3 codeproject.db ".schema"

# Run migrations
python -m alembic upgrade head

# Check migration status
python -m alembic current
```

## Project Structure

```
codeproject/
├── src/
│   ├── main.py                    # FastAPI app, webhook endpoint
│   ├── config.py                  # Settings management (pydantic)
│   ├── database.py                # SQLAlchemy models (Review, Finding, etc)
│   ├── review_service.py          # Orchestration: clone→diff→analyze→comment
│   ├── analysis/
│   │   ├── analyzer.py            # LLM orchestration (security + performance)
│   │   ├── diff_parser.py         # Unified diff parsing
│   │   ├── adaptive_severity.py   # Severity scoring
│   │   ├── context_builder.py     # Code context extraction
│   │   └── antipatterns.py        # Pattern detection
│   ├── llm/
│   │   ├── provider.py            # Abstract LLMProvider interface
│   │   ├── claude.py              # Claude API implementation
│   │   ├── ollama.py              # Local Ollama implementation
│   │   ├── openrouter.py          # OpenRouter multi-model support
│   │   └── enhanced_prompts.py    # Prompt templates
│   ├── suggestions/
│   │   ├── enrichment.py          # Add suggestions to findings
│   │   ├── validators.py          # Security validation for patches
│   │   └── cache.py               # SQLite-based caching
│   ├── integrations/
│   │   └── github_api.py          # Post PR comments, parse repos
│   ├── webhooks/
│   │   └── github.py              # GitHub webhook handler
│   ├── learning/                  # Feedback collection & ML
│   ├── prediction/                # Risk scoring, pattern learning
│   ├── architecture/              # Anti-pattern detection
│   ├── tools/                     # External tool integration (bandit, mypy)
│   ├── feedback/                  # User feedback collection
│   ├── metrics/                   # Metrics collection
│   └── reporting/                 # Report generation
├── tests/                         # 25+ test files, 84%+ coverage
├── migrations/                    # Database schema migrations
├── ARCHITECTURE.md                # Detailed technical design
├── USAGE_GUIDE.md                 # Practical examples
├── README.md                      # Quick start & feature overview
└── pyproject.toml                 # Dependencies & project metadata
```

## Critical Architecture Patterns

### 1. Data Flow: Request → Database → Analysis → Comment

```
GitHub Webhook (POST /webhook/github)
    ↓
Signature Verification (HMAC-SHA256)
    ↓
Create Review (PENDING) → Return 202 immediately
    ↓
[Background Processing]
    ↓
Clone Repo (shallow, --depth 1)
    ↓
Extract Diff (git diff origin/main...HEAD)
    ↓
Parse Unified Diff → FileDiff objects
    ↓
Parallel LLM Analysis (Security + Performance)
    ↓
Deduplicate Findings (by category, file, line, title)
    ↓
Store Finding records + Update Review (COMPLETED)
    ↓
Post PR Comment (non-critical: continues if fails)
```

### 2. LLM Provider Factory Pattern

Switchable providers via `settings.llm_provider` ("claude", "ollama", "openrouter"):

```python
def get_llm_provider() -> LLMProvider:
    if settings.llm_provider == "claude":
        return ClaudeProvider()
    elif settings.llm_provider == "ollama":
        return OllamaProvider(settings.ollama_base_url)
    else:
        return OpenRouterProvider(settings.openrouter_api_key)
```

**Interface**: All providers implement `analyze_security(code: str) → dict` and `analyze_performance(code: str) → dict`.

### 3. Database Models & Relationships

**Review** (1) → (Many) **Finding**
- ReviewStatus: PENDING → ANALYZING → COMPLETED/FAILED
- Cascade delete: removing Review removes all Findings
- Indexes on review_id, severity, repo_url for queries

### 4. Deduplication Strategy

Findings are deduplicated by composite key:
```python
dedup_key = (category, file_path, line_number, title)
```
Keeps highest confidence score when duplicates exist.

### 5. Suggestion Generation & Validation

**Cache**: SQLite-based (50-80% hit rate, TTL-based eviction)
**Validation**: No exec/eval/dangerous operations, confidence threshold ≥0.8
**Integration**: Enriches findings with `.suggested_fix` field before posting

## Important Implementation Details

### Configuration Management

Settings loaded from `.env` file via Pydantic:
- `LLM_PROVIDER`: Provider selection (default: "claude")
- `CLAUDE_API_KEY`: API key for Claude
- `WEBHOOK_SECRET`: HMAC-SHA256 verification secret
- `GITHUB_TOKEN`: For posting PR comments
- `DATABASE_URL`: SQLite or PostgreSQL (default: sqlite:///./codeproject.db)
- `cache_suggestions`: Enable/disable suggestion caching
- `suggestion_cache_ttl_days`: Cache expiration (default: 7 days)

### Error Handling Strategy

**Network Errors**: Logged, review marked FAILED, cleanup happens in finally block
**Parsing Errors**: Logged as warnings, continue with empty results (graceful degradation)
**Authorization Errors**: Logged, fail loudly without retry
**Database Errors**: Transaction rollback, log error

### Testing Patterns

- **Unit Tests**: Mock LLM providers, GitHub API, external services
- **Integration Tests**: Real in-memory SQLite, full pipeline
- **Coverage Target**: ≥80% unit, ≥70% integration
- **Mock Strategy**: Use pytest-mock, patch external calls, use fixtures

```python
# Example: Mocking Claude API
@patch("src.llm.claude.Anthropic")
def test_security_analysis(mock_api):
    mock_api.return_value.messages.create.return_value = {
        "content": [{"text": '{"findings": [...]}'}]
    }
```

### Security Considerations

1. **Webhook Verification**: Constant-time HMAC-SHA256 comparison (`hmac.compare_digest`)
2. **API Key Management**: Environment variables only, never logged
3. **SQL Injection**: SQLAlchemy ORM prevents parameterization issues
4. **Diff Parsing**: Handles Unicode, special characters, large files safely
5. **Repository Cloning**: Shallow clone with timeout (60s) to prevent DoS

## Key Files by Domain

**Webhook & Entry Point**
- `src/main.py` - FastAPI app, `/health` and `/webhook/github` endpoints
- `src/webhooks/github.py` - Signature verification, payload parsing

**Core Analysis Pipeline**
- `src/review_service.py` - Orchestrates full pipeline
- `src/analysis/analyzer.py` - LLM orchestration (security + perf parallel)
- `src/analysis/diff_parser.py` - Unified diff parsing

**LLM Integration**
- `src/llm/provider.py` - Abstract base class
- `src/llm/claude.py` - Claude API client
- `src/llm/ollama.py` - Local Ollama client
- `src/llm/openrouter.py` - OpenRouter multi-model client

**Database & Persistence**
- `src/database.py` - SQLAlchemy models (Review, Finding)
- `src/config.py` - Settings management

**Advanced Features**
- `src/suggestions/cache.py` - SQLite suggestion caching
- `src/suggestions/enrichment.py` - Add suggestions to findings
- `src/suggestions/validators.py` - Security validation for patches
- `src/architecture/` - Anti-pattern detection, cascade analysis
- `src/prediction/` - Risk scoring, failure prediction
- `src/learning/` - Feedback collection, pattern learning

## Testing Quick Reference

**Run all tests**: `pytest tests/ -v --cov=src`
**Run single module**: `pytest tests/test_analyzer.py -v`
**Run single test**: `pytest tests/test_analyzer.py::test_name -v`
**View coverage gaps**: `pytest tests/ --cov=src --cov-report=html`

**Common test files**:
- `tests/test_analyzer.py` - Core analysis pipeline
- `tests/test_diff_parser.py` - Diff parsing edge cases
- `tests/test_suggestion_enrichment.py` - Suggestion generation
- `tests/test_suggestion_cache.py` - Caching behavior
- `tests/test_llm_provider.py` - LLM provider interfaces

## Dependencies Reference

**Core**: FastAPI, Uvicorn, Pydantic, SQLAlchemy, GitPython
**LLM**: anthropic, requests (for Ollama/OpenRouter)
**Testing**: pytest, pytest-cov, pytest-asyncio
**Dev**: black, ruff, mypy (optional)

See `pyproject.toml` for exact versions and optional dependencies.

## Common Gotchas & Solutions

**1. Signature Verification Fails**
- Ensure `WEBHOOK_SECRET` matches GitHub webhook secret exactly
- Check `X-Hub-Signature-256` header format: `sha256=...`
- Use constant-time comparison

**2. LLM Response Parsing**
- LLM may return malformed JSON → log warning, continue with empty findings
- Missing fields handled gracefully → required fields only (title, description, severity)
- Always validate response schema before use

**3. Repository Operations**
- Shallow clone (--depth 1) prevents large history issues
- Timeout (60s) prevents hanging on large repos
- Cleanup always happens (finally block) even if analysis fails

**4. Suggestion Caching**
- Cache key combines: code_snippet + analysis_type (security/performance)
- TTL-based eviction (default: 7 days)
- Hit rate typically 50-80% for repeated patterns

**5. Finding Deduplication**
- Composite key: (category, file_path, line_number, title)
- Same issue at same line = deduplicated
- Different categories = both stored

## Database Queries (SQLAlchemy)

```python
# All findings for a review
findings = db.query(Finding).filter(Finding.review_id == review_id).all()

# Critical findings only
critical = db.query(Finding).filter(
    Finding.review_id == review_id,
    Finding.severity == FindingSeverity.CRITICAL
).all()

# Reviews by repository
reviews = db.query(Review).filter(
    Review.repo_url == "https://github.com/user/repo.git"
).order_by(Review.created_at.desc()).all()

# Findings by category
security = db.query(Finding).filter(
    Finding.category == FindingCategory.SECURITY
).all()
```

## Recent Development Notes

**Phase 4 (Completed)**: AI-generated suggestions with caching, validation, and enrichment
**Phase 5 (In Progress)**: Feedback collection, learning, and pattern analysis
**Future**: Trend analysis, PostgreSQL migration, advanced analytics dashboard

Current branch: `task-5-1-feedback-collection` (feedback infrastructure)

See `ARCHITECTURE.md` for detailed design and `IMPLEMENTATION_ROADMAP.md` for phase breakdown.
