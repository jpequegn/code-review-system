# Code Review System

LLM-powered CI/CD code review service analyzing **security vulnerabilities** and **performance/scalability issues** in pull requests.

## Overview

This service integrates with Git platforms (GitHub, GitLab, etc.) via webhooks to automatically analyze code changes using Claude AI or local LLM models. It:

- **Detects security vulnerabilities** (auth flaws, injection risks, hardcoded secrets, unsafe patterns)
- **Identifies performance issues** (algorithm inefficiencies, resource leaks, architectural patterns)
- **Posts PR comments** with findings and suggested fixes
- **Maintains audit trail** of all reviews for trend analysis and pattern detection
- **Works with pluggable LLM backends** (Claude API or local models via Ollama)

## Quick Start

### Prerequisites
- Python 3.11+
- Git
- Docker & Docker Compose (optional, for containerized deployment)
- Claude API key (for Claude backend) or Ollama (for local models)

### Installation

```bash
# Clone the repository
git clone https://github.com/jpequegn/code-review-system.git
cd code-review-system

# Set up environment
cp codeproject/.env.example codeproject/.env
# Edit .env with your API keys and configuration

# Install dependencies (Python)
cd codeproject
python3 -m venv venv
source venv/bin/activate
pip install -e .

# Or use Docker
docker-compose up
```

## Architecture

### Components

1. **Webhook Ingestion** - Receives Git platform webhook events
2. **Diff Parser** - Extracts and filters code changes
3. **Dual LLM Analyzer** - Runs security and performance analysis in parallel
4. **Findings Engine** - Correlates and deduplicates findings
5. **CI Integration** - Posts PR comments and generates reports
6. **Audit Database** - Stores review history and findings

### Data Flow

```
Git Webhook → Diff Extraction → LLM Analysis (Security + Performance)
→ Findings Correlation → PR Comments + Report → Audit Trail
```

## API Endpoints

- `POST /webhook/github` - GitHub webhook endpoint
- `GET /health` - Health check

## Configuration

See `codeproject/.env.example` for all configuration options:

- `LLM_PROVIDER` - Which LLM to use (claude or ollama)
- `CLAUDE_API_KEY` - Claude API key (if using Claude)
- `WEBHOOK_SECRET` - GitHub webhook secret for signature verification
- `GITHUB_TOKEN` - GitHub token for posting comments
- `DATABASE_URL` - Database connection string

## Development

### Running Tests

```bash
cd codeproject
pytest tests/ -v --cov=src
```

### Running the Service

```bash
# Development mode
cd codeproject
python -m uvicorn src.main:app --reload

# Production via Docker
docker-compose up -d
```

### Implementation Plan

The detailed implementation plan is in `docs/plans/2025-11-15-security-performance-code-review-implementation.md`. It breaks development into 10 bite-sized tasks following Test-Driven Development (TDD) principles.

## Design Document

See `docs/plans/2025-11-15-security-performance-code-review-design.md` for the complete system design, architecture, error handling, and testing strategy.

## Features (MVP Roadmap)

**Phase 1: MVP**
- ✓ GitHub webhook integration
- ✓ Security analysis (SQL injection, auth flaws, secrets)
- ✓ Performance analysis (algorithm complexity, resource leaks)
- ✓ PR comments with findings
- ✓ SQLite audit trail

**Phase 2: Optimization**
- Dual analysis optimization (batch requests)
- Confidence scoring
- Better prompt engineering

**Phase 3: Advanced Analytics**
- PostgreSQL migration
- Historical trend analysis
- HTML/JSON reports
- Analytics dashboard

**Phase 4: Multi-Platform**
- GitLab support
- Gitea support
- Generic Git server support

## Status

🔨 **In Development** - Core architecture and planning phase complete. Implementation starting with Phase 1 MVP.

## License

MIT

## Support

For issues, questions, or suggestions, please open a GitHub issue.
