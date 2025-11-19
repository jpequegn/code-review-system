# AI-Generated Suggestions System

## Overview

The Suggestions System provides intelligent, context-aware recommendations for fixing code security issues discovered by the Code Review System. It integrates with multiple LLM providers to generate:

- **Auto-fixes**: Safe, tested code patches (for High/Critical findings)
- **Explanations**: Educational explanations of security issues and impact
- **Improvement Suggestions**: Best practice recommendations for preventing similar issues

## Architecture

### Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Findings Stream                          │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          Enrichment Pipeline (enrichment.py)                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 1. Generate Suggestions (severity-based logic)      │   │
│  │    - Severity HIGH/CRITICAL: all 3 types           │   │
│  │    - Severity MEDIUM/LOW: explanation only         │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ 2. Cache Check (improve performance)               │   │
│  │    - Check for identical finding in cache          │   │
│  │    - Return cached suggestions if available        │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ 3. Call LLM (if cache miss)                        │   │
│  │    - Generate explanation, auto-fix, improvements  │   │
│  │    - Track generation time and tokens             │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ 4. Validate Suggestions                            │   │
│  │    - Syntax validation (ast.parse)                │   │
│  │    - Confidence threshold (minimum 0.8)           │   │
│  │    - Security checks (no exec, eval, etc)         │   │
│  ├──────────────────────────────────────────────────────┤   │
│  │ 5. Cache Storage                                   │   │
│  │    - Store validated suggestions in SQLite cache   │   │
│  │    - TTL: 7 days (configurable)                   │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              Enhanced Findings with Suggestions             │
└─────────────────────────────────────────────────────────────┘
```

### Key Modules

**enrichment.py** (171 LOC, 82% coverage)
- Main enrichment engine
- Severity-based suggestion logic
- Integration with cache and validators
- Graceful degradation (findings without suggestions are still valid)

**cache.py** (133 LOC, 82% coverage)
- SQLite-based suggestion cache
- Cache key generation (SHA256 hash)
- TTL management and expiration
- Hit/miss tracking for monitoring

**validators.py** (72 LOC, 94% coverage)
- Python syntax validation using ast.parse()
- Confidence threshold validation
- Security pattern detection
- Comprehensive validation compositing

## Severity-Based Behavior

### CRITICAL / HIGH Severity

All three suggestion types generated:
- **Auto-fix**: Code patch with confidence score (≥0.8)
- **Explanation**: Educational context about the vulnerability
- **Improvements**: Best practice recommendations

```python
# Example: SQL Injection vulnerability
Finding: "SQL Injection Vulnerability"
Auto-fix: "user = db.query(User).filter(User.id == user_id).first()"
Explanation: "User input directly concatenated into database queries can allow attackers to extract data or modify the database."
Improvements:
  - Use parameterized queries/ORM
  - Validate and sanitize all user inputs
  - Implement least-privilege database permissions
```

### MEDIUM / LOW Severity

Explanation only (no auto-fix):
- Security findings still get context
- Auto-fixes only for higher severity issues
- Reduces risk of incorrect suggestions

```python
# Example: Weak cryptography
Finding: "Weak Hashing Algorithm"
Explanation: "MD5 is deprecated and vulnerable to collision attacks. Use SHA-256 or bcrypt for password hashing."
Improvements:
  - Use bcrypt/scrypt for passwords
  - Use SHA-256+ for general hashing
```

## Caching Strategy

### Cache Key Generation

Cache keys are SHA256 hashes of:
- Finding title
- File path
- Code snippet (first 500 chars)

```python
cache_key = SHA256(
    f"{title}:{file_path}:{code_snippet[:500]}"
)
```

### Performance Impact

**Without Cache**:
- Average generation: 2-5 seconds per finding
- Multiple identical findings: 2-5s each

**With Cache**:
- Cache hit: <50ms
- Cache miss: ~2-5s (then stored)
- Same finding reoccurs: 50ms + stored result

**Real-world Impact**:
- Repositories with duplicate issues: 50-80% cache hit rate
- Typical PR: 3-8 findings (1-2 duplicates) → 60% reduction in LLM calls

## Validation Framework

### Python Syntax Validation

Uses `ast.parse()` to verify syntactic correctness:

```python
✅ Valid:
- x = user_input.strip()
- if value > 0: return value * 2
- def safe_func(): pass

❌ Invalid:
- if True\n x = 1 (bad indentation)
- x = 1 @@ 2 (invalid operator)
- open("file.txt" (unclosed paren)
```

### Confidence Threshold

Auto-fixes with confidence < 0.8 are rejected:

```python
Suggestion: "x = unsafe_code()"
Confidence: 0.75
Result: ❌ Rejected (below 0.8 threshold)

Suggestion: "x = user_input.strip()"
Confidence: 0.92
Result: ✅ Accepted (above threshold)
```

### Security Validation

Dangerous operations are blocked:

```
Blocked Operations:
- exec(), eval(), __import__()     → Remote code execution
- compile()                         → Code generation
- globals(), locals(), vars()       → Introspection abuse
- getattr(), setattr(), delattr()  → Attribute manipulation
- os.system(), os.popen()          → Shell command injection
- subprocess with shell=True       → Command injection
```

## Configuration

### Environment Variables

```bash
# Enable/disable caching
CACHE_SUGGESTIONS=true              # Default: true

# Cache TTL in days
SUGGESTION_CACHE_TTL_DAYS=7         # Default: 7

# LLM Provider
LLM_PROVIDER=claude                 # Options: claude, ollama, openrouter

# Provider-specific keys
CLAUDE_API_KEY=sk-...               # Required for Claude
OLLAMA_BASE_URL=http://localhost:11434  # Default for Ollama
OPENROUTER_API_KEY=sk-...          # Required for OpenRouter
```

### Programmatic Configuration

```python
from src.config import settings

# Check cache status
print(f"Cache enabled: {settings.cache_suggestions}")
print(f"Cache TTL: {settings.suggestion_cache_ttl_days} days")

# Check LLM provider
print(f"LLM: {settings.llm_provider}")

# Validate settings
settings.validate_llm_provider(settings.llm_provider)
```

## Testing Coverage

### Test Statistics

Total tests: 546
Suggestion-specific tests: 139 (25%)
Coverage: 84% (target: 85%+)

### Test Categories

**Validators (63 tests, 94% coverage)**
- Syntax validation: 12 tests
- Confidence validation: 10 tests
- Security validation: 16 tests
- Composite validation: 13 tests
- Edge cases: 12 tests

**Cache (26 tests, 82% coverage)**
- Cache operations: 11 tests
- TTL/expiration: 2 tests
- Statistics: 3 tests
- Cache disabled mode: 2 tests
- Key generation: 5 tests
- Edge cases: 3 tests

**Enrichment (30 tests, 82% coverage)**
- Severity-based logic: 5 tests
- Confidence filtering: 2 tests
- Graceful degradation: 4 tests
- Multiple findings: 3 tests
- Cache integration: 5 tests
- Error handling: 11 tests

**Suggestions (20 tests)**
- LLM provider integration: 12 tests
- Suggestion generation: 8 tests

## Usage Examples

### Basic Enrichment

```python
from src.suggestions import enrich_findings_with_suggestions
from src.llm.provider import LLMProvider

# Prepare findings from code analysis
findings = analyzer.analyze(code_diff)

# Enrich with suggestions
llm_provider = LLMProvider(provider="claude")
enriched = enrich_findings_with_suggestions(
    findings=findings,
    code_diff=code_diff,
    llm_provider=llm_provider,
    dry_run=False  # Set True to skip LLM calls
)

# Access suggestions
for finding in enriched:
    if finding.suggestions.has_suggestions():
        print(f"Finding: {finding.title}")
        print(f"  Explanation: {finding.suggestions.explanation}")
        if finding.suggestions.auto_fix:
            print(f"  Fix: {finding.suggestions.auto_fix}")
```

### Cache Management

```python
from src.suggestions.cache import get_cache

cache = get_cache()

# Check cache status
stats = cache.get_stats()
print(f"Cache entries: {stats['total_entries']}")
print(f"Cache hits: {stats['total_hits']}")
print(f"Cache size: {stats['size_mb']} MB")

# Clear cache
cache.clear_all()

# Clear expired entries
deleted = cache.clear_expired()
print(f"Deleted {deleted} expired entries")
```

### Validation

```python
from src.suggestions.validators import validate_auto_fix

code = "x = user_input.strip()"
confidence = 0.92

is_valid, errors = validate_auto_fix(
    code,
    confidence=confidence,
    syntax_required=True,
    confidence_threshold=0.8,
    security_required=True
)

if not is_valid:
    print(f"Validation failed: {errors}")
else:
    print("Suggestion passed all validations")
```

## Performance Characteristics

### Latency

```
Metric              | Without Cache | With Cache (Hit) | With Cache (Miss)
--------------------|---------------|-----------------|------------------
Single finding      | 2.5s          | 45ms            | 2.5s
10 findings (0 dup) | 25s           | 450ms           | 25s
10 findings (50% dup)| 25s           | 1.3s            | ~13s
100 findings (80% dup)| 250s         | 2.5s            | ~52s
```

### Storage

```
Cache Entry Size:     ~500 bytes (typical)
Max Entries (100MB):  ~200,000 entries
DB Size Growth:       ~500 bytes per unique finding
Cleanup (7d TTL):     Automatic, runs on cache.get()
```

### LLM Token Usage

```
Per Finding Type:    Approx Tokens
Explanation:         150-300 tokens
Auto-fix:            200-500 tokens
Improvements:        100-200 tokens
Total per finding:   450-1000 tokens

With Cache (50% hit):
100 findings:        45-100K tokens (vs 90-200K without cache)
```

## Error Handling

### Graceful Degradation

The system implements graceful degradation at every layer:

```python
# If explanation generation fails:
Finding → Returned with auto_fix + improvements only

# If auto_fix generation fails:
Finding → Returned with explanation + improvements only

# If all suggestion generation fails:
Finding → Returned without suggestions (still valid)
```

### Common Error Scenarios

| Scenario | Handling |
|----------|----------|
| LLM timeout | Skip that suggestion type, continue with others |
| Invalid JSON response | Log error, reject that suggestion |
| Syntax validation fails | Log error, reject auto_fix |
| Security check fails | Log error, reject auto_fix |
| Cache database error | Log error, skip cache (still generate) |

## Future Enhancements

1. **Multi-language Support**: Extend validators to Java, TypeScript, Go
2. **Batch Optimization**: Process multiple findings in parallel
3. **Learning Feedback**: Track which suggestions are most helpful
4. **Confidence Tuning**: Adjust thresholds based on accuracy metrics
5. **Custom Validators**: Allow teams to add domain-specific validators
6. **Caching Strategy**: Implement LRU cache with memory limits

## Troubleshooting

### Cache Not Working

```bash
# Check if cache is enabled
echo $CACHE_SUGGESTIONS  # Should be true

# Check cache directory
ls -la codeproject/suggestions_cache.db

# Clear cache and rebuild
rm codeproject/suggestions_cache.db
python3 -m pytest tests/test_suggestion_cache.py -v
```

### Low Confidence Scores

If most auto-fixes are rejected due to low confidence:

1. Check LLM model (some models are more conservative)
2. Review the code snippets being analyzed
3. Consider adjusting `SUGGESTION_CONFIDENCE_THRESHOLD` (carefully)
4. Check LLM logs for generation issues

### Validation Failures

If suggestions are being rejected by validators:

1. Run syntax validator directly on the code
2. Check security patterns for false positives
3. Review LLM output format
4. Enable debug logging: `LOG_LEVEL=DEBUG`

## References

- [SQLite Documentation](https://www.sqlite.org/docs.html)
- [Python AST Module](https://docs.python.org/3/library/ast.html)
- [Pydantic Settings](https://docs.pydantic.dev/latest/api/pydantic_settings/)
- [Claude API](https://docs.anthropic.com/)
- [Ollama](https://ollama.ai/)
