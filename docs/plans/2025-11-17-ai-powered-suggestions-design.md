# Phase 4: AI-Powered Suggestions - Design Document

**Date**: 2025-11-17
**Phase**: 4
**Status**: Design Complete, Ready for Implementation

## Overview

Phase 4 adds comprehensive AI-powered suggestions to the code review system, providing three levels of guidance for every finding:

1. **Auto-fix** (high/critical only) - Safe, runnable code patches
2. **Explanation** (all severities) - Educational context on why the issue matters
3. **Improvement suggestions** (high/critical only) - Best practices and optimization ideas

This makes the system more actionable and helps developers learn from code reviews.

---

## Architecture

### LLM Provider Extension

Extend the existing `LLMProvider` interface with three new methods:

```python
class LLMProvider(ABC):
    # Existing methods
    @abstractmethod
    def analyze_security(self, code_diff: str) -> str: ...

    @abstractmethod
    def analyze_performance(self, code_diff: str) -> str: ...

    # NEW: Suggestion generation methods
    @abstractmethod
    def generate_auto_fix(self, finding: Dict, code_diff: str) -> str:
        """
        Generate safe, conservative auto-fix for high/critical findings.

        Returns JSON: {"auto_fix": "...", "confidence": 0.0-1.0}
        Confidence <0.8 indicates risky fix and should be rejected.
        """
        pass

    @abstractmethod
    def generate_explanation(self, finding: Dict, code_diff: str) -> str:
        """
        Generate concise, educational explanation of the issue.
        Educational and actionable, 2-3 sentences.
        """
        pass

    @abstractmethod
    def generate_improvement_suggestions(self, finding: Dict, code_diff: str) -> str:
        """
        Generate 2-3 best practices or optimization ideas.
        For high/critical findings only.
        """
        pass
```

### Data Model Extension

Add three optional fields to the `Finding` model:

```python
class Finding(Base):
    # Existing fields
    id: int
    review_id: int
    category: str  # "security", "performance", "code_quality"
    severity: str  # "critical", "high", "medium", "low"
    title: str
    description: str
    file_path: str
    line_number: int
    suggested_fix: str
    created_at: datetime

    # NEW fields
    auto_fix: Optional[str] = None  # Only populated for high/critical
    explanation: Optional[str] = None  # All severities
    improvement_suggestions: Optional[str] = None  # Only for high/critical
```

### Orchestration Flow

After generating findings, enrich them synchronously with suggestions:

```python
def analyze_code(code_diff: str, context: Dict) -> List[Dict]:
    # Step 1: Generate findings (existing)
    security_findings = llm_provider.analyze_security(code_diff)
    performance_findings = llm_provider.analyze_performance(code_diff)

    all_findings = parse_findings(security_findings + performance_findings)

    # Step 2: Enrich with suggestions (NEW)
    enriched_findings = enrich_findings_with_suggestions(all_findings, code_diff)

    return enriched_findings
```

**Generation Rules**:
- **High/Critical**: Auto-fix + explanation + improvements
- **Medium**: Explanation only (auto-fix too risky, improvements not worth tokens)
- **Low**: Explanation only (optimization not prioritized)

---

## Enhanced Finding Response

**Example JSON response**:

```json
{
  "findings": [
    {
      "severity": "critical",
      "category": "security",
      "title": "SQL Injection Vulnerability",
      "description": "User input concatenated directly into SQL query without parameterization",
      "file_path": "app.py",
      "line_number": 42,
      "auto_fix": "cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))",
      "explanation": "SQL injection allows attackers to execute arbitrary database commands. Using parameterized queries (?) treats user input as data, not executable code, preventing this attack.",
      "improvement_suggestions": "Always use parameterized queries (?) instead of string concatenation. Consider using SQLAlchemy ORM for additional safety and consistency."
    },
    {
      "severity": "high",
      "category": "performance",
      "title": "N+1 Query Problem",
      "description": "Loop executes separate database query for each iteration",
      "file_path": "service.py",
      "line_number": 87,
      "auto_fix": "users = session.query(User).options(joinedload(User.posts)).all()",
      "explanation": "Each iteration queries the database separately, causing O(n) queries instead of O(1). Using joins fetches all data in a single query.",
      "improvement_suggestions": "Use eager loading with joinedload() or selectinload(). Cache results if queried multiple times. Consider pagination for large datasets."
    },
    {
      "severity": "medium",
      "category": "code_quality",
      "title": "Unused Import",
      "description": "Import 'os' is never used in this file",
      "file_path": "utils.py",
      "line_number": 3,
      "explanation": "Unused imports increase cognitive load and can mask missing dependencies.",
      "improvement_suggestions": null
    }
  ]
}
```

---

## Implementation Strategy

### Generation Methods

Each provider implements suggestion generation with specialized prompts:

**Claude Provider Example**:
```python
def generate_auto_fix(self, finding: Dict, code_diff: str) -> str:
    prompt = f"""Generate a SAFE, CONSERVATIVE fix for this code issue.

ISSUE: {finding['title']}
SEVERITY: {finding['severity']}
CATEGORY: {finding['category']}
FILE: {finding['file_path']}
LINE: {finding['line_number']}

CODE CONTEXT:
{code_diff}

REQUIREMENTS:
1. Fix ONLY the specific issue - no refactoring
2. Maintain existing code style
3. No new external dependencies
4. Valid, runnable Python code
5. Include confidence score (0.0-1.0)

Return JSON: {{"auto_fix": "...", "confidence": 0.95}}
"""
    response = self._call_claude(prompt)
    data = json.loads(response)

    # Validate confidence
    if data.get('confidence', 0) < 0.8:
        return json.dumps({"auto_fix": None, "confidence": data['confidence']})

    return response
```

### Safety & Validation

**Auto-fix Validation**:
- Python syntax check (ast.parse)
- Confidence score validation (reject <0.8)
- Security check: no dangerous operations (exec, eval, __import__)
- No file operations without explicit permission

**Graceful Degradation**:
- If suggestion generation fails: return finding without suggestions
- If LLM timeout: proceed with findings, skip suggestions
- If syntax invalid: skip that suggestion, return others
- No finding should be lost due to suggestion failure

### Caching

Cache identical suggestions to avoid duplicate LLM calls:

```python
# In src/suggestions/cache.py
class SuggestionCache:
    def get_or_generate(self, finding_hash: str, finding: Dict, code_diff: str):
        """Get cached suggestion or generate and cache it."""
        cache_key = hash(finding['title'] + finding['file_path'] + code_diff[:100])

        if cached := self.redis.get(cache_key):
            return cached

        suggestion = self.llm.generate_auto_fix(finding, code_diff)
        self.redis.set(cache_key, suggestion, ttl=7*24*3600)  # 1 week

        return suggestion
```

---

## Configuration

```python
# In src/config.py
class Settings:
    # Suggestion control
    enable_auto_fixes: bool = True
    enable_explanations: bool = True
    enable_improvements: bool = True

    # Safety thresholds
    auto_fix_confidence_threshold: float = 0.8
    suggestion_timeout_seconds: int = 10

    # Performance
    cache_suggestions: bool = True
    suggestion_cache_ttl_days: int = 7
```

---

## Testing Strategy

### Unit Tests

**Auto-fix Generation**:
- SQL injection → parameterized query
- Command injection → subprocess.run with list
- Hardcoded secrets → environment variables
- Invalid syntax rejection
- Confidence threshold validation

**Explanation Generation**:
- All severities tested
- Output length <500 chars
- Educational tone validated
- Actionable guidance present

**Improvement Suggestions**:
- 2-3 bullet points
- Best practices included
- Specific and non-generic
- High/critical only

**Error Handling**:
- Timeout → finding without suggestions
- Invalid JSON → graceful fallback
- Rate limiting → cached response
- LLM unavailable → findings without suggestions

### Integration Tests

- End-to-end analysis with suggestions
- Database persistence of suggestions
- Cache hit/miss validation
- Webhook response includes suggestions

### Test Coverage Target

- 85%+ code coverage on suggestion module
- All 3 providers tested (Claude, Ollama, OpenRouter)
- 20+ comprehensive test cases

---

## Success Criteria

✅ All high/critical findings include auto_fix + improvement_suggestions
✅ All findings include explanation
✅ Auto-fix confidence >0.8 for acceptance
✅ 85%+ test coverage on suggestion generation
✅ Suggestion generation <3 seconds per finding (with caching)
✅ Zero regressions in existing functionality
✅ All 3 LLM providers fully supported
✅ Graceful degradation: findings work even if suggestions fail
✅ Database schema supports new fields
✅ API response includes suggestions for all finding severities

---

## Risks & Mitigations

| Risk | Impact | Mitigation |
|------|--------|-----------|
| Auto-fix generates invalid code | High | Syntax validation + confidence threshold |
| LLM hallucination (bad fix) | High | Conservative prompts + safety checks |
| Suggestion generation slows analysis | Medium | Async if needed later + caching |
| Token cost increases | Medium | Severity-based generation (skip low) |
| Cache staleness | Low | 7-day TTL + manual invalidation option |

---

## Implementation Tasks

See GitHub issues for detailed task breakdown:
- **Task 4.1**: Extend LLM Provider Interface (3h)
- **Task 4.2**: Suggestion Integration (2h)
- **Task 4.3**: Data Model & Database (1h)
- **Task 4.4**: Caching & Optimization (2h)
- **Task 4.5**: Validation & Safety (2h)
- **Task 4.6**: Testing & Documentation (3h)

**Total Estimated Effort**: 13 hours

---

## Next Steps

1. ✅ Design complete
2. → Create GitHub issues for each task
3. → Create feature branch: `feature/phase-4-suggestions`
4. → Implement tasks in order (dependencies: 4.1 → 4.2/4.3 → 4.4/4.5 → 4.6)
5. → Open PR with comprehensive testing
6. → Merge and close Phase 4

