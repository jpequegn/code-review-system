# Phase 5: Intelligence & Learning System

**Vision**: Transform from "suggesting fixes" to "learning what works for your team"

**Duration**: 3-4 weeks | **Effort**: High | **Impact**: Critical for production

## Overview

Phase 5 implements an intelligent learning feedback loop that:
- Tracks which suggestions developers accept/reject
- Tunes suggestion confidence based on real outcomes
- Ranks suggestions by relevance and impact
- Provides team-level insights and learning paths
- Continuously improves accuracy over time

## Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    GitHub Webhook Events                       │
│        (PR opened, PR synchronized, Issue commented)          │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│             Feedback Collection & Tracking                     │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Parse PR comments (detect user acceptance/rejection)  │  │
│  │ 2. Track commit messages (applied suggestions)           │  │
│  │ 3. Link findings to outcomes (success/ignored)           │  │
│  │ 4. Store in feedback database                           │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│              Learning & Analysis Engine                        │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ 1. Acceptance Rate Analytics                            │  │
│  │    - By finding type, severity, category               │  │
│  │ 2. Confidence Tuning                                    │  │
│  │    - Adjust thresholds based on accuracy               │  │
│  │ 3. Suggestion Ranking                                  │  │
│  │    - Prioritize high-impact fixes                      │  │
│  │ 4. Team Insights Generation                            │  │
│  │    - Learning paths, patterns, anti-patterns           │  │
│  └──────────────────────────────────────────────────────────┘  │
└──────────────────────────┬─────────────────────────────────────┘
                           │
                           ▼
┌────────────────────────────────────────────────────────────────┐
│              Enhanced Suggestion Generation                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │ - Rank by confidence + acceptance rate                  │  │
│  │ - Filter duplicates intelligently                       │  │
│  │ - Personalize based on team patterns                    │  │
│  │ - Show only actionable suggestions                      │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

## Phase 5 Tasks

### Task 5.1: Feedback Collection & Storage (1 week)
**Objective**: Build infrastructure to track suggestion outcomes

#### Requirements
- Parse PR comments for feedback signals
- Detect acceptance indicators ("looks good", "merged", emoji reactions)
- Detect rejection indicators ("won't fix", "not applicable")
- Link findings to their outcomes
- Store feedback history with metadata

#### Implementation
```
Database Schema:
┌──────────────────────────────────────────┐
│ SuggestionFeedback                       │
├──────────────────────────────────────────┤
│ id (PK)                                  │
│ finding_id (FK) → Finding                │
│ feedback_type (ACCEPTED|REJECTED|IGNORED)│
│ confidence (0.0-1.0) - original score    │
│ developer_comment (text)                 │
│ commit_hash (if applied)                 │
│ pr_number                                │
│ created_at                               │
│ updated_at                               │
└──────────────────────────────────────────┘

┌──────────────────────────────────────────┐
│ SuggestionImpact                         │
├──────────────────────────────────────────┤
│ id (PK)                                  │
│ finding_id (FK)                          │
│ impact_score (0-100)                     │
│ acceptance_rate (%)                      │
│ avg_time_to_fix (hours)                  │
│ similar_findings_count                   │
│ last_updated                             │
└──────────────────────────────────────────┘
```

#### Components
1. **FeedbackParser** (src/learning/feedback_parser.py)
   - Parse PR comments for signals
   - Extract commit messages
   - Detect acceptance/rejection patterns

2. **FeedbackCollector** (src/learning/feedback_collector.py)
   - Link suggestions to feedback
   - Store in database
   - Handle edge cases

3. **Tests** (tests/test_feedback_*.py)
   - Parser tests: 20+ test cases
   - Collector tests: 15+ test cases
   - Integration tests: 10+ test cases

#### Success Criteria
- ✅ Parse 95%+ of acceptance/rejection signals correctly
- ✅ Link feedback to original findings
- ✅ No data loss on system restart
- ✅ 45+ tests covering all scenarios
- ✅ <100ms overhead per suggestion

---

### Task 5.2: Learning Engine & Analytics (1.5 weeks)
**Objective**: Analyze feedback and extract insights

#### Requirements
- Calculate acceptance rates by finding type
- Identify patterns in suggestion effectiveness
- Generate team learning insights
- Track confidence accuracy
- Predict which suggestions will be accepted

#### Implementation
```
Analytics Models:
┌─────────────────────────────────────┐
│ SuggestionAnalytics                 │
├─────────────────────────────────────┤
│ finding_category (SECURITY, PERF)   │
│ finding_severity (CRITICAL-LOW)     │
│ suggestion_type (AUTO_FIX, EXPLAIN) │
│ acceptance_rate (%)                 │
│ avg_confidence (0.0-1.0)            │
│ sample_size (n findings)            │
│ trend (improving, stable, declining)│
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│ TeamInsights                        │
├─────────────────────────────────────┤
│ team_id (from GitHub org)           │
│ top_vulnerabilities (list)          │
│ most_common_patterns (list)         │
│ improvement_trends (by category)    │
│ learning_path (recommendations)     │
│ generated_at                        │
└─────────────────────────────────────┘
```

#### Components
1. **LearningEngine** (src/learning/engine.py)
   - Calculate acceptance metrics
   - Analyze patterns
   - Generate insights

2. **Analyzer** (src/learning/analyzer.py)
   - Statistical analysis
   - Trend detection
   - Prediction models

3. **Tests** (tests/test_learning_*.py)
   - Analytics tests: 25+ test cases
   - Insight generation tests: 15+ test cases
   - Edge case tests: 10+ test cases

#### Success Criteria
- ✅ Analyze 10K+ data points accurately
- ✅ Identify patterns with 90%+ confidence
- ✅ Generate actionable insights
- ✅ 50+ tests covering all scenarios
- ✅ <500ms analytics generation

---

### Task 5.3: Confidence Tuning System (1 week)
**Objective**: Improve accuracy by adjusting thresholds based on outcomes

#### Requirements
- Calculate suggestion accuracy per category
- Adjust confidence thresholds dynamically
- Balance precision vs. recall
- Prevent threshold drift
- Provide audit trail for changes

#### Implementation
```
Confidence Tuning Algorithm:

For each (finding_category, severity, suggestion_type):
  1. Collect historical feedback
  2. Calculate:
     - True Positive Rate (accepted suggestions)
     - False Positive Rate (rejected suggestions)
     - Precision (TP / (TP + FP))
     - Recall (TP / (TP + FN))

  3. Optimize threshold to maximize:
     F1 = 2 * (Precision * Recall) / (Precision + Recall)

  4. Constraints:
     - Min confidence: 0.7 (safety)
     - Max confidence: 0.99 (always validate)
     - Max drift per update: ±0.05
     - Require min 50 samples before adjusting

┌──────────────────────────────────────┐
│ ConfidenceThreshold                  │
├──────────────────────────────────────┤
│ id (PK)                              │
│ finding_category                     │
│ suggestion_type                      │
│ current_threshold (0.0-1.0)          │
│ previous_threshold                   │
│ accuracy (TP rate)                   │
│ samples (n feedback items)           │
│ updated_at                           │
│ tuning_reason (explanation)          │
└──────────────────────────────────────┘
```

#### Components
1. **ConfidenceTuner** (src/learning/confidence_tuner.py)
   - Calculate optimal thresholds
   - Apply constraints
   - Track history

2. **Metrics Calculator** (src/learning/metrics.py)
   - Precision, recall, F1
   - Accuracy by category
   - Trend analysis

3. **Tests** (tests/test_confidence_*.py)
   - Tuning algorithm tests: 20+ test cases
   - Constraint validation tests: 15+ test cases
   - Edge case tests: 10+ test cases

#### Success Criteria
- ✅ Improve suggestion accuracy by 10-15%
- ✅ Maintain safety constraints (min 0.7 confidence)
- ✅ Prevent threshold oscillation
- ✅ Provide audit trail for all changes
- ✅ 45+ tests covering all scenarios

---

### Task 5.4: Suggestion Ranking & Deduplication (1 week)
**Objective**: Prioritize suggestions by impact and relevance

#### Requirements
- Rank suggestions by importance
- Deduplicate similar findings
- Group related issues
- Show only high-impact suggestions
- Personalize based on team patterns

#### Implementation
```
Ranking Algorithm:

For each suggestion:
  impact_score = (
    severity_weight (0.3) * severity_level +
    confidence_weight (0.3) * suggestion_confidence +
    acceptance_weight (0.2) * historical_acceptance_rate +
    frequency_weight (0.1) * how_common_this_issue_is +
    expertise_weight (0.1) * team_familiarity_with_fix
  )

  final_rank = impact_score * diversity_factor
    (reduce rank if similar suggestion already shown)

┌────────────────────────────────────────┐
│ SuggestionRanking                      │
├────────────────────────────────────────┤
│ suggestion_id (PK)                     │
│ impact_score (0-100)                   │
│ rank_position (1st, 2nd, 3rd...)      │
│ diversity_factor (0.0-1.0)             │
│ similar_suggestions (list)             │
│ explanation (why this rank)            │
│ generated_at                           │
└────────────────────────────────────────┘
```

#### Components
1. **RankingEngine** (src/learning/ranking_engine.py)
   - Calculate impact scores
   - Rank suggestions
   - Handle ties

2. **DeduplicationService** (src/learning/deduplication.py)
   - Find similar findings
   - Group related issues
   - Calculate diversity

3. **Tests** (tests/test_ranking_*.py)
   - Ranking tests: 20+ test cases
   - Deduplication tests: 15+ test cases
   - Integration tests: 10+ test cases

#### Success Criteria
- ✅ Rank suggestions correctly by impact
- ✅ Deduplicate 95%+ of redundant suggestions
- ✅ Show only top 3-5 relevant suggestions
- ✅ Personalization based on team patterns
- ✅ 45+ tests covering all scenarios

---

### Task 5.5: Team Insights Dashboard (1.5 weeks)
**Objective**: Provide actionable insights to development teams

#### Requirements
- Dashboard showing team metrics
- Vulnerability trends over time
- Common anti-patterns
- Learning paths and recommendations
- Suggestion effectiveness metrics
- ROI analysis (effort saved)

#### Implementation
```
Insights Engine Components:

1. Trend Analysis
   - Most common vulnerability types
   - Frequency over time (by month)
   - Improvement/regression detection

2. Learning Paths
   - Identify top 3-5 improvement areas
   - Recommended learning resources
   - Expected impact of improvement

3. Team Metrics
   - Fix acceptance rate (by developer)
   - Time to fix (average)
   - Suggestion quality score
   - ROI (hours saved from auto-fixes)

4. Comparison
   - Team vs. org/industry benchmarks
   - Top performers
   - Areas for improvement

Dashboard Models:
┌──────────────────────────────────┐
│ TeamMetrics                      │
├──────────────────────────────────┤
│ team_id                          │
│ period (week, month, all-time)   │
│ total_findings (n)               │
│ accepted_rate (%)                │
│ avg_fix_time (hours)             │
│ roi_hours_saved                  │
│ top_vulnerabilities (list)       │
│ trend_direction (↑ ↓ →)          │
│ generated_at                     │
└──────────────────────────────────┘

┌──────────────────────────────────┐
│ LearningPath                     │
├──────────────────────────────────┤
│ team_id                          │
│ rank (1st, 2nd, 3rd)            │
│ vulnerability_type               │
│ current_acceptance_rate (%)      │
│ potential_rate (with learning)   │
│ estimated_hours_saved            │
│ resources (links, guides)        │
│ priority_score (0-100)           │
└──────────────────────────────────┘
```

#### Components
1. **InsightsGenerator** (src/learning/insights.py)
   - Generate team metrics
   - Analyze trends
   - Create learning paths

2. **DashboardAPI** (src/api/insights_routes.py)
   - Expose metrics via API
   - Support filtering/grouping
   - Time-series data

3. **Frontend** (optional Phase 5.5)
   - React/Vue dashboard
   - Charts and visualizations
   - Real-time metrics

4. **Tests** (tests/test_insights_*.py)
   - Insights generation tests: 20+ test cases
   - API tests: 15+ test cases
   - Edge case tests: 10+ test cases

#### Success Criteria
- ✅ Dashboard shows actionable insights
- ✅ Trends detectable within 1-2 weeks
- ✅ Learning paths improve team velocity
- ✅ ROI visible and measurable
- ✅ 45+ tests covering all scenarios

---

### Task 5.6: Integration & Production (1 week)
**Objective**: Integrate all components and prepare for production

#### Requirements
- End-to-end testing
- Performance optimization
- Documentation
- Monitoring and alerting
- Gradual rollout strategy

#### Implementation
1. **Integration Testing**
   - Full workflow tests: feedback → learning → ranking
   - Database consistency checks
   - Performance benchmarks
   - Load testing (10K+ suggestions)

2. **Optimization**
   - Cache frequently used analytics
   - Batch learning jobs (run hourly)
   - Optimize database queries
   - Implement circuit breakers

3. **Monitoring**
   - Track learning engine health
   - Alert on anomalies
   - Monitor confidence threshold changes
   - Dashboard latency tracking

4. **Documentation**
   - Architecture guide
   - Operations manual
   - Troubleshooting guide
   - API documentation

5. **Rollout Strategy**
   - Beta with 10% of repos
   - Monitor for regressions
   - Gradual expansion
   - Kill switch if issues

#### Success Criteria
- ✅ All components integrated
- ✅ <1s dashboard load time
- ✅ Zero data loss
- ✅ 95%+ uptime
- ✅ Full documentation
- ✅ Monitoring in place

---

## Database Schema

### New Tables

```sql
-- Feedback tracking
CREATE TABLE suggestion_feedback (
    id INTEGER PRIMARY KEY,
    finding_id INTEGER NOT NULL,
    feedback_type TEXT NOT NULL, -- ACCEPTED, REJECTED, IGNORED
    confidence REAL,
    developer_id TEXT,
    developer_comment TEXT,
    commit_hash TEXT,
    pr_number INTEGER,
    created_at TIMESTAMP NOT NULL,
    updated_at TIMESTAMP NOT NULL,
    FOREIGN KEY (finding_id) REFERENCES finding(id)
);

-- Suggestion impact tracking
CREATE TABLE suggestion_impact (
    id INTEGER PRIMARY KEY,
    finding_id INTEGER NOT NULL,
    impact_score REAL,
    acceptance_rate REAL,
    avg_time_to_fix REAL,
    similar_findings_count INTEGER,
    last_updated TIMESTAMP,
    FOREIGN KEY (finding_id) REFERENCES finding(id)
);

-- Confidence thresholds (tuned)
CREATE TABLE confidence_threshold (
    id INTEGER PRIMARY KEY,
    finding_category TEXT NOT NULL,
    suggestion_type TEXT NOT NULL,
    current_threshold REAL,
    previous_threshold REAL,
    accuracy REAL,
    samples INTEGER,
    updated_at TIMESTAMP,
    tuning_reason TEXT
);

-- Suggestion rankings
CREATE TABLE suggestion_ranking (
    id INTEGER PRIMARY KEY,
    suggestion_id TEXT NOT NULL,
    impact_score REAL,
    rank_position INTEGER,
    diversity_factor REAL,
    similar_suggestions TEXT, -- JSON array
    explanation TEXT,
    generated_at TIMESTAMP
);

-- Team metrics
CREATE TABLE team_metrics (
    id INTEGER PRIMARY KEY,
    team_id TEXT NOT NULL,
    period TEXT, -- week, month, all-time
    total_findings INTEGER,
    accepted_rate REAL,
    avg_fix_time REAL,
    roi_hours_saved REAL,
    top_vulnerabilities TEXT, -- JSON array
    trend_direction TEXT,
    generated_at TIMESTAMP
);

-- Learning paths
CREATE TABLE learning_path (
    id INTEGER PRIMARY KEY,
    team_id TEXT NOT NULL,
    rank INTEGER,
    vulnerability_type TEXT,
    current_acceptance_rate REAL,
    potential_rate REAL,
    estimated_hours_saved REAL,
    resources TEXT, -- JSON array
    priority_score REAL,
    created_at TIMESTAMP
);
```

### Indexes

```sql
CREATE INDEX idx_feedback_finding ON suggestion_feedback(finding_id);
CREATE INDEX idx_feedback_type ON suggestion_feedback(feedback_type);
CREATE INDEX idx_impact_finding ON suggestion_impact(finding_id);
CREATE INDEX idx_threshold_category ON confidence_threshold(finding_category);
CREATE INDEX idx_ranking_position ON suggestion_ranking(rank_position);
CREATE INDEX idx_team_metrics_team ON team_metrics(team_id);
CREATE INDEX idx_learning_path_team ON learning_path(team_id);
```

---

## Testing Strategy

### Test Coverage Targets
- **Unit Tests**: 300+ tests (learning module)
- **Integration Tests**: 50+ tests
- **Performance Tests**: 20+ tests
- **Target Coverage**: 85%+

### Test Categories

1. **Feedback Collection** (45 tests)
   - Parser accuracy
   - Data integrity
   - Edge cases

2. **Learning Engine** (50 tests)
   - Analytics calculations
   - Pattern detection
   - Insight generation

3. **Confidence Tuning** (45 tests)
   - Threshold calculation
   - Constraint validation
   - Drift prevention

4. **Ranking & Deduplication** (50 tests)
   - Impact scoring
   - Deduplication accuracy
   - Ranking correctness

5. **Dashboard API** (40 tests)
   - API endpoints
   - Data filtering
   - Performance

6. **Integration** (50 tests)
   - End-to-end workflows
   - Database consistency
   - Error handling

---

## Success Metrics

### Learning Effectiveness
- ✅ Suggestion acceptance rate: +15-20% (from current 60%)
- ✅ False positive rate: -30-40%
- ✅ Time to fix: -20% (faster application)
- ✅ Developer satisfaction: +25% (from surveys)

### Performance
- ✅ Learning engine: <100ms per update
- ✅ Dashboard load: <1 second
- ✅ Ranking: <50ms per PR
- ✅ API response: <200ms p95

### Quality
- ✅ Test coverage: 85%+
- ✅ Zero regressions in Phase 4
- ✅ Data consistency: 100%
- ✅ Uptime: 99.9%

---

## Risk Mitigation

### Risks & Mitigations

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|-----------|
| Confidence threshold oscillation | Medium | Medium | Constraints + audit trail |
| False positive feedback signals | High | Medium | Manual validation + thresholds |
| Performance degradation | High | Low | Caching + batch processing |
| Data loss on feedback | Critical | Low | Transaction safety + backups |
| Dashboard wrong insights | Medium | Medium | Validation tests + alerts |

---

## Timeline & Milestones

```
Week 1-2:   Task 5.1 (Feedback Collection) + Task 5.2 (Learning Engine)
Week 2-3:   Task 5.3 (Confidence Tuning) + Task 5.4 (Ranking)
Week 3-4:   Task 5.5 (Insights Dashboard) + Task 5.6 (Integration)

Milestones:
- Day 7:  Feedback collection working, 30+ tests
- Day 14: Learning engine complete, 100+ tests
- Day 21: Full system integrated, 250+ tests
- Day 28: Production ready, 300+ tests
```

---

## Future Enhancements (Phase 6+)

1. **Predictive Models**
   - ML models to predict suggestion acceptance
   - Personalization per developer
   - Anomaly detection

2. **Collaborative Learning**
   - Share insights across teams
   - Industry benchmarks
   - Best practice library

3. **Automated Fixes**
   - Auto-commit suggestions (with review)
   - Batch apply low-risk fixes
   - Safe mode (test validation required)

4. **Extended Analytics**
   - Code quality trends
   - Security posture tracking
   - ROI dashboards for management

---

## References

- Phase 4 Specification: `docs/SUGGESTIONS.md`
- Test Strategy: `docs/TESTING_STRATEGY.md`
- API Design: `docs/API_DESIGN.md`
