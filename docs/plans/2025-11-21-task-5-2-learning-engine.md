# Task 5.2: Learning Engine & Analytics

**Objective**: Analyze feedback data to improve suggestion accuracy, ranking, and team insights

**Duration**: 1-1.5 weeks | **Effort**: 40-50 hours | **Depends On**: Task 5.1 (complete)

**Start Date**: 2025-11-21

---

## Overview

Task 5.2 builds on the feedback collection infrastructure (Task 5.1) to:

1. **Analyze Acceptance Patterns** - What gets accepted vs rejected by type/severity/category
2. **Tune Confidence Scores** - Recalibrate based on real outcomes
3. **Rank Suggestions** - Prioritize high-impact, high-confidence fixes
4. **Learn Team Patterns** - Identify common issues and anti-patterns specific to the team

This transforms raw feedback data into actionable intelligence for better suggestions.

---

## Architecture & Data Flow

```
SuggestionFeedback (from Task 5.1)
    ↓
Learning Engine
    ├─ Acceptance Rate Analytics
    │   ├─ By finding type (security, performance)
    │   ├─ By severity (critical, high, medium, low)
    │   ├─ By category (injection, auth, resource leak, etc)
    │   └─ Store in LearningMetrics
    │
    ├─ Confidence Tuning
    │   ├─ Compare original → actual acceptance
    │   ├─ Calibration curve: precision vs recall
    │   ├─ Adjust thresholds dynamically
    │   └─ Store in ConfidenceCalibration
    │
    ├─ Suggestion Ranking
    │   ├─ Combine: confidence × acceptance_rate × impact
    │   ├─ Rank by team-weighted score
    │   └─ Store in SuggestionRankings
    │
    └─ Pattern Learning
        ├─ Cluster similar findings
        ├─ Detect anti-patterns
        ├─ Track fix time (time from suggestion to commit)
        └─ Store in PatternMetrics
            ↓
Enhanced Suggestion Generation (Phase 5.3)
    ├─ Filter by updated confidence thresholds
    ├─ Rank by relevance + impact
    └─ Personalize based on team patterns
```

---

## Task Breakdown (5 subtasks)

### Subtask 5.2.1: Learning Metrics & Schema (4 hours)

**Objective**: Define data structures for analytics

**Files to Create**:
- `src/learning/metrics.py` - Data classes for learning metrics
- `src/database.py` - Add new tables (3 tables, ~200 lines total)
- `tests/test_learning_metrics.py` - Schema validation

**Schema Changes** (add to `src/database.py`):

```python
# Table 1: LearningMetrics
class LearningMetrics(Base):
    __tablename__ = "learning_metrics"

    id: int (PK)
    finding_category: str (e.g., "SQL Injection", "Resource Leak")
    severity: str (critical, high, medium, low)
    issue_type: str (security, performance)  # Composite key: (category, severity, type)

    total_findings: int
    accepted_count: int
    rejected_count: int
    ignored_count: int
    acceptance_rate: float (0.0-1.0)

    avg_original_confidence: float
    avg_actual_confidence: float  # After tuning

    avg_time_to_fix: float (hours, nullable)
    fix_rate: float (% of suggestions that were actually applied)

    last_updated: datetime
    created_at: datetime

# Table 2: ConfidenceCalibration
class ConfidenceCalibration(Base):
    __tablename__ = "confidence_calibration"

    id: int (PK)
    bin_id: int (0-10, representing 0.0-0.1, 0.1-0.2, ..., 0.9-1.0)

    original_confidence_range: str (e.g., "0.8-0.9")
    sample_size: int (how many samples in this bin)

    actual_acceptance_rate: float (what % were actually accepted)
    precision: float (of accepted, how many were correct)
    recall: float (of all correct, how many were found)
    f1_score: float

    calibrated_threshold: float (new threshold for this confidence bin)

    last_updated: datetime
    created_at: datetime

# Table 3: PatternMetrics
class PatternMetrics(Base):
    __tablename__ = "pattern_metrics"

    id: int (PK)
    pattern_hash: str (sha256 of normalized pattern)
    pattern_type: str (e.g., "n_plus_one_query", "unclosed_file")

    occurrences: int
    files_affected: str (JSON list of file paths)
    avg_severity: float (0-1)

    acceptance_rate: float
    fix_count: int
    anti_pattern: bool (True if pattern is anti-pattern)

    team_prevalence: str (rare, occasional, common)
    recommended_fix: str (template)

    last_updated: datetime
    created_at: datetime
```

**Data Classes** (`src/learning/metrics.py`):

```python
@dataclass
class AcceptanceMetrics:
    """Acceptance rates by category"""
    finding_category: str
    severity: str
    total: int
    accepted: int
    rejected: int
    ignored: int
    acceptance_rate: float
    confidence_avg: float
    fix_rate: float

@dataclass
class ConfidenceBin:
    """Confidence calibration for a bin"""
    bin_id: int  # 0-10
    confidence_range: str
    original_acceptance_rate: float
    precision: float
    recall: float
    f1_score: float
    calibrated_threshold: float
    sample_size: int

@dataclass
class PatternInfo:
    """Detected pattern with metrics"""
    pattern_type: str
    occurrences: int
    files: list[str]
    avg_severity: float
    acceptance_rate: float
    anti_pattern: bool
    team_prevalence: str
    recommended_fix: str
```

**Success Criteria**:
- ✅ All 3 tables created with proper relationships
- ✅ Foreign keys to Finding/SuggestionFeedback where applicable
- ✅ Indexes on (category, severity) and (pattern_type)
- ✅ Tests verify schema integrity
- ✅ Dataclasses match ORM models

---

### Subtask 5.2.2: Acceptance Rate Analytics (6 hours)

**Objective**: Calculate acceptance rates by category, severity, and type

**Files to Create**:
- `src/learning/analytics.py` - AcceptanceAnalyzer class
- `tests/test_acceptance_analytics.py` - Comprehensive test suite

**AcceptanceAnalyzer Class** (~250 lines):

```python
class AcceptanceAnalyzer:
    """Analyzes feedback to extract acceptance patterns"""

    def __init__(self, db: Session):
        self.db = db

    # ============================================================================
    # Core Analytics Methods
    # ============================================================================

    def calculate_acceptance_by_category(
        self,
        days: int = 30
    ) -> list[AcceptanceMetrics]:
        """
        Acceptance rates for each finding category

        Returns: [(SQL Injection: 0.78), (Resource Leak: 0.65), ...]
        """
        # Query all unique categories from findings
        # For each category:
        #   - Count total findings
        #   - Count accepted (feedback_type == ACCEPTED)
        #   - Count rejected (feedback_type == REJECTED)
        #   - Count ignored (no feedback)
        #   - Calculate: acceptance_rate = accepted / (accepted + rejected)
        #   - Calculate: fix_rate = actually_applied / total
        # Return sorted by acceptance_rate DESC

    def calculate_acceptance_by_severity(
        self,
        days: int = 30
    ) -> list[AcceptanceMetrics]:
        """
        Acceptance rates by severity level

        Returns: [(CRITICAL: 0.92), (HIGH: 0.78), (MEDIUM: 0.45), (LOW: 0.20)]
        """
        # Similar to by_category but group by severity enum

    def calculate_acceptance_by_issue_type(
        self,
        days: int = 30
    ) -> list[AcceptanceMetrics]:
        """
        Acceptance rates by issue type (security vs performance)

        Returns: [(SECURITY: 0.85), (PERFORMANCE: 0.68)]
        """
        # Group by finding category (FindingCategory enum)

    def calculate_composite_acceptance(
        self,
        category: str | None = None,
        severity: str | None = None,
        issue_type: str | None = None,
        days: int = 30
    ) -> AcceptanceMetrics:
        """
        Calculate acceptance for specific filter combination

        Example: acceptance_for_sql_injection_critical =
            calculate_composite_acceptance(
                category="SQL Injection",
                severity="CRITICAL"
            )
        """
        # Filter findings by provided criteria
        # Calculate metrics
        # Return single AcceptanceMetrics

    def get_fixing_timeline(
        self,
        category: str | None = None,
        limit: int = 100
    ) -> list[dict]:
        """
        Time from suggestion to actual fix (commit)

        Returns: [
            {
                'finding_id': 42,
                'suggested_at': '2025-11-01 10:00:00',
                'fixed_at': '2025-11-02 14:30:00',
                'time_to_fix_hours': 28.5,
                'accepted': True
            },
            ...
        ]
        """
        # Join Finding → SuggestionFeedback → commit_hash timestamp
        # Calculate delta between created_at and commit timestamp
        # Return sorted by time_to_fix DESC

    # ============================================================================
    # Aggregation & Persistence
    # ============================================================================

    def persist_metrics_for_category(
        self,
        category: str,
        severity: str,
        issue_type: str
    ) -> LearningMetrics:
        """
        Calculate and persist metrics for specific category+severity+type combo

        Returns: LearningMetrics record (new or updated)
        """
        # Calculate acceptance metrics
        # Create or update LearningMetrics record
        # Update last_updated timestamp
        # Commit to database

    def persist_all_metrics(self) -> int:
        """
        Recalculate and persist metrics for all category+severity combos

        Returns: Number of metrics records updated/created
        """
        # Get all unique (category, severity, type) combinations
        # For each: call persist_metrics_for_category()
        # Log progress
        # Return count

    def get_metrics_by_category(
        self,
        category: str
    ) -> list[LearningMetrics]:
        """Retrieve persisted metrics for a category"""
        # Query LearningMetrics table
        # Filter by finding_category
        # Return sorted by severity (CRITICAL > HIGH > MEDIUM > LOW)

    def get_top_categories_by_acceptance(
        self,
        limit: int = 10,
        ascending: bool = False
    ) -> list[tuple[str, float]]:
        """
        Get categories ranked by acceptance rate

        Returns: [("SQL Injection", 0.92), ("Auth Flaw", 0.78), ...]
        """
        # Query LearningMetrics
        # Sort by acceptance_rate
        # Return top N
```

**Test Coverage** (~350 lines, 20+ tests):

```python
class TestAcceptanceAnalytics:

    def test_acceptance_by_category_basic(self):
        # Create findings: 3 SQL Injection (2 accepted, 1 rejected)
        # Expect: acceptance_rate = 0.67

    def test_acceptance_by_severity_all_levels(self):
        # Create findings across all severity levels
        # Verify acceptance_rate varies by severity

    def test_acceptance_by_issue_type(self):
        # Create security and performance findings
        # Verify rates calculated separately

    def test_composite_acceptance_filtering(self):
        # Calculate with multiple filters
        # Verify filters compose correctly

    def test_fixing_timeline_calculation(self):
        # Create feedback with commit_hash + timestamp
        # Verify time_to_fix calculated correctly

    def test_fixing_timeline_missing_commit(self):
        # Feedback without commit_hash
        # Expect graceful handling (no time_to_fix)

    def test_persist_metrics_new_record(self):
        # Call persist_metrics_for_category()
        # Verify LearningMetrics created in DB

    def test_persist_metrics_update_existing(self):
        # Call twice with same category
        # Verify second call updates (not duplicates)

    def test_persist_all_metrics_coverage(self):
        # Create findings across multiple categories
        # Call persist_all_metrics()
        # Verify all combos persisted

    def test_time_window_filtering(self):
        # Create feedback from 60 days ago and 10 days ago
        # Query with days=30
        # Verify only recent feedback counted

    def test_empty_dataset(self):
        # No findings in DB
        # Expect empty list or 0.0 rates, no crashes
```

**Success Criteria**:
- ✅ All 6 analytics methods implemented and tested
- ✅ Acceptance rates accurate (verified with manual calculation)
- ✅ Time-window filtering working (days parameter)
- ✅ Composite filters compose correctly
- ✅ Metrics persist to database with update logic
- ✅ 20+ test cases covering normal and edge cases

---

### Subtask 5.2.3: Confidence Calibration & Tuning (8 hours)

**Objective**: Recalibrate confidence thresholds based on feedback outcomes

**Files to Create**:
- `src/learning/confidence_tuner.py` - ConfidenceTuner class
- `tests/test_confidence_tuning.py` - Calibration tests

**ConfidenceTuner Class** (~350 lines):

```python
class ConfidenceTuner:
    """
    Recalibrate confidence scores based on actual acceptance outcomes

    Problem: LLM gives confidence 0.95, but developers reject 40% → miscalibrated
    Solution: Create confidence bins, measure actual acceptance, adjust thresholds
    """

    def __init__(self, db: Session):
        self.db = db
        self.confidence_bins = 10  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0

    # ============================================================================
    # Confidence Binning
    # ============================================================================

    def bin_confidence(self, confidence: float) -> int:
        """
        Convert confidence score to bin number

        0.95 → bin 9 (0.9-1.0)
        0.55 → bin 5 (0.5-0.6)
        0.05 → bin 0 (0.0-0.1)
        """
        bin_id = min(int(confidence * 10), 9)
        return bin_id

    def get_confidence_range_for_bin(self, bin_id: int) -> tuple[float, float]:
        """Get min, max for a bin"""
        return (bin_id * 0.1, (bin_id + 1) * 0.1)

    # ============================================================================
    # Calibration Analysis
    # ============================================================================

    def calculate_bin_statistics(self, bin_id: int) -> ConfidenceBin:
        """
        For all suggestions in this confidence bin, calculate:
        - Original acceptance rate (what was reported)
        - Actual acceptance rate (what feedback shows)
        - Precision: TP / (TP + FP)
        - Recall: TP / (TP + FN)
        - F1 Score

        Returns: ConfidenceBin with calibration metrics
        """
        # Query all SuggestionFeedback with original_confidence in range
        # Count: accepted (TP), rejected (FP), no feedback (FN)
        # Calculate: precision = TP / (TP + FP)
        #            recall = TP / (TP + FN)
        #            f1 = 2 * (precision * recall) / (precision + recall)
        # Return ConfidenceBin

    def calculate_all_bin_statistics(self) -> list[ConfidenceBin]:
        """
        Calibrate all 10 confidence bins

        Returns: [ConfidenceBin for bin 0, bin 1, ..., bin 9]
        """
        return [self.calculate_bin_statistics(i) for i in range(10)]

    # ============================================================================
    # Threshold Computation
    # ============================================================================

    def compute_optimal_threshold(
        self,
        target_precision: float = 0.85,
        target_recall: float = 0.70
    ) -> ConfidenceBin:
        """
        Find confidence bin that best balances precision and recall

        Strategy: Find bin where:
        - Precision ≥ target_precision (don't report false positives)
        - Recall ≥ target_recall (catch most real issues)
        - F1 is maximized

        Returns: Best ConfidenceBin to use as threshold
        """
        # Calculate all bins
        # Score each bin: distance to (target_precision, target_recall) in 2D space
        # Return bin with best score

    def suggest_calibrated_thresholds(self) -> dict[str, float]:
        """
        Suggest new confidence thresholds for different use cases

        Returns: {
            'aggressive': 0.65,    # Catch more issues, tolerate false positives
            'balanced': 0.78,      # Default: 85% precision, 70% recall
            'conservative': 0.88   # Only high-confidence: 95% precision, 50% recall
        }
        """
        bins = self.calculate_all_bin_statistics()

        # aggressive: lowest threshold where precision still > 0.70
        # balanced: where f1 is maximized
        # conservative: where precision > 0.95

        return {
            'aggressive': ...,
            'balanced': ...,
            'conservative': ...
        }

    # ============================================================================
    # Persistence & Application
    # ============================================================================

    def persist_calibration(self, bin_id: int) -> ConfidenceCalibration:
        """
        Calculate and save calibration for one bin

        Returns: ConfidenceCalibration record
        """
        bin_stats = self.calculate_bin_statistics(bin_id)
        record = ConfidenceCalibration(
            bin_id=bin_id,
            original_confidence_range=bin_stats.confidence_range,
            sample_size=bin_stats.sample_size,
            actual_acceptance_rate=bin_stats.original_acceptance_rate,
            precision=bin_stats.precision,
            recall=bin_stats.recall,
            f1_score=bin_stats.f1_score,
            calibrated_threshold=bin_stats.calibrated_threshold
        )
        self.db.add(record)
        self.db.commit()
        return record

    def persist_all_calibrations(self) -> int:
        """
        Recalibrate all 10 bins and persist

        Returns: Count of records created/updated
        """
        count = 0
        for bin_id in range(10):
            self.persist_calibration(bin_id)
            count += 1
        return count

    def apply_calibration_to_finding(
        self,
        original_confidence: float,
        recalibration_mode: str = 'balanced'
    ) -> float:
        """
        Convert original confidence to calibrated confidence

        Example: Original 0.95 in bin 9 with precision 0.92
        → Calibrated 0.92 (based on actual acceptance rate)
        """
        bin_id = self.bin_confidence(original_confidence)
        calib = self.db.query(ConfidenceCalibration).filter(
            ConfidenceCalibration.bin_id == bin_id
        ).first()

        if not calib:
            return original_confidence  # Fallback if no calibration

        if recalibration_mode == 'aggressive':
            return calib.actual_acceptance_rate * 1.1  # Slightly boost
        elif recalibration_mode == 'conservative':
            return calib.actual_acceptance_rate * 0.9  # Slightly reduce
        else:  # balanced (default)
            return calib.actual_acceptance_rate

    # ============================================================================
    # Reporting
    # ============================================================================

    def get_calibration_report(self) -> dict:
        """
        Generate calibration report showing before/after

        Returns: {
            'bins': [
                {
                    'bin_id': 0,
                    'confidence_range': '0.0-0.1',
                    'sample_size': 5,
                    'original_rate': 0.05,
                    'actual_rate': 0.03,
                    'precision': 0.50,
                    'recall': 0.20,
                    'f1_score': 0.29,
                    'calibrated_threshold': 0.03
                },
                ...
            ],
            'recommended_thresholds': {
                'aggressive': 0.65,
                'balanced': 0.78,
                'conservative': 0.88
            },
            'improvement_summary': 'Calibration would reduce false positives by 15%'
        }
        """
        # Gather all calibration data
        # Compare original vs calibrated thresholds
        # Generate summary metrics
```

**Test Coverage** (~300 lines, 18+ tests):

```python
class TestConfidenceTuning:

    def test_bin_confidence_boundaries(self):
        # 0.0 → bin 0, 0.05 → bin 0, 0.1 → bin 1, 0.99 → bin 9
        # Verify boundaries correct

    def test_get_confidence_range(self):
        # bin 0 → (0.0, 0.1), bin 5 → (0.5, 0.6)

    def test_calculate_bin_statistics_basic(self):
        # 10 findings in bin 8 (0.8-0.9):
        #   - 8 accepted (TP)
        #   - 1 rejected (FP)
        #   - 1 no feedback (FN)
        # Expect: precision=0.89, recall=0.89, f1=0.89

    def test_calculate_bin_statistics_empty(self):
        # No findings in bin
        # Expect graceful handling

    def test_all_bins_coverage(self):
        # Create findings across all 10 bins
        # Call calculate_all_bin_statistics()
        # Verify 10 records returned

    def test_compute_optimal_threshold_default(self):
        # Use default targets: precision=0.85, recall=0.70
        # Verify returned bin maximizes f1 within constraints

    def test_compute_optimal_threshold_conservative(self):
        # Use targets: precision=0.95, recall=0.50
        # Expect higher confidence threshold

    def test_suggest_calibrated_thresholds_structure(self):
        # Verify returned dict has 'aggressive', 'balanced', 'conservative'
        # Verify aggressive < balanced < conservative

    def test_suggest_thresholds_precision_requirements(self):
        # Create specific feedback distribution
        # Verify thresholds meet precision requirements

    def test_persist_calibration_new(self):
        # Call persist_calibration(5)
        # Verify ConfidenceCalibration record created in DB

    def test_persist_calibration_update(self):
        # Call twice with same bin_id
        # Verify second call updates (not duplicate)

    def test_persist_all_calibrations_full_sweep(self):
        # Call persist_all_calibrations()
        # Verify 10 records in database

    def test_apply_calibration_adjustment(self):
        # Setup: bin 8 has actual_acceptance_rate=0.80
        # apply_calibration(0.95, 'balanced') → 0.80

    def test_apply_calibration_modes(self):
        # Same original confidence in 'aggressive', 'balanced', 'conservative'
        # Verify aggressive > balanced > conservative

    def test_apply_calibration_no_data(self):
        # Confidence bin not yet calibrated
        # Expect fallback to original confidence

    def test_get_calibration_report_structure(self):
        # Verify report includes bins, thresholds, summary

    def test_get_calibration_report_improvement_calc(self):
        # Verify summary calculates real improvements
```

**Success Criteria**:
- ✅ Confidence binning working (0-10 bins)
- ✅ Precision, recall, F1 calculations verified
- ✅ Optimal threshold computation accurate
- ✅ Three threshold suggestions (aggressive, balanced, conservative)
- ✅ Calibration persists to database
- ✅ Application of calibration to new scores working
- ✅ 18+ test cases covering all scenarios

---

### Subtask 5.2.4: Suggestion Ranking Engine (6 hours)

**Objective**: Rank suggestions by impact and relevance

**Files to Create**:
- `src/learning/suggestion_ranker.py` - SuggestionRanker class
- `tests/test_suggestion_ranking.py` - Ranking tests

**SuggestionRanker Class** (~250 lines):

```python
class SuggestionRanker:
    """
    Rank suggestions by multiple factors:
    - Confidence (calibrated)
    - Acceptance rate (from feedback)
    - Impact score (severity × prevalence)
    - Fix time (how fast developers can fix)
    - Team preferences (learned patterns)
    """

    def __init__(self, db: Session):
        self.db = db
        self.weights = {
            'confidence': 0.30,        # Tuning-adjusted confidence
            'acceptance_rate': 0.25,   # How often accepted
            'impact_score': 0.25,      # Severity × prevalence
            'fix_time': 0.10,          # How quick to fix
            'team_preference': 0.10    # Team-learned patterns
        }

    # ============================================================================
    # Individual Score Components
    # ============================================================================

    def get_confidence_score(
        self,
        finding: Finding,
        tuner: ConfidenceTuner | None = None
    ) -> float:
        """Get calibrated confidence for finding"""
        if not finding.confidence:
            return 0.5

        if tuner:
            return tuner.apply_calibration_to_finding(
                finding.confidence,
                'balanced'
            )
        return finding.confidence

    def get_acceptance_rate_score(
        self,
        finding: Finding
    ) -> float:
        """
        Get acceptance rate for this finding's category+severity

        Returns: 0.0-1.0 based on historical feedback
        """
        metrics = self.db.query(LearningMetrics).filter(
            LearningMetrics.finding_category == finding.title,
            LearningMetrics.severity == finding.severity.value
        ).first()

        if not metrics:
            return 0.5  # Default if no history

        return metrics.acceptance_rate

    def get_impact_score(
        self,
        finding: Finding
    ) -> float:
        """
        Impact = Severity × Prevalence

        Critical + Common = 1.0 (highest impact)
        Low + Rare = 0.2 (lowest impact)
        """
        severity_weight = {
            'CRITICAL': 1.0,
            'HIGH': 0.75,
            'MEDIUM': 0.50,
            'LOW': 0.25
        }

        severity_score = severity_weight.get(finding.severity.value, 0.5)

        # Get prevalence from PatternMetrics
        pattern = self.db.query(PatternMetrics).filter(
            PatternMetrics.pattern_type == finding.title
        ).first()

        if pattern:
            prevalence_weight = {
                'rare': 0.3,
                'occasional': 0.6,
                'common': 1.0
            }
            prevalence_score = prevalence_weight.get(
                pattern.team_prevalence.lower(),
                0.5
            )
        else:
            prevalence_score = 0.5

        return (severity_score + prevalence_score) / 2

    def get_fix_time_score(
        self,
        finding: Finding
    ) -> float:
        """
        Faster to fix = higher score

        <1 hour = 1.0
        1-4 hours = 0.75
        4-8 hours = 0.50
        >8 hours = 0.25
        """
        metrics = self.db.query(LearningMetrics).filter(
            LearningMetrics.finding_category == finding.title
        ).first()

        if not metrics or not metrics.avg_time_to_fix:
            return 0.5

        hours = metrics.avg_time_to_fix
        if hours < 1:
            return 1.0
        elif hours < 4:
            return 0.75
        elif hours < 8:
            return 0.50
        else:
            return 0.25

    def get_team_preference_score(
        self,
        finding: Finding,
        team_id: str | None = None
    ) -> float:
        """
        Score based on team's historical preferences

        - High acceptance rate → higher score
        - Matches learned patterns → higher score
        - Anti-pattern → lower score
        """
        acceptance_rate = self.get_acceptance_rate_score(finding)

        # Check if anti-pattern
        pattern = self.db.query(PatternMetrics).filter(
            PatternMetrics.pattern_type == finding.title,
            PatternMetrics.anti_pattern == True
        ).first()

        if pattern:
            # Team actively avoiding this pattern → boost
            return acceptance_rate * 1.2
        else:
            return acceptance_rate

    # ============================================================================
    # Composite Ranking
    # ============================================================================

    def calculate_ranking_score(
        self,
        finding: Finding,
        tuner: ConfidenceTuner | None = None
    ) -> float:
        """
        Calculate composite ranking score (0-1)

        Score = sum(component × weight)
        """
        scores = {
            'confidence': self.get_confidence_score(finding, tuner),
            'acceptance_rate': self.get_acceptance_rate_score(finding),
            'impact_score': self.get_impact_score(finding),
            'fix_time': self.get_fix_time_score(finding),
            'team_preference': self.get_team_preference_score(finding)
        }

        total_score = sum(
            scores[key] * self.weights[key]
            for key in scores
        )

        return min(total_score, 1.0)  # Cap at 1.0

    def rank_findings(
        self,
        findings: list[Finding],
        tuner: ConfidenceTuner | None = None
    ) -> list[tuple[Finding, float, dict]]:
        """
        Rank findings by composite score

        Returns: [
            (finding, total_score, component_scores),
            ...
        ]
        sorted by total_score DESC
        """
        ranked = []
        for finding in findings:
            scores = {
                'confidence': self.get_confidence_score(finding, tuner),
                'acceptance_rate': self.get_acceptance_rate_score(finding),
                'impact_score': self.get_impact_score(finding),
                'fix_time': self.get_fix_time_score(finding),
                'team_preference': self.get_team_preference_score(finding)
            }

            total_score = sum(
                scores[key] * self.weights[key]
                for key in scores
            )

            ranked.append((finding, total_score, scores))

        return sorted(ranked, key=lambda x: x[1], reverse=True)

    # ============================================================================
    # Custom Weighting
    # ============================================================================

    def set_weights(self, new_weights: dict[str, float]) -> None:
        """
        Adjust ranking weights

        Example: Emphasize security over speed
        ranker.set_weights({
            'confidence': 0.40,
            'acceptance_rate': 0.30,
            'impact_score': 0.20,
            'fix_time': 0.05,
            'team_preference': 0.05
        })
        """
        if abs(sum(new_weights.values()) - 1.0) > 0.01:
            raise ValueError("Weights must sum to ~1.0")
        self.weights = new_weights

    def get_weights(self) -> dict[str, float]:
        """Get current weights"""
        return self.weights.copy()
```

**Test Coverage** (~280 lines, 16+ tests):

```python
class TestSuggestionRanking:

    def test_confidence_score_calculation(self):
        # Finding with confidence=0.85
        # Expect: get_confidence_score() returns 0.85 (or calibrated)

    def test_acceptance_rate_score_with_history(self):
        # Finding category has 80% historical acceptance
        # Expect: get_acceptance_rate_score() returns 0.80

    def test_acceptance_rate_score_no_history(self):
        # Finding category never seen before
        # Expect: returns 0.5 (default)

    def test_impact_score_critical_common(self):
        # Critical severity + common pattern
        # Expect: score close to 1.0

    def test_impact_score_low_rare(self):
        # Low severity + rare pattern
        # Expect: score < 0.5

    def test_fix_time_score_quick(self):
        # Finding avg_time_to_fix = 0.5 hours
        # Expect: 1.0

    def test_fix_time_score_slow(self):
        # Finding avg_time_to_fix = 12 hours
        # Expect: 0.25

    def test_team_preference_score_anti_pattern(self):
        # Finding matches learned anti-pattern
        # Expect: score boosted relative to acceptance

    def test_calculate_ranking_score_composition(self):
        # Create finding, calculate score
        # Verify: score = weighted sum of components

    def test_ranking_score_bounds(self):
        # Even with extreme values, score stays 0-1

    def test_rank_findings_basic(self):
        # 5 findings with different scores
        # Expect: returned sorted DESC by score

    def test_rank_findings_component_scores(self):
        # Verify returned tuples include component breakdown

    def test_set_weights_custom(self):
        # set_weights({'confidence': 0.5, ...})
        # Verify new weights applied

    def test_set_weights_validation(self):
        # set_weights that don't sum to 1.0
        # Expect: ValueError

    def test_rank_findings_security_emphasis(self):
        # Set weights emphasizing impact/confidence
        # Verify high-impact findings ranked first

    def test_rank_findings_empty_list(self):
        # rank_findings([])
        # Expect: []
```

**Success Criteria**:
- ✅ All 5 component scores calculated independently
- ✅ Composite score is weighted sum (verified via unit tests)
- ✅ Ranking sort order correct (highest first)
- ✅ Component scores included in output for transparency
- ✅ Custom weighting working and validated
- ✅ 16+ test cases covering all scenarios

---

### Subtask 5.2.5: Pattern Learning & Detection (6 hours)

**Objective**: Identify and learn team-specific patterns

**Files to Create**:
- `src/learning/pattern_learner.py` - PatternLearner class
- `tests/test_pattern_learning.py` - Pattern detection tests

**PatternLearner Class** (~300 lines):

```python
class PatternLearner:
    """
    Learn team-specific patterns from finding history

    Detects:
    - Common anti-patterns (recurring issues)
    - Suggestion patterns (what suggestions work best)
    - Severity patterns (which patterns are critical)
    - Fix patterns (how team usually fixes issues)
    """

    def __init__(self, db: Session):
        self.db = db

    # ============================================================================
    # Pattern Detection
    # ============================================================================

    def detect_common_patterns(
        self,
        min_occurrences: int = 5,
        days: int = 90
    ) -> list[PatternInfo]:
        """
        Find patterns that appear ≥min_occurrences times

        Returns: [
            PatternInfo(
                pattern_type="unclosed_file_handle",
                occurrences=12,
                files=["src/db.py", "src/api.py", ...],
                avg_severity=0.75,
                acceptance_rate=0.85,
                anti_pattern=True,
                team_prevalence="common",
                recommended_fix="Use context managers: with open(...) as f:"
            ),
            ...
        ]
        sorted by occurrences DESC
        """
        # Group findings by title/category
        # For each group with ≥min_occurrences:
        #   - Get unique file paths
        #   - Calculate avg severity
        #   - Get acceptance rate from feedback
        #   - Check if anti_pattern (always rejected)
        #   - Determine prevalence (rare <3, occasional 3-10, common >10)
        #   - Get recommended fix (template)
        # Return sorted by occurrences DESC

    def detect_anti_patterns(
        self,
        rejection_threshold: float = 0.7,
        min_samples: int = 5
    ) -> list[PatternInfo]:
        """
        Detect patterns with high rejection rate (>70%)

        Returns: Patterns where acceptance_rate < (1 - rejection_threshold)
        """
        patterns = self.detect_common_patterns(min_occurrences=min_samples)

        return [
            p for p in patterns
            if p.acceptance_rate < (1 - rejection_threshold)
        ]

    def detect_best_practices(
        self,
        acceptance_threshold: float = 0.8,
        min_samples: int = 5
    ) -> list[PatternInfo]:
        """
        Detect patterns with high acceptance rate (>80%)

        These are patterns developers like → recommend more

        Returns: Patterns where acceptance_rate > acceptance_threshold
        """
        patterns = self.detect_common_patterns(min_occurrences=min_samples)

        return [
            p for p in patterns
            if p.acceptance_rate > acceptance_threshold
        ]

    def get_pattern_severity_distribution(
        self,
        pattern_type: str
    ) -> dict[str, int]:
        """
        Get severity distribution for a pattern

        Returns: {
            'CRITICAL': 5,
            'HIGH': 8,
            'MEDIUM': 3,
            'LOW': 2
        }
        """
        findings = self.db.query(Finding).filter(
            Finding.title == pattern_type
        ).all()

        distribution = {}
        for finding in findings:
            severity = finding.severity.value
            distribution[severity] = distribution.get(severity, 0) + 1

        return distribution

    def get_pattern_files(
        self,
        pattern_type: str,
        limit: int = 20
    ) -> list[str]:
        """
        Get files most affected by a pattern

        Returns: ["src/db.py", "src/api.py", ...]
        sorted by frequency DESC
        """
        findings = self.db.query(Finding).filter(
            Finding.title == pattern_type
        ).all()

        file_counts = {}
        for finding in findings:
            f = finding.file_path
            file_counts[f] = file_counts.get(f, 0) + 1

        return sorted(
            file_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:limit]

    def is_anti_pattern(
        self,
        pattern_type: str,
        threshold: float = 0.7
    ) -> bool:
        """
        Check if pattern is anti-pattern (rejection rate > threshold)
        """
        patterns = self.detect_anti_patterns()
        return any(p.pattern_type == pattern_type for p in patterns)

    # ============================================================================
    # Fix Pattern Analysis
    # ============================================================================

    def get_fix_patterns(
        self,
        pattern_type: str
    ) -> list[dict]:
        """
        Get how the team usually fixes this pattern

        Returns: [
            {
                'suggestion_fix': 'Use parameterized queries: cursor.execute(...)',
                'acceptance_count': 8,
                'fix_time_avg_hours': 2.5,
                'effectiveness': 0.92
            },
            ...
        ]
        sorted by acceptance_count DESC
        """
        # Query findings with this pattern
        # For each with suggestions:
        #   - Get suggested_fix text
        #   - Count acceptances (from feedback)
        #   - Get avg fix time
        #   - Calculate effectiveness (accepted / total)
        # Return sorted by acceptance DESC

    def get_recommended_fix(
        self,
        pattern_type: str
    ) -> str | None:
        """
        Get the most commonly used fix for this pattern

        Returns: The suggested_fix text that's most often accepted
        """
        fixes = self.get_fix_patterns(pattern_type)
        if fixes:
            return fixes[0]['suggestion_fix']
        return None

    # ============================================================================
    # Pattern Persistence
    # ============================================================================

    def persist_pattern(
        self,
        pattern_info: PatternInfo
    ) -> PatternMetrics:
        """
        Save pattern to database

        Returns: PatternMetrics record
        """
        record = PatternMetrics(
            pattern_hash=hashlib.sha256(
                pattern_info.pattern_type.encode()
            ).hexdigest(),
            pattern_type=pattern_info.pattern_type,
            occurrences=pattern_info.occurrences,
            files_affected=json.dumps(pattern_info.files),
            avg_severity=pattern_info.avg_severity,
            acceptance_rate=pattern_info.acceptance_rate,
            anti_pattern=pattern_info.anti_pattern,
            team_prevalence=pattern_info.team_prevalence,
            recommended_fix=pattern_info.recommended_fix
        )

        self.db.add(record)
        self.db.commit()
        return record

    def persist_all_patterns(self) -> int:
        """
        Detect and persist all patterns

        Returns: Count of patterns persisted
        """
        patterns = self.detect_common_patterns()
        count = 0

        for pattern in patterns:
            self.persist_pattern(pattern)
            count += 1

        return count

    def get_pattern_report(self) -> dict:
        """
        Generate comprehensive pattern report

        Returns: {
            'total_patterns': 23,
            'anti_patterns': 5,
            'best_practices': 8,
            'common_patterns': [
                {
                    'type': 'SQL Injection',
                    'occurrences': 15,
                    'prevalence': 'common',
                    'acceptance_rate': 0.85,
                    'anti_pattern': False,
                    'recommended_fix': '...'
                },
                ...
            ],
            'files_affected': {
                'src/db.py': 12,
                'src/api.py': 8,
                ...
            }
        }
        """
        patterns = self.detect_common_patterns()
        anti = self.detect_anti_patterns()
        best = self.detect_best_practices()

        all_files = {}
        for pattern in patterns:
            for f in pattern.files:
                all_files[f] = all_files.get(f, 0) + 1

        return {
            'total_patterns': len(patterns),
            'anti_patterns': len(anti),
            'best_practices': len(best),
            'common_patterns': [
                {
                    'type': p.pattern_type,
                    'occurrences': p.occurrences,
                    'prevalence': p.team_prevalence,
                    'acceptance_rate': p.acceptance_rate,
                    'anti_pattern': p.anti_pattern,
                    'recommended_fix': p.recommended_fix
                }
                for p in patterns
            ],
            'files_affected': dict(
                sorted(all_files.items(), key=lambda x: x[1], reverse=True)
            )
        }
```

**Test Coverage** (~320 lines, 18+ tests):

```python
class TestPatternLearning:

    def test_detect_common_patterns_basic(self):
        # Create 5+ findings with same title
        # Expect: detected as common pattern

    def test_detect_common_patterns_min_occurrences(self):
        # Create 3 findings of type A, 8 of type B
        # With min_occurrences=5:
        #   Expect: only B returned

    def test_detect_anti_patterns_high_rejection(self):
        # Pattern: 2 accepted, 8 rejected
        # Expect: detected as anti-pattern

    def test_detect_anti_patterns_threshold(self):
        # Use rejection_threshold=0.5
        # Pattern with 60% rejection → included
        # Pattern with 40% rejection → excluded

    def test_detect_best_practices_high_acceptance(self):
        # Pattern: 9 accepted, 1 rejected
        # Expect: detected as best practice

    def test_detect_best_practices_threshold(self):
        # Use acceptance_threshold=0.85
        # Pattern with 90% acceptance → included
        # Pattern with 75% acceptance → excluded

    def test_get_pattern_severity_distribution(self):
        # Create findings: 3 CRITICAL, 2 HIGH, 1 LOW
        # Expect: {'CRITICAL': 3, 'HIGH': 2, 'LOW': 1}

    def test_get_pattern_files_frequency(self):
        # Pattern in: src/a.py (4x), src/b.py (2x), src/c.py (1x)
        # Expect: [src/a.py, src/b.py, src/c.py]

    def test_is_anti_pattern_true(self):
        # Pattern with 75% rejection
        # Expect: True

    def test_is_anti_pattern_false(self):
        # Pattern with 20% rejection
        # Expect: False

    def test_get_fix_patterns_ordering(self):
        # Multiple fixes for same pattern
        # Expect: sorted by acceptance count DESC

    def test_get_fix_patterns_effectiveness(self):
        # Fix A: 8 accepted, 2 rejected → 0.80 effectiveness
        # Fix B: 4 accepted, 1 rejected → 0.80 effectiveness
        # Verify effectiveness calculated correctly

    def test_get_recommended_fix_returns_most_accepted(self):
        # Multiple fixes, A has 8 acceptances, B has 3
        # Expect: A returned

    def test_get_recommended_fix_no_data(self):
        # Pattern never fixed before
        # Expect: None

    def test_persist_pattern_creates_record(self):
        # Call persist_pattern()
        # Verify PatternMetrics created in DB

    def test_persist_all_patterns_coverage(self):
        # Multiple common patterns
        # Call persist_all_patterns()
        # Verify all persisted with count returned

    def test_get_pattern_report_structure(self):
        # Verify report includes all required fields

    def test_get_pattern_report_files_aggregation(self):
        # Multiple patterns, overlapping files
        # Verify files_affected aggregates correctly
```

**Success Criteria**:
- ✅ Pattern detection working (common, anti, best practices)
- ✅ Severity and file distribution calculated correctly
- ✅ Fix pattern analysis tracking acceptance
- ✅ Persistence to PatternMetrics working
- ✅ Report generation complete and accurate
- ✅ 18+ test cases covering all scenarios

---

## Integration Points

### With Task 5.1 (Feedback Collection)
- Use `SuggestionFeedback` table for feedback data
- Query `FeedbackCollector.calculate_acceptance_rate()` for rates
- Use confidence scores from Finding.confidence

### With Task 5.3 (Reporting & Dashboards)
- Expose metrics via `get_calibration_report()`
- Expose ranking via `rank_findings()`
- Expose patterns via `get_pattern_report()`

### With Finding Generation (Phase 6)
- Apply calibrated confidence when creating Finding.confidence
- Use ranking to order PR comments (high-impact first)
- Use patterns to personalize suggestions

---

## Success Criteria Summary

| Subtask | Files | Lines | Tests | Acceptance Criteria |
|---------|-------|-------|-------|-------------------|
| 5.2.1 | 2 | 200 | 5 | Schema complete, dataclasses match |
| 5.2.2 | 2 | 250 | 20 | Analytics accurate, metrics persist |
| 5.2.3 | 2 | 350 | 18 | Calibration working, thresholds valid |
| 5.2.4 | 2 | 250 | 16 | Ranking scores verified, sort correct |
| 5.2.5 | 2 | 300 | 18 | Patterns detected, fixes tracked |
| **Total** | **10** | **1,350** | **77** | **All subtasks complete & integrated** |

---

## Testing Strategy

**Unit Tests** (77 tests, ~2,500 lines)
- Each method tested independently
- Mock database with in-memory SQLite
- Fixture-based setup/teardown

**Integration Tests** (15+ tests, ~500 lines)
- Full workflows: feedback → analytics → ranking → patterns
- End-to-end scenarios
- Report generation accuracy

**Performance Tests** (optional)
- 1000+ findings: analytics speed <1s
- Ranking 500 suggestions: <500ms
- Pattern detection: <2s

---

## Deliverables

1. **Code Implementation** - All 5 subtasks with full test coverage
2. **Database Migrations** - Schema updates for new tables
3. **Integration Tests** - Workflows between components
4. **Documentation** - Docstrings, examples, architecture diagrams
5. **API Endpoints** (Phase 5.3) - Expose reports/rankings to UI

---

## Notes for Implementation

### Order of Implementation
1. Start with 5.2.1 (schema) - foundation for everything
2. Then 5.2.2 (analytics) - needed for tuning & ranking
3. Then 5.2.3 (tuning) - uses analytics
4. Then 5.2.4 (ranking) - uses both analytics and tuning
5. Finally 5.2.5 (patterns) - independent, can parallelize with 4

### Database Considerations
- Add indexes on (category, severity) for query performance
- Use batch operations for metric persistence
- Consider periodic cleanup of old patterns (>6 months)

### Error Handling
- Missing feedback data → graceful defaults (0.5 scores)
- Empty bins in calibration → skip
- Patterns with <min_occurrences → exclude
- No history for finding → use defaults, don't crash

### Code Quality
- Comprehensive type hints throughout
- Docstrings with examples for complex methods
- Consistent naming (snake_case for methods, PascalCase for classes)
- DRY: extract common aggregation patterns
