"""
Learning Metrics - Data structures for analytics and learning components.

Provides type-safe dataclasses for:
- Acceptance rate analytics
- Confidence calibration by score bins
- Pattern metrics and detection
"""

from dataclasses import dataclass
from typing import Optional


# ============================================================================
# Acceptance Rate Analytics
# ============================================================================


@dataclass
class AcceptanceMetrics:
    """
    Acceptance rates and metrics for a specific finding category.

    Aggregates feedback data to understand which types of findings
    are accepted vs rejected by developers.
    """

    finding_category: str
    """Type of finding (e.g., "SQL Injection", "Resource Leak")"""

    severity: str
    """Severity level (CRITICAL, HIGH, MEDIUM, LOW)"""

    total: int
    """Total number of findings in this category"""

    accepted: int
    """Number of findings that were accepted/applied"""

    rejected: int
    """Number of findings that were rejected/ignored"""

    ignored: int
    """Number of findings with no feedback"""

    acceptance_rate: float
    """Acceptance rate as decimal 0.0-1.0 (accepted / (accepted + rejected))"""

    confidence_avg: float
    """Average confidence score for findings in this category"""

    fix_rate: float
    """Percentage of suggestions that were actually applied (0.0-1.0)"""

    def __post_init__(self):
        """Validate metrics."""
        if not (0.0 <= self.acceptance_rate <= 1.0):
            raise ValueError(f"acceptance_rate must be 0.0-1.0, got {self.acceptance_rate}")
        if not (0.0 <= self.confidence_avg <= 1.0):
            raise ValueError(f"confidence_avg must be 0.0-1.0, got {self.confidence_avg}")
        if not (0.0 <= self.fix_rate <= 1.0):
            raise ValueError(f"fix_rate must be 0.0-1.0, got {self.fix_rate}")


# ============================================================================
# Confidence Calibration
# ============================================================================


@dataclass
class ConfidenceBin:
    """
    Calibration metrics for a single confidence score bin (0.0-0.1, 0.1-0.2, etc).

    Used to recalibrate confidence thresholds based on actual feedback outcomes.
    Helps translate "model confidence" to "real-world acceptance probability".
    """

    bin_id: int
    """Bin number 0-9 representing ranges 0.0-0.1, 0.1-0.2, ..., 0.9-1.0"""

    confidence_range: str
    """Human-readable range e.g. "0.8-0.9" """

    original_acceptance_rate: float
    """Acceptance rate at time of prediction (what LLM saw in training)"""

    actual_acceptance_rate: float
    """Real acceptance rate from actual feedback in this confidence bin"""

    precision: float
    """Of accepted suggestions, how many were correct (TP / (TP + FP))"""

    recall: float
    """Of all correct suggestions, how many were found (TP / (TP + FN))"""

    f1_score: float
    """Harmonic mean: 2 * (precision * recall) / (precision + recall)"""

    calibrated_threshold: float
    """Adjusted confidence threshold for this bin based on actual performance"""

    sample_size: int
    """Number of suggestions in this confidence bin"""

    def __post_init__(self):
        """Validate calibration metrics."""
        if not (0 <= self.bin_id <= 9):
            raise ValueError(f"bin_id must be 0-9, got {self.bin_id}")
        if not (0.0 <= self.actual_acceptance_rate <= 1.0):
            raise ValueError(f"actual_acceptance_rate must be 0.0-1.0, got {self.actual_acceptance_rate}")
        if not (0.0 <= self.precision <= 1.0):
            raise ValueError(f"precision must be 0.0-1.0, got {self.precision}")
        if not (0.0 <= self.recall <= 1.0):
            raise ValueError(f"recall must be 0.0-1.0, got {self.recall}")
        if not (0.0 <= self.f1_score <= 1.0):
            raise ValueError(f"f1_score must be 0.0-1.0, got {self.f1_score}")
        if not (0.0 <= self.calibrated_threshold <= 1.0):
            raise ValueError(f"calibrated_threshold must be 0.0-1.0, got {self.calibrated_threshold}")
        if self.sample_size < 0:
            raise ValueError(f"sample_size must be >= 0, got {self.sample_size}")


# ============================================================================
# Pattern Detection
# ============================================================================


@dataclass
class PatternInfo:
    """
    Information about a detected pattern in the codebase.

    Patterns can be:
    - Anti-patterns: recurring issues developers keep encountering
    - Best practices: patterns with high acceptance/fix rates
    - Common patterns: frequently occurring issues
    """

    pattern_type: str
    """Type/name of pattern (e.g., "unclosed_file_handle", "n_plus_one_query")"""

    occurrences: int
    """Number of times this pattern has been detected"""

    files: list[str]
    """List of file paths where this pattern was found"""

    avg_severity: float
    """Average severity score 0.0-1.0 where 1.0 is most severe"""

    acceptance_rate: float
    """Percentage of suggestions for this pattern that were accepted (0.0-1.0)"""

    fix_count: int
    """Number of times this pattern was actually fixed"""

    anti_pattern: bool
    """True if this is a pattern developers should avoid (high rejection rate)"""

    team_prevalence: str
    """Prevalence level: 'rare' (<3), 'occasional' (3-10), 'common' (>10)"""

    recommended_fix: str
    """Template or recommendation for fixing this pattern"""

    def __post_init__(self):
        """Validate pattern info."""
        if self.occurrences < 1:
            raise ValueError(f"occurrences must be >= 1, got {self.occurrences}")
        if not (0.0 <= self.avg_severity <= 1.0):
            raise ValueError(f"avg_severity must be 0.0-1.0, got {self.avg_severity}")
        if not (0.0 <= self.acceptance_rate <= 1.0):
            raise ValueError(f"acceptance_rate must be 0.0-1.0, got {self.acceptance_rate}")
        if self.fix_count < 0:
            raise ValueError(f"fix_count must be >= 0, got {self.fix_count}")
        if self.team_prevalence not in ('rare', 'occasional', 'common'):
            raise ValueError(
                f"team_prevalence must be 'rare', 'occasional', or 'common', "
                f"got '{self.team_prevalence}'"
            )
        if not self.files:
            raise ValueError("files list cannot be empty")
        if not self.recommended_fix:
            raise ValueError("recommended_fix cannot be empty")


# ============================================================================
# Analysis Results
# ============================================================================


@dataclass
class FixPattern:
    """
    How a specific pattern is usually fixed in the team's codebase.

    Tracks effectiveness of different suggested fixes for the same issue.
    """

    suggestion_fix: str
    """The suggested fix text/template"""

    acceptance_count: int
    """Number of times this fix was accepted"""

    rejection_count: int
    """Number of times this fix was rejected"""

    fix_time_avg_hours: float
    """Average time from suggestion to actual fix in hours"""

    effectiveness: float
    """Percentage of acceptances (0.0-1.0)"""

    def __post_init__(self):
        """Validate fix pattern."""
        if not self.suggestion_fix:
            raise ValueError("suggestion_fix cannot be empty")
        if self.acceptance_count < 0:
            raise ValueError(f"acceptance_count must be >= 0, got {self.acceptance_count}")
        if self.rejection_count < 0:
            raise ValueError(f"rejection_count must be >= 0, got {self.rejection_count}")
        if self.fix_time_avg_hours < 0:
            raise ValueError(f"fix_time_avg_hours must be >= 0, got {self.fix_time_avg_hours}")
        if not (0.0 <= self.effectiveness <= 1.0):
            raise ValueError(f"effectiveness must be 0.0-1.0, got {self.effectiveness}")


@dataclass
class CalibrationReport:
    """
    Comprehensive calibration report showing before/after for all bins.

    Used to understand how well confidence scores align with actual outcomes.
    """

    bins: list[ConfidenceBin]
    """Calibration data for all 10 confidence bins"""

    recommended_thresholds: dict[str, float]
    """
    Suggested confidence thresholds for different strategies:
    - 'aggressive': Lower threshold, catch more issues (accept more false positives)
    - 'balanced': Middle threshold (default), balanced precision/recall
    - 'conservative': Higher threshold, only high-confidence suggestions (strict precision)
    """

    improvement_summary: str
    """Summary of calibration improvements and recommendations"""

    def __post_init__(self):
        """Validate report."""
        if len(self.bins) != 10:
            raise ValueError(f"Must have exactly 10 bins, got {len(self.bins)}")
        required_thresholds = {'aggressive', 'balanced', 'conservative'}
        if set(self.recommended_thresholds.keys()) != required_thresholds:
            raise ValueError(
                f"Must have thresholds for {required_thresholds}, "
                f"got {set(self.recommended_thresholds.keys())}"
            )
        for key, value in self.recommended_thresholds.items():
            if not (0.0 <= value <= 1.0):
                raise ValueError(
                    f"Threshold '{key}' must be 0.0-1.0, got {value}"
                )


@dataclass
class PatternReport:
    """
    Comprehensive pattern analysis report.

    Shows all detected patterns, organized by prevalence and effectiveness.
    """

    total_patterns: int
    """Total unique patterns detected"""

    anti_patterns: int
    """Number of anti-patterns (patterns to avoid)"""

    best_practices: int
    """Number of best practices (high acceptance rate)"""

    common_patterns: list[dict]
    """List of patterns with metadata, sorted by occurrence"""

    files_affected: dict[str, int]
    """File paths and how many patterns found in each file"""

    def __post_init__(self):
        """Validate report."""
        if self.total_patterns < 0:
            raise ValueError(f"total_patterns must be >= 0, got {self.total_patterns}")
        if self.anti_patterns < 0:
            raise ValueError(f"anti_patterns must be >= 0, got {self.anti_patterns}")
        if self.best_practices < 0:
            raise ValueError(f"best_practices must be >= 0, got {self.best_practices}")
        if self.anti_patterns + self.best_practices > self.total_patterns:
            raise ValueError(
                "anti_patterns + best_practices cannot exceed total_patterns"
            )


# ============================================================================
# Ranking Results
# ============================================================================


@dataclass
class RankingScores:
    """
    Individual component scores for a ranked finding.

    Shows breakdown of how a finding's overall ranking score was calculated.
    Useful for transparency and debugging ranking decisions.
    """

    confidence: float
    """Calibrated confidence score (0.0-1.0)"""

    acceptance_rate: float
    """Historical acceptance rate for this type of finding (0.0-1.0)"""

    impact_score: float
    """Impact = severity × prevalence (0.0-1.0)"""

    fix_time: float
    """Fix time score (0.0-1.0, higher = faster to fix)"""

    team_preference: float
    """Learned team preference (0.0-1.0)"""

    total_score: float
    """Weighted composite score (0.0-1.0)"""

    def __post_init__(self):
        """Validate scores."""
        for name, value in [
            ('confidence', self.confidence),
            ('acceptance_rate', self.acceptance_rate),
            ('impact_score', self.impact_score),
            ('fix_time', self.fix_time),
            ('team_preference', self.team_preference),
            ('total_score', self.total_score),
        ]:
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"{name} must be 0.0-1.0, got {value}")
