"""
Prediction and Trend Intelligence Module

Provides predictive analysis of code quality, failure likelihood,
and technical debt accumulation based on historical patterns.
"""

from src.prediction.history_tracker import (
    CodeQualitySnapshot,
    HistoryDatabase,
)
from src.prediction.risk_scorer import (
    RiskScore,
    RiskScorer,
    RiskLevel,
)
from src.prediction.pattern_learner import (
    PersonalPattern,
    PatternLearner,
)
from src.prediction.failure_predictor import (
    FailurePrediction,
    FailurePredictor,
)
from src.prediction.temporal_analysis import (
    TemporalPattern,
    TemporalAnalyzer,
)
from src.prediction.debt_predictor import (
    DebtMetrics,
    DebtPredictor,
)

__all__ = [
    "CodeQualitySnapshot",
    "HistoryDatabase",
    "RiskScore",
    "RiskScorer",
    "RiskLevel",
    "PersonalPattern",
    "PatternLearner",
    "FailurePrediction",
    "FailurePredictor",
    "TemporalPattern",
    "TemporalAnalyzer",
    "DebtMetrics",
    "DebtPredictor",
]
