"""
Confidence Calibration & Tuning - Recalibrate confidence scores based on feedback

Problem: LLM confidence scores don't match real-world acceptance rates
Solution: Measure actual acceptance in confidence bins, recalibrate thresholds
"""

import math
from typing import Optional
from datetime import datetime, timezone
from sqlalchemy.orm import Session
from sqlalchemy import func

from src.database import (
    Finding,
    SuggestionFeedback,
    ConfidenceCalibration,
)
from src.learning.metrics import ConfidenceBin, CalibrationReport


class ConfidenceTuner:
    """
    Recalibrate confidence scores based on actual acceptance outcomes.

    The core problem: An LLM assigns confidence 0.95 to a suggestion, but
    developers only accept 40% of suggestions with that confidence. This means
    the confidence is miscalibrated.

    Solution: Create 10 confidence bins (0.0-0.1, 0.1-0.2, ..., 0.9-1.0),
    measure actual acceptance in each bin, and compute new thresholds.

    Example:
        tuner = ConfidenceTuner(db_session)

        # Measure actual acceptance in each bin
        bins = tuner.calculate_all_bin_statistics()
        for bin_stats in bins:
            print(f"Bin {bin_stats.bin_id}: {bin_stats.actual_acceptance_rate:.1%}")

        # Suggest new thresholds
        thresholds = tuner.suggest_calibrated_thresholds()
        print(f"Use threshold {thresholds['balanced']:.2f} for balanced approach")

        # Apply to new predictions
        original_confidence = 0.95
        calibrated = tuner.apply_calibration_to_finding(original_confidence)
    """

    def __init__(self, db: Session):
        """
        Initialize tuner with database session.

        Args:
            db: SQLAlchemy database session
        """
        self.db = db
        self.confidence_bins = 10  # 0.0-0.1, 0.1-0.2, ..., 0.9-1.0

    # ============================================================================
    # Confidence Binning
    # ============================================================================

    def bin_confidence(self, confidence: float) -> int:
        """
        Convert confidence score to bin number (0-9).

        Maps confidence to bin:
        - 0.0-0.1 → bin 0
        - 0.1-0.2 → bin 1
        - ...
        - 0.9-1.0 → bin 9

        Args:
            confidence: Confidence score (0.0-1.0)

        Returns:
            Bin ID (0-9)
        """
        bin_id = min(int(confidence * 10), 9)
        return max(0, bin_id)

    def get_confidence_range_for_bin(self, bin_id: int) -> tuple[float, float]:
        """
        Get minimum and maximum confidence for a bin.

        Args:
            bin_id: Bin number (0-9)

        Returns:
            (min_confidence, max_confidence) tuple
        """
        return (bin_id * 0.1, (bin_id + 1) * 0.1)

    # ============================================================================
    # Bin Statistics Calculation
    # ============================================================================

    def calculate_bin_statistics(self, bin_id: int) -> ConfidenceBin:
        """
        Calculate calibration metrics for a confidence bin.

        For all SuggestionFeedback in this bin, calculate:
        - Sample size (how many suggestions in this bin)
        - Original acceptance rate (reported when suggestion was made)
        - Actual acceptance rate (what feedback shows)
        - Precision: TP / (TP + FP) = accepted / (accepted + rejected)
        - Recall: TP / (TP + FN) = accepted / (accepted + no_feedback)
        - F1 Score: 2 * (precision * recall) / (precision + recall)

        Args:
            bin_id: Confidence bin (0-9)

        Returns:
            ConfidenceBin with calibration metrics
        """
        min_conf, max_conf = self.get_confidence_range_for_bin(bin_id)

        # Query all feedback in this confidence range
        feedback_list = (
            self.db.query(SuggestionFeedback)
            .filter(
                SuggestionFeedback.confidence >= min_conf,
                SuggestionFeedback.confidence < max_conf,
            )
            .all()
        )

        sample_size = len(feedback_list)

        if sample_size == 0:
            # Empty bin - return zero metrics
            confidence_range = f"{min_conf:.1f}-{max_conf:.1f}"
            return ConfidenceBin(
                bin_id=bin_id,
                confidence_range=confidence_range,
                original_acceptance_rate=0.0,
                actual_acceptance_rate=0.0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                calibrated_threshold=min_conf,
                sample_size=0,
            )

        # Count acceptance outcomes
        accepted = sum(1 for f in feedback_list if f.feedback_type == "ACCEPTED")
        rejected = sum(1 for f in feedback_list if f.feedback_type == "REJECTED")

        # Calculate rates
        total_with_feedback = accepted + rejected
        actual_acceptance_rate = (
            accepted / total_with_feedback if total_with_feedback > 0 else 0.0
        )

        # Precision: of accepted suggestions, how many were correct
        # In this context: accepted / (accepted + rejected)
        precision = (
            accepted / total_with_feedback if total_with_feedback > 0 else 0.0
        )

        # Recall: of all suggestions, how many were accepted
        # In this context: accepted / sample_size
        recall = accepted / sample_size if sample_size > 0 else 0.0

        # F1 Score: harmonic mean of precision and recall
        if precision + recall > 0:
            f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            f1_score = 0.0

        # Calibrated threshold: use actual acceptance rate
        calibrated_threshold = actual_acceptance_rate

        confidence_range = f"{min_conf:.1f}-{max_conf:.1f}"

        return ConfidenceBin(
            bin_id=bin_id,
            confidence_range=confidence_range,
            original_acceptance_rate=min_conf,  # Original confidence range start
            actual_acceptance_rate=actual_acceptance_rate,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            calibrated_threshold=calibrated_threshold,
            sample_size=sample_size,
        )

    def calculate_all_bin_statistics(self) -> list[ConfidenceBin]:
        """
        Calculate calibration statistics for all 10 confidence bins.

        Returns:
            List of ConfidenceBin objects for bins 0-9
        """
        return [self.calculate_bin_statistics(i) for i in range(10)]

    # ============================================================================
    # Threshold Computation
    # ============================================================================

    def compute_optimal_threshold(
        self, target_precision: float = 0.85, target_recall: float = 0.70
    ) -> ConfidenceBin:
        """
        Find confidence bin that best balances precision and recall.

        Strategy:
        1. Calculate metrics for all bins
        2. Score each bin by distance to (target_precision, target_recall)
        3. Return bin with best score

        Args:
            target_precision: Minimum acceptable precision (0.0-1.0)
            target_recall: Minimum acceptable recall (0.0-1.0)

        Returns:
            ConfidenceBin of optimal threshold
        """
        bins = self.calculate_all_bin_statistics()

        best_bin = None
        best_score = float("inf")

        for bin_stats in bins:
            # Score: euclidean distance to target point
            precision_diff = (bin_stats.precision - target_precision) ** 2
            recall_diff = (bin_stats.recall - target_recall) ** 2
            score = math.sqrt(precision_diff + recall_diff)

            # Prefer bins that meet minimum thresholds
            if (
                bin_stats.precision >= target_precision
                and bin_stats.recall >= target_recall
            ):
                # Strong preference for bins that meet both targets
                score *= 0.5

            if score < best_score:
                best_score = score
                best_bin = bin_stats

        return best_bin if best_bin else bins[5]  # Default to middle bin

    def suggest_calibrated_thresholds(self) -> dict[str, float]:
        """
        Suggest confidence thresholds for different use cases.

        Returns three thresholds:
        - aggressive: Lower threshold, catch more issues (tolerate false positives)
        - balanced: Default, balance precision and recall
        - conservative: Higher threshold, only high-confidence (strict precision)

        Returns:
            Dict with keys: 'aggressive', 'balanced', 'conservative'
        """
        bins = self.calculate_all_bin_statistics()

        # aggressive: lowest threshold where precision > 0.70
        aggressive_threshold = 0.5
        for bin_stats in bins:
            if bin_stats.precision > 0.70:
                aggressive_threshold = bin_stats.calibrated_threshold
                break

        # balanced: where F1 is maximized (balance precision and recall)
        balanced_bin = max(bins, key=lambda b: b.f1_score)
        balanced_threshold = balanced_bin.calibrated_threshold

        # conservative: where precision > 0.95
        conservative_threshold = 0.9
        for bin_stats in reversed(bins):
            if bin_stats.precision > 0.95:
                conservative_threshold = bin_stats.calibrated_threshold
                break

        return {
            "aggressive": aggressive_threshold,
            "balanced": balanced_threshold,
            "conservative": conservative_threshold,
        }

    # ============================================================================
    # Persistence & Application
    # ============================================================================

    def persist_calibration(self, bin_id: int) -> ConfidenceCalibration:
        """
        Calculate and save calibration for one bin.

        Creates or updates ConfidenceCalibration record in database.

        Args:
            bin_id: Confidence bin (0-9)

        Returns:
            ConfidenceCalibration record
        """
        bin_stats = self.calculate_bin_statistics(bin_id)

        # Get or create record
        record = (
            self.db.query(ConfidenceCalibration)
            .filter(ConfidenceCalibration.bin_id == bin_id)
            .first()
        )

        if not record:
            record = ConfidenceCalibration(
                bin_id=bin_id,
                original_confidence_range=bin_stats.confidence_range,
                sample_size=bin_stats.sample_size,
                original_acceptance_rate=bin_stats.original_acceptance_rate,
                actual_acceptance_rate=bin_stats.actual_acceptance_rate,
                precision=bin_stats.precision,
                recall=bin_stats.recall,
                f1_score=bin_stats.f1_score,
                calibrated_threshold=bin_stats.calibrated_threshold,
            )
            self.db.add(record)
        else:
            # Update existing
            record.sample_size = bin_stats.sample_size
            record.original_acceptance_rate = bin_stats.original_acceptance_rate
            record.actual_acceptance_rate = bin_stats.actual_acceptance_rate
            record.precision = bin_stats.precision
            record.recall = bin_stats.recall
            record.f1_score = bin_stats.f1_score
            record.calibrated_threshold = bin_stats.calibrated_threshold
            record.last_updated = datetime.now(timezone.utc)

        self.db.commit()
        return record

    def persist_all_calibrations(self) -> int:
        """
        Recalibrate all 10 bins and persist to database.

        Returns:
            Count of records created/updated
        """
        count = 0
        for bin_id in range(10):
            self.persist_calibration(bin_id)
            count += 1
        return count

    def apply_calibration_to_finding(
        self, original_confidence: float, recalibration_mode: str = "balanced"
    ) -> float:
        """
        Apply calibration to adjust confidence score.

        Converts original LLM confidence to calibrated confidence based on
        actual acceptance outcomes in that confidence bin.

        Example:
        - Original confidence: 0.95 (bin 9)
        - Bin 9 actual acceptance rate: 0.82
        - Modes:
          - balanced: return 0.82
          - aggressive: return 0.82 * 1.1 = 0.902
          - conservative: return 0.82 * 0.9 = 0.738

        Args:
            original_confidence: Original confidence score (0.0-1.0)
            recalibration_mode: 'aggressive', 'balanced', or 'conservative'

        Returns:
            Calibrated confidence score
        """
        bin_id = self.bin_confidence(original_confidence)
        calib = (
            self.db.query(ConfidenceCalibration)
            .filter(ConfidenceCalibration.bin_id == bin_id)
            .first()
        )

        if not calib:
            # No calibration data - fallback to original
            return original_confidence

        actual_rate = calib.actual_acceptance_rate

        if recalibration_mode == "aggressive":
            # Boost score slightly to catch more issues
            return min(actual_rate * 1.1, 1.0)
        elif recalibration_mode == "conservative":
            # Reduce score slightly to be more strict
            return max(actual_rate * 0.9, 0.0)
        else:  # balanced (default)
            # Use actual acceptance rate directly
            return actual_rate

    # ============================================================================
    # Reporting
    # ============================================================================

    def get_calibration_report(self) -> dict:
        """
        Generate comprehensive calibration report.

        Includes:
        - Calibration metrics for each of 10 bins
        - Recommended thresholds for different strategies
        - Summary of improvements and recommendations

        Returns:
            Dict with keys: 'bins', 'recommended_thresholds', 'improvement_summary'
        """
        bins = self.calculate_all_bin_statistics()
        thresholds = self.suggest_calibrated_thresholds()

        # Compute improvement summary
        avg_precision = sum(b.precision for b in bins) / len(bins)
        avg_f1 = sum(b.f1_score for b in bins) / len(bins)

        improvement_summary = (
            f"Calibration report based on {sum(b.sample_size for b in bins)} feedback items. "
            f"Average precision across bins: {avg_precision:.1%}. "
            f"Average F1 score: {avg_f1:.2f}. "
            f"Recommended threshold: {thresholds['balanced']:.2f} (balanced approach)."
        )

        return {
            "bins": [
                {
                    "bin_id": b.bin_id,
                    "confidence_range": b.confidence_range,
                    "sample_size": b.sample_size,
                    "actual_acceptance_rate": b.actual_acceptance_rate,
                    "precision": b.precision,
                    "recall": b.recall,
                    "f1_score": b.f1_score,
                    "calibrated_threshold": b.calibrated_threshold,
                }
                for b in bins
            ],
            "recommended_thresholds": thresholds,
            "improvement_summary": improvement_summary,
        }
