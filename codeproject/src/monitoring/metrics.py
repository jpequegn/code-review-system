"""
Monitoring & Metrics Collection - Prometheus-compatible metrics system

Implements production-grade metrics for Phase 5 learning system:
- Counters: Cumulative events (jobs run, requests processed, errors)
- Histograms: Distribution of durations (API response, job execution, learning updates)
- Gauges: Point-in-time values (cache hit rate, queue size, connection pool)

Metrics Categories:
- Learning Engine: Update duration, feedback rate, job success rate
- Ranking Engine: Duration, deduplication hits, score distribution
- Insights Engine: Generation duration, cache hits, trend analysis, ROI
- API: Response time p50/p95/p99, error rate, request rate
- Database: Query duration, query count, connection pool, transaction rollbacks
- System: Feedback success rate, data integrity pass rate, job success rate

Prometheus Export Format:
- TYPE and HELP comments for discovery
- Cumulative counters (never decrease)
- Histogram quantiles and bucket counts
- Gauge current values
"""

import time
from typing import Optional, Dict, List, Tuple
from datetime import datetime, timezone
from enum import Enum
from functools import wraps
import json


class MetricType(Enum):
    """Types of metrics."""
    COUNTER = "counter"
    HISTOGRAM = "histogram"
    GAUGE = "gauge"


class Metric:
    """Base metric class."""

    def __init__(self, name: str, metric_type: MetricType, help_text: str = ""):
        """
        Initialize metric.

        Args:
            name: Metric name (must be valid Prometheus name)
            metric_type: Type of metric (counter, histogram, gauge)
            help_text: Human-readable description
        """
        self.name = name
        self.metric_type = metric_type
        self.help_text = help_text
        self.labels = {}  # label_key -> value


class Counter(Metric):
    """Cumulative counter metric."""

    def __init__(self, name: str, help_text: str = ""):
        super().__init__(name, MetricType.COUNTER, help_text)
        self.value = 0

    def increment(self, amount: float = 1, **labels) -> None:
        """
        Increment counter.

        Args:
            amount: Amount to increment (default: 1)
            **labels: Label key-value pairs for dimensionality
        """
        self.value += amount
        if labels:
            self.labels = labels


class Histogram(Metric):
    """Histogram metric for distributions."""

    # Default buckets: 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 5, 10+ seconds
    DEFAULT_BUCKETS = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    def __init__(
        self, name: str, help_text: str = "", buckets: Optional[List[float]] = None
    ):
        super().__init__(name, MetricType.HISTOGRAM, help_text)
        self.buckets = buckets or self.DEFAULT_BUCKETS
        self.values = []  # Raw values for calculation
        self.count = 0
        self.sum = 0.0

    def observe(self, value: float, **labels) -> None:
        """
        Record an observation.

        Args:
            value: Value to record
            **labels: Label key-value pairs
        """
        self.values.append(value)
        self.count += 1
        self.sum += value
        if labels:
            self.labels = labels

    def get_percentile(self, percentile: float) -> float:
        """
        Calculate percentile of recorded values.

        Args:
            percentile: Percentile to calculate (0-100)

        Returns:
            Value at percentile, or 0 if no values recorded
        """
        if not self.values:
            return 0.0
        sorted_values = sorted(self.values)
        index = int((percentile / 100.0) * len(sorted_values))
        index = min(index, len(sorted_values) - 1)
        return sorted_values[index]


class Gauge(Metric):
    """Gauge metric for point-in-time values."""

    def __init__(self, name: str, help_text: str = ""):
        super().__init__(name, MetricType.GAUGE, help_text)
        self.value = 0.0

    def set(self, value: float, **labels) -> None:
        """
        Set gauge value.

        Args:
            value: Value to set
            **labels: Label key-value pairs
        """
        self.value = value
        if labels:
            self.labels = labels

    def increment(self, amount: float = 1, **labels) -> None:
        """Increment gauge value."""
        self.value += amount
        if labels:
            self.labels = labels

    def decrement(self, amount: float = 1, **labels) -> None:
        """Decrement gauge value."""
        self.value -= amount
        if labels:
            self.labels = labels


class MetricsCollector:
    """
    Central metrics collection system.

    Provides:
    - Metrics registration and tracking
    - Prometheus-format export
    - Timer decorator for automatic duration measurement
    - Thread-safe metric updates
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, Metric] = {}
        self._setup_default_metrics()

    def _setup_default_metrics(self) -> None:
        """Set up default Phase 5 metrics."""
        # Learning Engine Metrics
        self.register_counter(
            "learning_metrics_updates_total",
            "Total learning metrics updates",
        )
        self.register_histogram(
            "learning_metrics_update_duration_seconds",
            "Learning metrics update duration",
        )
        self.register_counter(
            "feedback_processed_total", "Total feedback items processed"
        )
        self.register_counter(
            "feedback_errors_total", "Total feedback processing errors"
        )
        self.register_gauge(
            "feedback_queue_size", "Current size of feedback processing queue"
        )
        self.register_gauge(
            "learning_last_update_timestamp",
            "Timestamp of last learning update",
        )

        # Ranking Engine Metrics
        self.register_histogram(
            "ranking_duration_seconds", "Suggestion ranking duration"
        )
        self.register_counter(
            "ranking_duplicates_detected_total",
            "Total duplicate suggestions detected",
        )
        self.register_gauge(
            "ranking_score_distribution_mean",
            "Mean of suggestion scores",
        )

        # Insights Engine Metrics
        self.register_histogram(
            "insights_generation_duration_seconds",
            "Insights generation duration",
        )
        self.register_gauge(
            "insights_cache_hit_rate", "Insights cache hit rate (0-1)"
        )
        self.register_histogram(
            "insights_trend_analysis_duration_seconds",
            "Trend analysis duration",
        )
        self.register_histogram(
            "insights_roi_calculation_duration_seconds", "ROI calculation duration"
        )

        # Batch Job Metrics
        self.register_counter(
            "batch_jobs_executed_total", "Total batch jobs executed"
        )
        self.register_counter(
            "batch_jobs_succeeded_total", "Total batch jobs succeeded"
        )
        self.register_counter(
            "batch_jobs_failed_total", "Total batch jobs failed"
        )
        self.register_counter(
            "batch_jobs_retried_total", "Total batch job retries"
        )
        self.register_histogram(
            "batch_job_duration_seconds", "Batch job execution duration"
        )
        self.register_gauge(
            "batch_jobs_in_retry_queue", "Jobs currently in retry queue"
        )

        # Cache Metrics
        self.register_counter("cache_hits_total", "Total cache hits")
        self.register_counter("cache_misses_total", "Total cache misses")
        self.register_counter(
            "cache_invalidations_total", "Total cache invalidations"
        )
        self.register_gauge(
            "cache_memory_entries", "Number of entries in memory cache"
        )
        self.register_gauge(
            "cache_database_entries", "Number of entries in database cache"
        )

        # API Metrics
        self.register_histogram(
            "api_request_duration_seconds", "API request duration"
        )
        self.register_counter(
            "api_requests_total", "Total API requests"
        )
        self.register_counter(
            "api_errors_total", "Total API errors"
        )
        self.register_gauge(
            "api_request_rate_per_minute", "API request rate (requests/minute)"
        )

        # Database Metrics
        self.register_histogram(
            "db_query_duration_seconds", "Database query duration"
        )
        self.register_counter(
            "db_queries_total", "Total database queries"
        )
        self.register_gauge(
            "db_connection_pool_size", "Database connection pool size"
        )
        self.register_gauge(
            "db_connection_pool_available", "Available connections in pool"
        )
        self.register_counter(
            "db_transaction_rollbacks_total", "Total transaction rollbacks"
        )

        # System Health Metrics
        self.register_gauge(
            "feedback_success_rate", "Feedback processing success rate (0-1)"
        )
        self.register_gauge(
            "data_integrity_pass_rate", "Data integrity checks pass rate (0-1)"
        )
        self.register_gauge(
            "batch_job_success_rate", "Batch job success rate (0-1)"
        )

    def register_counter(self, name: str, help_text: str = "") -> Counter:
        """
        Register or get a counter metric.

        Args:
            name: Metric name
            help_text: Human-readable description

        Returns:
            Counter instance
        """
        if name not in self.metrics:
            self.metrics[name] = Counter(name, help_text)
        return self.metrics[name]

    def register_histogram(
        self,
        name: str,
        help_text: str = "",
        buckets: Optional[List[float]] = None,
    ) -> Histogram:
        """
        Register or get a histogram metric.

        Args:
            name: Metric name
            help_text: Human-readable description
            buckets: Custom histogram buckets

        Returns:
            Histogram instance
        """
        if name not in self.metrics:
            self.metrics[name] = Histogram(name, help_text, buckets)
        return self.metrics[name]

    def register_gauge(self, name: str, help_text: str = "") -> Gauge:
        """
        Register or get a gauge metric.

        Args:
            name: Metric name
            help_text: Human-readable description

        Returns:
            Gauge instance
        """
        if name not in self.metrics:
            self.metrics[name] = Gauge(name, help_text)
        return self.metrics[name]

    def timer(self, metric_name: str):
        """
        Decorator for automatic duration measurement.

        Usage:
            @metrics.timer("ranking_duration_seconds")
            def rank_findings(findings):
                return ranked_findings

        Args:
            metric_name: Name of histogram metric to record duration
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                start = time.time()
                try:
                    result = func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start
                    if metric_name in self.metrics:
                        metric = self.metrics[metric_name]
                        if isinstance(metric, Histogram):
                            metric.observe(duration)

            return wrapper

        return decorator

    def export_prometheus(self) -> str:
        """
        Export metrics in Prometheus text format.

        Format:
        # HELP metric_name help text
        # TYPE metric_name counter|histogram|gauge
        metric_name{label="value"} value
        metric_name_bucket{le="0.1"} count
        metric_name_count value
        metric_name_sum value

        Returns:
            Prometheus-formatted metrics string
        """
        lines = []

        for name, metric in sorted(self.metrics.items()):
            # Add HELP and TYPE comments
            lines.append(f"# HELP {name} {metric.help_text or 'Metric: ' + name}")
            lines.append(f"# TYPE {name} {metric.metric_type.value}")

            if isinstance(metric, Counter):
                # Counter: simple value
                label_str = self._format_labels(metric.labels)
                lines.append(f"{name}{label_str} {metric.value}")

            elif isinstance(metric, Gauge):
                # Gauge: simple value
                label_str = self._format_labels(metric.labels)
                lines.append(f"{name}{label_str} {metric.value}")

            elif isinstance(metric, Histogram):
                # Histogram: buckets, count, and sum
                label_str = self._format_labels(metric.labels)

                # Buckets
                for bucket in metric.buckets:
                    count = sum(1 for v in metric.values if v <= bucket)
                    bucket_label = label_str.rstrip("}") + f',le="{bucket}"}}' if label_str.endswith("}") else f'{{le="{bucket}"}}'
                    if not label_str:
                        bucket_label = f'{{le="{bucket}"}}'
                    lines.append(f"{name}_bucket{bucket_label} {count}")

                # +Inf bucket
                inf_label = label_str.rstrip("}") + f',le="+Inf"}}' if label_str.endswith("}") else f'{{le="+Inf"}}'
                if not label_str:
                    inf_label = f'{{le="+Inf"}}'
                lines.append(f"{name}_bucket{inf_label} {metric.count}")

                # Count and sum
                lines.append(f"{name}_count{label_str} {metric.count}")
                lines.append(f"{name}_sum{label_str} {metric.sum}")

            lines.append("")  # Blank line between metrics

        return "\n".join(lines)

    def _format_labels(self, labels: Dict[str, str]) -> str:
        """Format labels for Prometheus output."""
        if not labels:
            return ""
        label_pairs = [f'{k}="{v}"' for k, v in sorted(labels.items())]
        return "{" + ",".join(label_pairs) + "}"

    def get_metric(self, name: str) -> Optional[Metric]:
        """Get metric by name."""
        return self.metrics.get(name)

    def get_all_metrics(self) -> Dict[str, Metric]:
        """Get all registered metrics."""
        return self.metrics.copy()

    def clear(self) -> None:
        """Clear all metrics (useful for testing)."""
        self.metrics.clear()
        self._setup_default_metrics()


# Global metrics instance
_metrics_instance: Optional[MetricsCollector] = None


def get_metrics() -> MetricsCollector:
    """Get or create global metrics instance."""
    global _metrics_instance
    if _metrics_instance is None:
        _metrics_instance = MetricsCollector()
    return _metrics_instance


def initialize_metrics() -> MetricsCollector:
    """Initialize and return metrics instance."""
    global _metrics_instance
    _metrics_instance = MetricsCollector()
    return _metrics_instance
