"""
Tests for monitoring and metrics collection system.

Coverage:
- Counter, Histogram, Gauge metrics
- MetricsCollector registration and tracking
- Prometheus format export
- Timer decorator for duration measurement
- Global metrics instance management
"""

import pytest
import time
from src.monitoring import (
    MetricsCollector,
    Counter,
    Histogram,
    Gauge,
    get_metrics,
    initialize_metrics,
)


class TestCounter:
    """Test counter metrics."""

    def test_counter_initialization(self):
        """Counter: initializes with zero value."""
        counter = Counter("test_counter", "Test counter")
        assert counter.value == 0
        assert counter.name == "test_counter"

    def test_counter_increment(self):
        """Counter: increments by 1 by default."""
        counter = Counter("test_counter")
        counter.increment()
        assert counter.value == 1

    def test_counter_increment_with_amount(self):
        """Counter: increments by specified amount."""
        counter = Counter("test_counter")
        counter.increment(5)
        assert counter.value == 5
        counter.increment(3)
        assert counter.value == 8

    def test_counter_with_labels(self):
        """Counter: stores labels."""
        counter = Counter("test_counter")
        counter.increment(1, status="success")
        assert counter.labels == {"status": "success"}


class TestHistogram:
    """Test histogram metrics."""

    def test_histogram_initialization(self):
        """Histogram: initializes with default buckets."""
        histogram = Histogram("test_histogram", "Test histogram")
        assert histogram.count == 0
        assert histogram.sum == 0.0
        assert histogram.buckets == Histogram.DEFAULT_BUCKETS

    def test_histogram_custom_buckets(self):
        """Histogram: accepts custom buckets."""
        custom_buckets = [0.1, 0.5, 1.0]
        histogram = Histogram("test_histogram", buckets=custom_buckets)
        assert histogram.buckets == custom_buckets

    def test_histogram_observe(self):
        """Histogram: records observations."""
        histogram = Histogram("test_histogram")
        histogram.observe(0.5)
        histogram.observe(1.2)
        histogram.observe(0.1)

        assert histogram.count == 3
        assert histogram.sum == 1.8
        assert histogram.values == [0.5, 1.2, 0.1]

    def test_histogram_percentiles(self):
        """Histogram: calculates percentiles correctly."""
        histogram = Histogram("test_histogram")
        for i in range(100):
            histogram.observe(i / 100.0)  # 0, 0.01, 0.02, ..., 0.99

        assert histogram.get_percentile(50) == pytest.approx(0.5, abs=0.01)
        assert histogram.get_percentile(95) == pytest.approx(0.95, abs=0.01)
        assert histogram.get_percentile(99) == pytest.approx(0.99, abs=0.01)

    def test_histogram_percentile_empty(self):
        """Histogram: returns 0 for percentile when empty."""
        histogram = Histogram("test_histogram")
        assert histogram.get_percentile(50) == 0.0


class TestGauge:
    """Test gauge metrics."""

    def test_gauge_initialization(self):
        """Gauge: initializes with zero value."""
        gauge = Gauge("test_gauge", "Test gauge")
        assert gauge.value == 0.0

    def test_gauge_set(self):
        """Gauge: sets value."""
        gauge = Gauge("test_gauge")
        gauge.set(42.5)
        assert gauge.value == 42.5

    def test_gauge_increment(self):
        """Gauge: increments value."""
        gauge = Gauge("test_gauge")
        gauge.set(10)
        gauge.increment(5)
        assert gauge.value == 15

    def test_gauge_decrement(self):
        """Gauge: decrements value."""
        gauge = Gauge("test_gauge")
        gauge.set(10)
        gauge.decrement(3)
        assert gauge.value == 7


class TestMetricsCollector:
    """Test metrics collector."""

    def test_collector_initialization(self):
        """Collector: initializes with default metrics."""
        collector = MetricsCollector()
        assert len(collector.metrics) > 0
        assert "cache_hits_total" in collector.metrics
        assert "batch_jobs_executed_total" in collector.metrics

    def test_register_counter(self):
        """Collector: registers counters."""
        collector = MetricsCollector()
        counter = collector.register_counter("test_counter", "Test counter")
        assert isinstance(counter, Counter)
        assert collector.metrics["test_counter"] is counter

    def test_register_histogram(self):
        """Collector: registers histograms."""
        collector = MetricsCollector()
        histogram = collector.register_histogram("test_histogram", "Test histogram")
        assert isinstance(histogram, Histogram)
        assert collector.metrics["test_histogram"] is histogram

    def test_register_gauge(self):
        """Collector: registers gauges."""
        collector = MetricsCollector()
        gauge = collector.register_gauge("test_gauge", "Test gauge")
        assert isinstance(gauge, Gauge)
        assert collector.metrics["test_gauge"] is gauge

    def test_get_metric(self):
        """Collector: retrieves metrics by name."""
        collector = MetricsCollector()
        collector.register_counter("my_counter", "My counter")
        metric = collector.get_metric("my_counter")
        assert metric is not None
        assert metric.name == "my_counter"

    def test_get_all_metrics(self):
        """Collector: returns copy of all metrics."""
        collector = MetricsCollector()
        all_metrics = collector.get_all_metrics()
        assert len(all_metrics) > 0
        # Verify it's a copy
        all_metrics["new_metric"] = "test"
        assert "new_metric" not in collector.metrics

    def test_clear_metrics(self):
        """Collector: clears and reinitializes metrics."""
        collector = MetricsCollector()
        original_count = len(collector.metrics)
        collector.metrics["custom"] = Counter("custom")
        assert len(collector.metrics) > original_count

        collector.clear()
        assert len(collector.metrics) == original_count
        assert "custom" not in collector.metrics


class TestPrometheusExport:
    """Test Prometheus format export."""

    def test_export_counter(self):
        """Export: formats counters correctly."""
        collector = MetricsCollector()
        collector.metrics.clear()
        counter = collector.register_counter("test_counter", "A test counter")
        counter.increment(42)

        export = collector.export_prometheus()
        assert "# HELP test_counter A test counter" in export
        assert "# TYPE test_counter counter" in export
        assert "test_counter 42" in export

    def test_export_gauge(self):
        """Export: formats gauges correctly."""
        collector = MetricsCollector()
        collector.metrics.clear()
        gauge = collector.register_gauge("test_gauge", "A test gauge")
        gauge.set(3.14)

        export = collector.export_prometheus()
        assert "# HELP test_gauge A test gauge" in export
        assert "# TYPE test_gauge gauge" in export
        assert "test_gauge 3.14" in export

    def test_export_histogram(self):
        """Export: formats histograms with buckets."""
        collector = MetricsCollector()
        collector.metrics.clear()
        histogram = collector.register_histogram(
            "test_histogram", "A test histogram", buckets=[0.1, 0.5, 1.0]
        )
        histogram.observe(0.05)
        histogram.observe(0.3)
        histogram.observe(0.8)

        export = collector.export_prometheus()
        assert "# HELP test_histogram A test histogram" in export
        assert "# TYPE test_histogram histogram" in export
        assert 'test_histogram_bucket{le="0.1"}' in export
        assert 'test_histogram_bucket{le="0.5"}' in export
        assert 'test_histogram_bucket{le="1.0"}' in export
        assert 'test_histogram_bucket{le="+Inf"}' in export
        assert "test_histogram_count" in export
        assert "test_histogram_sum" in export

    def test_export_with_labels(self):
        """Export: formats labels correctly."""
        collector = MetricsCollector()
        collector.metrics.clear()
        counter = collector.register_counter("test_counter")
        counter.increment(1, status="success", region="us-west")

        export = collector.export_prometheus()
        assert 'test_counter{region="us-west",status="success"}' in export

    def test_export_is_sorted(self):
        """Export: sorts metrics by name."""
        collector = MetricsCollector()
        collector.metrics.clear()
        collector.register_counter("zzz_counter")
        collector.register_counter("aaa_counter")
        collector.register_counter("mmm_counter")

        export = collector.export_prometheus()
        aaa_pos = export.find("aaa_counter")
        mmm_pos = export.find("mmm_counter")
        zzz_pos = export.find("zzz_counter")

        assert aaa_pos < mmm_pos < zzz_pos


class TestTimerDecorator:
    """Test automatic timer decorator."""

    def test_timer_decorator_records_duration(self):
        """Timer: records function execution duration."""
        collector = MetricsCollector()
        collector.metrics.clear()
        collector.register_histogram("test_duration", "Test duration")

        @collector.timer("test_duration")
        def slow_function():
            time.sleep(0.01)
            return "result"

        result = slow_function()
        assert result == "result"

        histogram = collector.get_metric("test_duration")
        assert histogram.count == 1
        assert histogram.sum >= 0.01

    def test_timer_decorator_with_exception(self):
        """Timer: still records duration even if exception raised."""
        collector = MetricsCollector()
        collector.metrics.clear()
        collector.register_histogram("test_duration", "Test duration")

        @collector.timer("test_duration")
        def failing_function():
            time.sleep(0.01)
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()

        histogram = collector.get_metric("test_duration")
        assert histogram.count == 1
        assert histogram.sum >= 0.01


class TestGlobalMetricsInstance:
    """Test global metrics instance management."""

    def test_get_metrics_returns_singleton(self):
        """Global: get_metrics returns same instance."""
        metrics1 = get_metrics()
        metrics2 = get_metrics()
        assert metrics1 is metrics2

    def test_initialize_metrics_creates_new(self):
        """Global: initialize_metrics creates fresh instance."""
        # Get initial instance
        metrics1 = get_metrics()
        original_count = len(metrics1.metrics)

        # Add custom metric
        metrics1.register_counter("custom_metric")
        assert len(metrics1.metrics) > original_count

        # Initialize fresh
        metrics2 = initialize_metrics()
        assert len(metrics2.metrics) == original_count
        assert "custom_metric" not in metrics2.metrics


class TestMetricsIntegration:
    """Integration tests for metrics collection."""

    def test_learning_engine_metrics_present(self):
        """Integration: learning engine metrics are registered."""
        collector = MetricsCollector()
        assert "learning_metrics_updates_total" in collector.metrics
        assert "feedback_processed_total" in collector.metrics
        assert "feedback_queue_size" in collector.metrics

    def test_ranking_engine_metrics_present(self):
        """Integration: ranking engine metrics are registered."""
        collector = MetricsCollector()
        assert "ranking_duration_seconds" in collector.metrics
        assert "ranking_duplicates_detected_total" in collector.metrics
        assert "ranking_score_distribution_mean" in collector.metrics

    def test_insights_engine_metrics_present(self):
        """Integration: insights engine metrics are registered."""
        collector = MetricsCollector()
        assert "insights_generation_duration_seconds" in collector.metrics
        assert "insights_cache_hit_rate" in collector.metrics
        assert "insights_trend_analysis_duration_seconds" in collector.metrics

    def test_batch_job_metrics_present(self):
        """Integration: batch job metrics are registered."""
        collector = MetricsCollector()
        assert "batch_jobs_executed_total" in collector.metrics
        assert "batch_jobs_succeeded_total" in collector.metrics
        assert "batch_jobs_failed_total" in collector.metrics
        assert "batch_jobs_retried_total" in collector.metrics
        assert "batch_job_duration_seconds" in collector.metrics

    def test_cache_metrics_present(self):
        """Integration: cache metrics are registered."""
        collector = MetricsCollector()
        assert "cache_hits_total" in collector.metrics
        assert "cache_misses_total" in collector.metrics
        assert "cache_invalidations_total" in collector.metrics
        assert "cache_memory_entries" in collector.metrics
        assert "cache_database_entries" in collector.metrics

    def test_api_metrics_present(self):
        """Integration: API metrics are registered."""
        collector = MetricsCollector()
        assert "api_request_duration_seconds" in collector.metrics
        assert "api_requests_total" in collector.metrics
        assert "api_errors_total" in collector.metrics

    def test_full_metrics_workflow(self):
        """Integration: complete metrics workflow."""
        collector = MetricsCollector()

        # Record cache hit
        collector.register_counter("cache_hits_total").increment()

        # Record batch job execution
        collector.register_counter("batch_jobs_executed_total").increment()
        collector.register_histogram("batch_job_duration_seconds").observe(0.5)

        # Record API request
        collector.register_counter("api_requests_total").increment()

        # Set cache hit rate
        collector.register_gauge("insights_cache_hit_rate").set(0.85)

        # Export to Prometheus format
        export = collector.export_prometheus()
        assert "cache_hits_total 1" in export
        assert "batch_jobs_executed_total 1" in export
        assert "batch_job_duration_seconds" in export
        assert "api_requests_total 1" in export
        assert "insights_cache_hit_rate 0.85" in export
