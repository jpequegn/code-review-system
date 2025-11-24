"""
Alert Rules Configuration - Pre-defined alert rules for Phase 5 operations

Defines critical, warning, and info alert rules with thresholds and
notification channels.
"""

from src.monitoring.alerting import (
    AlertManager,
    AlertRule,
    AlertSeverity,
    ThresholdRule,
)
from src.monitoring import get_metrics


def setup_critical_alert_rules(alert_manager: AlertManager) -> None:
    """
    Set up critical alert rules.

    Critical alerts trigger PagerDuty for immediate action.

    Args:
        alert_manager: AlertManager instance
    """
    metrics = get_metrics()

    # Learning job failure rate > 5%
    def get_job_failure_rate():
        """Calculate job failure rate."""
        jobs_failed = metrics.get_metric("batch_jobs_failed_total")
        jobs_total = metrics.get_metric("batch_jobs_executed_total")

        if not jobs_failed or not jobs_total or jobs_total.value == 0:
            return 0.0

        return (jobs_failed.value / jobs_total.value) * 100.0

    failure_rate_rule = ThresholdRule(
        name="high_job_failure_rate",
        severity=AlertSeverity.CRITICAL,
        description="Learning job failure rate exceeds 5%",
        metric_name="job_failure_rate",
        threshold=5.0,
        operator=">",
        value_getter=get_job_failure_rate,
        notify_channels=["pagerduty", "slack", "email"],
    )
    alert_manager.register_rule(failure_rate_rule)

    # Insights latency > 5 seconds (p95)
    def get_insights_latency():
        """Get insights generation latency p95."""
        metric = metrics.get_metric("insights_generation_duration_seconds")
        if not metric:
            return 0.0
        return metric.get_percentile(95)

    insights_latency_rule = ThresholdRule(
        name="high_insights_latency",
        severity=AlertSeverity.CRITICAL,
        description="Insights generation latency (p95) exceeds 5 seconds",
        metric_name="insights_generation_duration_seconds_p95",
        threshold=5.0,
        operator=">",
        value_getter=get_insights_latency,
        notify_channels=["pagerduty", "slack", "email"],
    )
    alert_manager.register_rule(insights_latency_rule)

    # Cache hit rate < 20% (critical threshold)
    def get_cache_hit_rate():
        """Get cache hit rate."""
        metric = metrics.get_metric("insights_cache_hit_rate")
        if not metric:
            return 0.0
        return metric.value * 100.0  # Convert to percentage

    cache_hit_rule = ThresholdRule(
        name="low_cache_hit_rate",
        severity=AlertSeverity.CRITICAL,
        description="Cache hit rate below 20%",
        metric_name="insights_cache_hit_rate",
        threshold=20.0,
        operator="<",
        value_getter=get_cache_hit_rate,
        notify_channels=["pagerduty", "slack"],
    )
    alert_manager.register_rule(cache_hit_rule)


def setup_warning_alert_rules(alert_manager: AlertManager) -> None:
    """
    Set up warning alert rules.

    Warning alerts trigger Slack for soon attention.

    Args:
        alert_manager: AlertManager instance
    """
    metrics = get_metrics()

    # Learning latency > 2 seconds (p95)
    def get_learning_latency():
        """Get learning metrics update latency p95."""
        metric = metrics.get_metric("learning_metrics_update_duration_seconds")
        if not metric:
            return 0.0
        return metric.get_percentile(95)

    learning_latency_rule = ThresholdRule(
        name="high_learning_latency",
        severity=AlertSeverity.WARNING,
        description="Learning metrics update latency (p95) exceeds 2 seconds",
        metric_name="learning_metrics_update_duration_seconds_p95",
        threshold=2.0,
        operator=">",
        value_getter=get_learning_latency,
        notify_channels=["slack", "email"],
    )
    alert_manager.register_rule(learning_latency_rule)

    # Cache hit rate < 50% (warning threshold)
    def get_cache_hit_rate():
        """Get cache hit rate."""
        metric = metrics.get_metric("insights_cache_hit_rate")
        if not metric:
            return 0.0
        return metric.value * 100.0

    cache_hit_warning = ThresholdRule(
        name="degraded_cache_performance",
        severity=AlertSeverity.WARNING,
        description="Cache hit rate below 50%",
        metric_name="insights_cache_hit_rate",
        threshold=50.0,
        operator="<",
        value_getter=get_cache_hit_rate,
        notify_channels=["slack"],
    )
    alert_manager.register_rule(cache_hit_warning)

    # API error rate > 1%
    def get_api_error_rate():
        """Calculate API error rate."""
        api_errors = metrics.get_metric("api_errors_total")
        api_requests = metrics.get_metric("api_requests_total")

        if not api_errors or not api_requests or api_requests.value == 0:
            return 0.0

        return (api_errors.value / api_requests.value) * 100.0

    api_error_rule = ThresholdRule(
        name="high_api_error_rate",
        severity=AlertSeverity.WARNING,
        description="API error rate exceeds 1%",
        metric_name="api_error_rate",
        threshold=1.0,
        operator=">",
        value_getter=get_api_error_rate,
        notify_channels=["slack", "email"],
    )
    alert_manager.register_rule(api_error_rule)

    # Ranking latency > 100ms (p95)
    def get_ranking_latency():
        """Get ranking duration p95."""
        metric = metrics.get_metric("ranking_duration_seconds")
        if not metric:
            return 0.0
        return metric.get_percentile(95) * 1000  # Convert to ms

    ranking_latency_rule = ThresholdRule(
        name="high_ranking_latency",
        severity=AlertSeverity.WARNING,
        description="Ranking latency (p95) exceeds 100ms",
        metric_name="ranking_duration_seconds_p95",
        threshold=100.0,
        operator=">",
        value_getter=get_ranking_latency,
        notify_channels=["slack"],
    )
    alert_manager.register_rule(ranking_latency_rule)


def setup_info_alert_rules(alert_manager: AlertManager) -> None:
    """
    Set up info alert rules.

    Info alerts go to logs only.

    Args:
        alert_manager: AlertManager instance
    """
    metrics = get_metrics()

    # High batch job success rate (informational)
    def get_job_success_rate():
        """Calculate job success rate."""
        jobs_succeeded = metrics.get_metric("batch_jobs_succeeded_total")
        jobs_total = metrics.get_metric("batch_jobs_executed_total")

        if not jobs_succeeded or not jobs_total or jobs_total.value == 0:
            return 0.0

        return (jobs_succeeded.value / jobs_total.value) * 100.0

    # This rule has inverted logic - we want success > 95%
    # For demonstration, we'll create a simple info rule
    success_rate_rule = ThresholdRule(
        name="job_success_rate_info",
        severity=AlertSeverity.INFO,
        description="Batch job success rate (informational)",
        metric_name="job_success_rate",
        threshold=95.0,
        operator=">",
        value_getter=get_job_success_rate,
        notify_channels=["logs"],
    )
    alert_manager.register_rule(success_rate_rule)

    # Cache performance info
    def get_cache_hit_rate():
        """Get cache hit rate."""
        metric = metrics.get_metric("insights_cache_hit_rate")
        if not metric:
            return 0.0
        return metric.value * 100.0

    cache_performance_rule = ThresholdRule(
        name="cache_performance_info",
        severity=AlertSeverity.INFO,
        description="Cache performance statistics (informational)",
        metric_name="insights_cache_hit_rate",
        threshold=80.0,  # Info when > 80%
        operator=">",
        value_getter=get_cache_hit_rate,
        notify_channels=["logs"],
    )
    alert_manager.register_rule(cache_performance_rule)


def setup_all_alert_rules(alert_manager: AlertManager) -> None:
    """
    Set up all alert rules.

    Args:
        alert_manager: AlertManager instance
    """
    setup_critical_alert_rules(alert_manager)
    setup_warning_alert_rules(alert_manager)
    setup_info_alert_rules(alert_manager)
