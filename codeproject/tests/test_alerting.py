"""
Tests for alerting system.

Coverage:
- Alert rules evaluation
- AlertManager coordination
- Notification channels
- Alert history and deduplication
"""

import pytest
from datetime import datetime, timezone, timedelta
from src.monitoring import (
    Alert,
    AlertSeverity,
    AlertStatus,
    AlertManager,
    ThresholdRule,
    LoggingChannel,
    PagerDutyChannel,
    SlackChannel,
    EmailChannel,
)


class TestAlert:
    """Test alert model."""

    def test_alert_creation(self):
        """Alert: creates with all fields."""
        alert = Alert(
            rule_name="test_rule",
            severity=AlertSeverity.CRITICAL,
            message="Test message",
            value=10.5,
            threshold=5.0,
        )
        assert alert.rule_name == "test_rule"
        assert alert.severity == AlertSeverity.CRITICAL
        assert alert.status == AlertStatus.TRIGGERED
        assert alert.value == 10.5

    def test_alert_acknowledge(self):
        """Alert: can be acknowledged."""
        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.WARNING,
            message="Test",
            value=1.0,
            threshold=2.0,
        )
        assert alert.status == AlertStatus.TRIGGERED
        alert.acknowledge()
        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_at is not None

    def test_alert_resolve(self):
        """Alert: can be resolved."""
        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.INFO,
            message="Test",
            value=1.0,
            threshold=2.0,
        )
        alert.resolve()
        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None

    def test_alert_to_dict(self):
        """Alert: converts to dictionary."""
        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.CRITICAL,
            message="Test",
            value=5.0,
            threshold=3.0,
        )
        alert_dict = alert.to_dict()
        assert alert_dict["rule_name"] == "test"
        assert alert_dict["severity"] == AlertSeverity.CRITICAL
        assert alert_dict["value"] == 5.0


class TestThresholdRule:
    """Test threshold-based alert rules."""

    def test_rule_greater_than_threshold(self):
        """Rule: evaluates > operator correctly."""
        def get_value():
            return 10.0

        rule = ThresholdRule(
            name="high_value",
            severity=AlertSeverity.CRITICAL,
            description="Value too high",
            metric_name="test_metric",
            threshold=5.0,
            operator=">",
            value_getter=get_value,
            notify_channels=["logs"],
        )

        alert = rule.evaluate()
        assert alert is not None
        assert alert.value == 10.0
        assert alert.threshold == 5.0

    def test_rule_less_than_threshold(self):
        """Rule: evaluates < operator correctly."""
        def get_value():
            return 2.0

        rule = ThresholdRule(
            name="low_value",
            severity=AlertSeverity.WARNING,
            description="Value too low",
            metric_name="test_metric",
            threshold=5.0,
            operator="<",
            value_getter=get_value,
            notify_channels=["slack"],
        )

        alert = rule.evaluate()
        assert alert is not None
        assert alert.value == 2.0

    def test_rule_no_trigger(self):
        """Rule: returns None when condition not met."""
        def get_value():
            return 3.0

        rule = ThresholdRule(
            name="high_value",
            severity=AlertSeverity.CRITICAL,
            description="Value too high",
            metric_name="test_metric",
            threshold=5.0,
            operator=">",
            value_getter=get_value,
            notify_channels=["logs"],
        )

        alert = rule.evaluate()
        assert alert is None

    def test_rule_deduplication(self):
        """Rule: respects deduplication interval."""
        def get_value():
            return 10.0

        rule = ThresholdRule(
            name="high_value",
            severity=AlertSeverity.CRITICAL,
            description="Test",
            metric_name="test",
            threshold=5.0,
            operator=">",
            value_getter=get_value,
            notify_channels=["logs"],
        )

        # First trigger should notify
        assert rule.should_notify() is True
        rule.record_trigger()

        # Immediate second trigger should not notify (default 5 minute interval)
        assert rule.should_notify() is False

        # But custom check with 0 minute interval should notify
        assert rule.should_notify(min_interval_minutes=0) is True


class TestNotificationChannels:
    """Test notification channels."""

    def test_logging_channel_sends(self):
        """LoggingChannel: sends alert."""
        channel = LoggingChannel()
        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.WARNING,
            message="Test alert",
            value=5.0,
            threshold=3.0,
        )
        notification_id = channel.send(alert)
        assert notification_id.startswith("log-")

    def test_pagerduty_channel_disabled(self):
        """PagerDutyChannel: handles disabled state."""
        channel = PagerDutyChannel()  # No API key
        assert channel.enabled is False

        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.CRITICAL,
            message="Critical alert",
            value=5.0,
            threshold=3.0,
        )
        notification_id = channel.send(alert)
        assert "disabled" in notification_id

    def test_pagerduty_channel_enabled(self):
        """PagerDutyChannel: handles enabled state."""
        channel = PagerDutyChannel(api_key="test-key-12345")
        assert channel.enabled is True

        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.CRITICAL,
            message="Critical alert",
            value=5.0,
            threshold=3.0,
        )
        notification_id = channel.send(alert)
        assert notification_id.startswith("pagerduty-")

    def test_slack_channel_disabled(self):
        """SlackChannel: handles disabled state."""
        channel = SlackChannel()  # No webhook
        assert channel.enabled is False

        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.WARNING,
            message="Warning",
            value=5.0,
            threshold=3.0,
        )
        notification_id = channel.send(alert)
        assert "disabled" in notification_id

    def test_slack_channel_enabled(self):
        """SlackChannel: handles enabled state."""
        channel = SlackChannel(webhook_url="https://hooks.slack.com/...")
        assert channel.enabled is True

        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.WARNING,
            message="Warning",
            value=5.0,
            threshold=3.0,
        )
        notification_id = channel.send(alert)
        assert notification_id.startswith("slack-")


class TestAlertManager:
    """Test alert manager."""

    def test_manager_initialization(self):
        """Manager: initializes with channels."""
        manager = AlertManager()
        assert len(manager.rules) == 0
        assert len(manager.channels) >= 4  # logs, pagerduty, slack, email

    def test_register_rule(self):
        """Manager: registers rules."""
        manager = AlertManager()

        def get_value():
            return 5.0

        rule = ThresholdRule(
            name="test_rule",
            severity=AlertSeverity.WARNING,
            description="Test",
            metric_name="test",
            threshold=3.0,
            operator=">",
            value_getter=get_value,
            notify_channels=["logs"],
        )

        manager.register_rule(rule)
        assert len(manager.rules) == 1
        assert manager.rules[0].name == "test_rule"

    def test_evaluate_all_rules(self):
        """Manager: evaluates all rules."""
        manager = AlertManager()

        def get_high():
            return 10.0

        def get_low():
            return 1.0

        rule1 = ThresholdRule(
            name="high_rule",
            severity=AlertSeverity.CRITICAL,
            description="High",
            metric_name="test1",
            threshold=5.0,
            operator=">",
            value_getter=get_high,
            notify_channels=["logs"],
        )

        rule2 = ThresholdRule(
            name="low_rule",
            severity=AlertSeverity.WARNING,
            description="Low",
            metric_name="test2",
            threshold=5.0,
            operator=">",
            value_getter=get_low,
            notify_channels=["logs"],
        )

        manager.register_rule(rule1)
        manager.register_rule(rule2)

        alerts = manager.evaluate_all()
        assert len(alerts) == 1  # Only high_rule triggers
        assert alerts[0].rule_name == "high_rule"

    def test_check_critical_alerts(self):
        """Manager: filters critical alerts."""
        manager = AlertManager()

        def get_value():
            return 10.0

        critical_rule = ThresholdRule(
            name="critical",
            severity=AlertSeverity.CRITICAL,
            description="Critical",
            metric_name="test",
            threshold=5.0,
            operator=">",
            value_getter=get_value,
            notify_channels=["logs"],
        )

        warning_rule = ThresholdRule(
            name="warning",
            severity=AlertSeverity.WARNING,
            description="Warning",
            metric_name="test",
            threshold=5.0,
            operator=">",
            value_getter=get_value,
            notify_channels=["logs"],
        )

        manager.register_rule(critical_rule)
        manager.register_rule(warning_rule)

        critical_alerts = manager.check_critical_alerts()
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL

    def test_process_alert(self):
        """Manager: processes alert and records history."""
        manager = AlertManager()
        alert = Alert(
            rule_name="test",
            severity=AlertSeverity.WARNING,
            message="Test",
            value=5.0,
            threshold=3.0,
        )

        # Add rule for notification
        def get_value():
            return 5.0

        rule = ThresholdRule(
            name="test",
            severity=AlertSeverity.WARNING,
            description="Test",
            metric_name="test",
            threshold=3.0,
            operator=">",
            value_getter=get_value,
            notify_channels=["logs"],
        )
        manager.register_rule(rule)

        result = manager.process_alert(alert)
        assert result is True
        assert len(manager.alert_history) == 1

    def test_alert_deduplication(self):
        """Manager: deduplicates recent alerts."""
        manager = AlertManager()

        def get_value():
            return 10.0

        rule = ThresholdRule(
            name="test",
            severity=AlertSeverity.CRITICAL,
            description="Test",
            metric_name="test",
            threshold=5.0,
            operator=">",
            value_getter=get_value,
            notify_channels=["logs"],
        )
        manager.register_rule(rule)

        alert1 = Alert(
            rule_name="test",
            severity=AlertSeverity.CRITICAL,
            message="Alert 1",
            value=10.0,
            threshold=5.0,
        )

        alert2 = Alert(
            rule_name="test",
            severity=AlertSeverity.CRITICAL,
            message="Alert 2",
            value=10.0,
            threshold=5.0,
        )

        # First alert should process
        assert manager.process_alert(alert1) is True

        # Second alert should be skipped (too recent)
        assert manager.process_alert(alert2) is False

    def test_get_alert_history(self):
        """Manager: retrieves alert history."""
        manager = AlertManager()

        for i in range(3):
            alert = Alert(
                rule_name=f"rule_{i}",
                severity=AlertSeverity.WARNING,
                message=f"Alert {i}",
                value=float(i),
                threshold=5.0,
            )
            manager.alert_history.append(alert)

        history = manager.get_alert_history()
        assert len(history) == 3

        # Filter by rule
        rule_0_history = manager.get_alert_history(rule_name="rule_0")
        assert len(rule_0_history) == 1
        assert rule_0_history[0].rule_name == "rule_0"

    def test_get_active_alerts(self):
        """Manager: returns only active alerts."""
        manager = AlertManager()

        active = Alert(
            rule_name="test",
            severity=AlertSeverity.CRITICAL,
            message="Active",
            value=5.0,
            threshold=3.0,
        )

        resolved = Alert(
            rule_name="test",
            severity=AlertSeverity.WARNING,
            message="Resolved",
            value=5.0,
            threshold=3.0,
        )
        resolved.resolve()

        manager.alert_history.append(active)
        manager.alert_history.append(resolved)

        active_alerts = manager.get_active_alerts()
        assert len(active_alerts) == 1
        assert active_alerts[0].status == AlertStatus.TRIGGERED

    def test_metrics_summary(self):
        """Manager: provides metrics summary."""
        manager = AlertManager()

        alert1 = Alert(
            rule_name="test1",
            severity=AlertSeverity.CRITICAL,
            message="Critical",
            value=5.0,
            threshold=3.0,
        )

        alert2 = Alert(
            rule_name="test2",
            severity=AlertSeverity.WARNING,
            message="Warning",
            value=5.0,
            threshold=3.0,
        )

        manager.alert_history.append(alert1)
        manager.alert_history.append(alert2)
        alert1.acknowledge()

        summary = manager.get_metrics_summary()
        assert summary["total_alerts"] == 2
        assert summary["active_alerts"] == 2
        assert summary["critical_alerts"] == 1
        assert summary["warning_alerts"] == 1
        assert summary["acknowledged_alerts"] == 1
