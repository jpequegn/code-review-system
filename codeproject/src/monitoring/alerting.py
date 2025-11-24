"""
Alerting System - Critical alerts, thresholds, and notifications

Implements production-grade alerting for Phase 5 operations:
- AlertRule: Threshold-based rule evaluation
- AlertManager: Central alert coordination
- Notification Channels: PagerDuty, Slack, Email, Logs
- Alert Deduplication: Prevent alert floods
- Alert History: Audit trail of all alerts

Alert Severity Levels:
- CRITICAL: Immediate action required (PagerDuty)
- WARNING: Should be addressed soon (Slack)
- INFO: Informational only (Logs)
"""

import json
import logging
from typing import Optional, List, Dict, Callable
from datetime import datetime, timezone, timedelta
from enum import Enum
from dataclasses import dataclass, field, asdict
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""
    CRITICAL = "critical"
    WARNING = "warning"
    INFO = "info"


class AlertStatus(str, Enum):
    """Alert status."""
    TRIGGERED = "triggered"
    ACKNOWLEDGED = "acknowledged"
    RESOLVED = "resolved"


@dataclass
class Alert:
    """Alert instance."""
    rule_name: str
    severity: AlertSeverity
    message: str
    value: float
    threshold: float
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    status: AlertStatus = AlertStatus.TRIGGERED
    acknowledged_at: Optional[datetime] = None
    resolved_at: Optional[datetime] = None
    notification_ids: List[str] = field(default_factory=list)

    def acknowledge(self) -> None:
        """Mark alert as acknowledged."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now(timezone.utc)

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)

    def to_dict(self) -> Dict:
        """Convert to dictionary."""
        return asdict(self)


class AlertRule(ABC):
    """Base class for alert rules."""

    def __init__(
        self,
        name: str,
        severity: AlertSeverity,
        description: str = "",
        notify_channels: Optional[List[str]] = None,
    ):
        """
        Initialize alert rule.

        Args:
            name: Rule name
            severity: Alert severity (CRITICAL, WARNING, INFO)
            description: Human-readable description
            notify_channels: List of notification channels (pagerduty, slack, email, logs)
        """
        self.name = name
        self.severity = severity
        self.description = description
        self.notify_channels = notify_channels or []
        self.last_triggered: Optional[datetime] = None
        self.trigger_count = 0

    @abstractmethod
    def evaluate(self) -> Optional[Alert]:
        """
        Evaluate rule condition.

        Returns:
            Alert if condition triggered, None otherwise
        """
        pass

    def should_notify(self, min_interval_minutes: int = 5) -> bool:
        """
        Check if enough time has passed to notify again.

        Args:
            min_interval_minutes: Minimum minutes between notifications

        Returns:
            True if should notify, False if too soon
        """
        if self.last_triggered is None:
            return True

        elapsed = datetime.now(timezone.utc) - self.last_triggered
        return elapsed >= timedelta(minutes=min_interval_minutes)

    def record_trigger(self) -> None:
        """Record rule trigger for deduplication."""
        self.last_triggered = datetime.now(timezone.utc)
        self.trigger_count += 1


class ThresholdRule(AlertRule):
    """Rule with a simple threshold comparison."""

    def __init__(
        self,
        name: str,
        severity: AlertSeverity,
        description: str,
        metric_name: str,
        threshold: float,
        operator: str,  # ">" or "<" or ">=" or "<="
        value_getter: Callable[[], float],
        notify_channels: Optional[List[str]] = None,
    ):
        """
        Initialize threshold rule.

        Args:
            name: Rule name
            severity: Alert severity
            description: Rule description
            metric_name: Name of metric being evaluated
            threshold: Threshold value
            operator: Comparison operator (>, <, >=, <=)
            value_getter: Callable that returns current metric value
            notify_channels: Notification channels
        """
        super().__init__(name, severity, description, notify_channels)
        self.metric_name = metric_name
        self.threshold = threshold
        self.operator = operator
        self.value_getter = value_getter

    def evaluate(self) -> Optional[Alert]:
        """Evaluate threshold condition."""
        try:
            current_value = self.value_getter()
        except Exception as e:
            logger.error(f"Error evaluating metric {self.metric_name}: {str(e)}")
            return None

        # Evaluate condition based on operator
        condition_met = False
        if self.operator == ">":
            condition_met = current_value > self.threshold
        elif self.operator == "<":
            condition_met = current_value < self.threshold
        elif self.operator == ">=":
            condition_met = current_value >= self.threshold
        elif self.operator == "<=":
            condition_met = current_value <= self.threshold

        if condition_met:
            message = f"{self.name}: {self.metric_name} = {current_value:.2f} (threshold: {self.threshold})"
            alert = Alert(
                rule_name=self.name,
                severity=self.severity,
                message=message,
                value=current_value,
                threshold=self.threshold,
            )
            return alert

        return None


class NotificationChannel(ABC):
    """Base class for notification channels."""

    @abstractmethod
    def send(self, alert: Alert) -> str:
        """
        Send alert notification.

        Args:
            alert: Alert to send

        Returns:
            Notification ID
        """
        pass


class LoggingChannel(NotificationChannel):
    """Log-based notification channel."""

    def send(self, alert: Alert) -> str:
        """Log alert."""
        log_func = {
            AlertSeverity.CRITICAL: logger.critical,
            AlertSeverity.WARNING: logger.warning,
            AlertSeverity.INFO: logger.info,
        }.get(alert.severity, logger.info)

        log_func(f"[{alert.severity.value.upper()}] {alert.message}")
        return f"log-{alert.timestamp.timestamp()}"


class PagerDutyChannel(NotificationChannel):
    """PagerDuty notification channel for critical alerts."""

    def __init__(self, api_key: str = ""):
        """
        Initialize PagerDuty channel.

        Args:
            api_key: PagerDuty API key (from environment or config)
        """
        self.api_key = api_key
        self.enabled = bool(api_key)

    def send(self, alert: Alert) -> str:
        """
        Send alert to PagerDuty.

        In production, this would use the PagerDuty Events API.
        For now, logs and returns a mock notification ID.

        Args:
            alert: Alert to send

        Returns:
            PagerDuty incident ID
        """
        if not self.enabled:
            logger.warning("PagerDuty not configured, skipping notification")
            return f"pagerduty-disabled-{alert.timestamp.timestamp()}"

        logger.info(
            f"Sending PagerDuty alert: {alert.rule_name} - {alert.message}"
        )

        # In production:
        # response = requests.post(
        #     "https://events.pagerduty.com/v2/enqueue",
        #     json={
        #         "routing_key": self.api_key,
        #         "event_action": "trigger",
        #         "dedup_key": f"{alert.rule_name}-{alert.timestamp.isoformat()}",
        #         "payload": {
        #             "summary": alert.message,
        #             "severity": alert.severity.value,
        #             "source": "Code Review System",
        #             "custom_details": alert.to_dict(),
        #         }
        #     }
        # )
        # return response.json().get("dedup_key")

        return f"pagerduty-{alert.timestamp.timestamp()}"


class SlackChannel(NotificationChannel):
    """Slack notification channel for warning alerts."""

    def __init__(self, webhook_url: str = ""):
        """
        Initialize Slack channel.

        Args:
            webhook_url: Slack webhook URL
        """
        self.webhook_url = webhook_url
        self.enabled = bool(webhook_url)

    def send(self, alert: Alert) -> str:
        """
        Send alert to Slack.

        In production, this would use the Slack webhook.
        For now, logs and returns a mock notification ID.

        Args:
            alert: Alert to send

        Returns:
            Slack message timestamp
        """
        if not self.enabled:
            logger.warning("Slack not configured, skipping notification")
            return f"slack-disabled-{alert.timestamp.timestamp()}"

        logger.info(f"Sending Slack alert: {alert.rule_name} - {alert.message}")

        # In production:
        # color_map = {
        #     AlertSeverity.CRITICAL: "danger",
        #     AlertSeverity.WARNING: "warning",
        #     AlertSeverity.INFO: "good",
        # }
        # requests.post(
        #     self.webhook_url,
        #     json={
        #         "attachments": [{
        #             "color": color_map.get(alert.severity, "good"),
        #             "title": f"Alert: {alert.rule_name}",
        #             "text": alert.message,
        #             "fields": [
        #                 {"title": "Severity", "value": alert.severity.value, "short": True},
        #                 {"title": "Current Value", "value": f"{alert.value:.2f}", "short": True},
        #                 {"title": "Threshold", "value": f"{alert.threshold:.2f}", "short": True},
        #                 {"title": "Time", "value": alert.timestamp.isoformat(), "short": False},
        #             ]
        #         }]
        #     }
        # )

        return f"slack-{alert.timestamp.timestamp()}"


class EmailChannel(NotificationChannel):
    """Email notification channel."""

    def __init__(self, smtp_config: Optional[Dict] = None):
        """
        Initialize email channel.

        Args:
            smtp_config: SMTP configuration dict with host, port, from_addr
        """
        self.smtp_config = smtp_config or {}
        self.enabled = bool(smtp_config)

    def send(self, alert: Alert) -> str:
        """
        Send alert via email.

        In production, this would use SMTP.
        For now, logs and returns a mock notification ID.

        Args:
            alert: Alert to send

        Returns:
            Email message ID
        """
        if not self.enabled:
            logger.warning("Email not configured, skipping notification")
            return f"email-disabled-{alert.timestamp.timestamp()}"

        logger.info(
            f"Sending email alert: {alert.rule_name} - {alert.message}"
        )

        # In production:
        # msg = EmailMessage()
        # msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.rule_name}"
        # msg["From"] = self.smtp_config.get("from_addr")
        # msg["To"] = self.smtp_config.get("to_addrs", [])
        # msg.set_content(f"{alert.message}\n\nValue: {alert.value}\nThreshold: {alert.threshold}")
        # with smtplib.SMTP(self.smtp_config.get("host")) as smtp:
        #     smtp.send_message(msg)

        return f"email-{alert.timestamp.timestamp()}"


class AlertManager:
    """
    Central alert management system.

    Manages alert rules, evaluation, notification, and history.
    """

    def __init__(self):
        """Initialize alert manager."""
        self.rules: List[AlertRule] = []
        self.channels: Dict[str, NotificationChannel] = {
            "logs": LoggingChannel(),
            "pagerduty": PagerDutyChannel(),
            "slack": SlackChannel(),
            "email": EmailChannel(),
        }
        self.alert_history: List[Alert] = []
        self.max_history = 1000

    def register_rule(self, rule: AlertRule) -> None:
        """
        Register an alert rule.

        Args:
            rule: AlertRule instance
        """
        self.rules.append(rule)
        logger.info(f"Registered alert rule: {rule.name}")

    def configure_channel(
        self, channel_name: str, channel: NotificationChannel
    ) -> None:
        """
        Configure notification channel.

        Args:
            channel_name: Channel name (pagerduty, slack, email, logs)
            channel: NotificationChannel instance
        """
        self.channels[channel_name] = channel
        logger.info(f"Configured notification channel: {channel_name}")

    def evaluate_all(self) -> List[Alert]:
        """
        Evaluate all alert rules.

        Returns:
            List of triggered alerts
        """
        triggered_alerts = []

        for rule in self.rules:
            try:
                alert = rule.evaluate()
                if alert:
                    triggered_alerts.append(alert)
            except Exception as e:
                logger.error(f"Error evaluating rule {rule.name}: {str(e)}")

        return triggered_alerts

    def check_critical_alerts(self) -> List[Alert]:
        """
        Check only critical alerts.

        Returns:
            List of critical alerts
        """
        return [a for a in self.evaluate_all() if a.severity == AlertSeverity.CRITICAL]

    def check_warning_alerts(self) -> List[Alert]:
        """
        Check only warning alerts.

        Returns:
            List of warning alerts
        """
        return [a for a in self.evaluate_all() if a.severity == AlertSeverity.WARNING]

    def process_alert(self, alert: Alert) -> bool:
        """
        Process and notify for an alert.

        Args:
            alert: Alert to process

        Returns:
            True if alert was notified, False if skipped (deduplication)
        """
        # Find rule to check deduplication
        rule = next((r for r in self.rules if r.name == alert.rule_name), None)
        if rule and not rule.should_notify():
            logger.info(f"Skipping alert {alert.rule_name} (too recent)")
            return False

        # Send notifications
        for channel_name in rule.notify_channels if rule else []:
            if channel_name in self.channels:
                try:
                    notification_id = self.channels[channel_name].send(alert)
                    alert.notification_ids.append(notification_id)
                except Exception as e:
                    logger.error(
                        f"Error sending {channel_name} notification: {str(e)}"
                    )

        # Record trigger for deduplication
        if rule:
            rule.record_trigger()

        # Add to history
        self.alert_history.append(alert)
        if len(self.alert_history) > self.max_history:
            self.alert_history = self.alert_history[-self.max_history :]

        return True

    def process_alerts(self, alerts: List[Alert]) -> int:
        """
        Process multiple alerts.

        Args:
            alerts: List of alerts to process

        Returns:
            Number of alerts notified
        """
        notified_count = 0
        for alert in alerts:
            if self.process_alert(alert):
                notified_count += 1
        return notified_count

    def get_alert_history(
        self, rule_name: Optional[str] = None, limit: int = 50
    ) -> List[Alert]:
        """
        Get alert history.

        Args:
            rule_name: Optional filter by rule name
            limit: Maximum number of records to return

        Returns:
            List of alerts
        """
        history = self.alert_history
        if rule_name:
            history = [a for a in history if a.rule_name == rule_name]
        return history[-limit:]

    def acknowledge_alert(self, alert_index: int) -> bool:
        """
        Acknowledge an alert.

        Args:
            alert_index: Index in alert history

        Returns:
            True if acknowledged, False if not found
        """
        if 0 <= alert_index < len(self.alert_history):
            self.alert_history[alert_index].acknowledge()
            logger.info(
                f"Acknowledged alert: {self.alert_history[alert_index].rule_name}"
            )
            return True
        return False

    def get_active_alerts(self) -> List[Alert]:
        """
        Get all active (non-resolved) alerts.

        Returns:
            List of active alerts
        """
        return [
            a
            for a in self.alert_history
            if a.status != AlertStatus.RESOLVED
        ]

    def get_metrics_summary(self) -> Dict:
        """
        Get summary of alert metrics.

        Returns:
            Summary dict
        """
        active = self.get_active_alerts()
        history = self.alert_history

        return {
            "total_alerts": len(history),
            "active_alerts": len(active),
            "critical_alerts": len(
                [a for a in active if a.severity == AlertSeverity.CRITICAL]
            ),
            "warning_alerts": len(
                [a for a in active if a.severity == AlertSeverity.WARNING]
            ),
            "acknowledged_alerts": len(
                [a for a in history if a.status == AlertStatus.ACKNOWLEDGED]
            ),
            "resolved_alerts": len(
                [a for a in history if a.status == AlertStatus.RESOLVED]
            ),
        }


# Global alert manager instance
_alert_manager_instance: Optional[AlertManager] = None


def get_alert_manager() -> AlertManager:
    """Get or create global alert manager instance."""
    global _alert_manager_instance
    if _alert_manager_instance is None:
        _alert_manager_instance = AlertManager()
    return _alert_manager_instance


def initialize_alert_manager() -> AlertManager:
    """Initialize and return alert manager instance."""
    global _alert_manager_instance
    _alert_manager_instance = AlertManager()
    return _alert_manager_instance
