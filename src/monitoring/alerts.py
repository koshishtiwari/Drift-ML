"""
Alerting module for Drift-ML platform.
Provides functionality for setting up alerts based on metrics and drift detection.
"""
import os
import time
import json
import logging
from typing import Dict, List, Optional, Union, Any, Callable
from datetime import datetime
import threading
import smtplib
import requests
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from loguru import logger
from pydantic import BaseModel, Field

class AlertRule(BaseModel):
    """Pydantic model for alert rule definition."""
    name: str
    description: str
    metric_name: str
    threshold: float
    operator: str = ">"  # >, <, >=, <=, ==, !=
    severity: str = "warning"  # info, warning, error, critical
    enabled: bool = True
    cool_down_seconds: int = 300  # Minimum time between alerts

class AlertChannel(BaseModel):
    """Pydantic model for alert channel definition."""
    name: str
    type: str  # email, slack, webhook, pagerduty
    config: Dict[str, Any]
    enabled: bool = True

class AlertEvent(BaseModel):
    """Pydantic model for alert event."""
    rule_name: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metric_name: str
    metric_value: float
    threshold: float
    operator: str
    severity: str
    message: str
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    additional_info: Optional[Dict[str, Any]] = None

class AlertManager:
    """
    Manages alerts for the Drift-ML platform.
    """
    
    def __init__(self):
        """Initialize the alert manager."""
        self.rules: Dict[str, AlertRule] = {}
        self.channels: Dict[str, AlertChannel] = {}
        self.last_triggered: Dict[str, datetime] = {}
        
        # Set up a thread for periodic evaluation
        self.stop_event = threading.Event()
        self.evaluation_thread = None
    
    def add_rule(self, rule: AlertRule) -> None:
        """
        Add or update an alert rule.
        
        Args:
            rule: Alert rule to add
        """
        self.rules[rule.name] = rule
        logger.info(f"Added alert rule: {rule.name}")
    
    def remove_rule(self, rule_name: str) -> bool:
        """
        Remove an alert rule.
        
        Args:
            rule_name: Name of the rule to remove
            
        Returns:
            True if the rule was removed, False otherwise
        """
        if rule_name in self.rules:
            del self.rules[rule_name]
            logger.info(f"Removed alert rule: {rule_name}")
            return True
        
        return False
    
    def add_channel(self, channel: AlertChannel) -> None:
        """
        Add or update an alert channel.
        
        Args:
            channel: Alert channel to add
        """
        self.channels[channel.name] = channel
        logger.info(f"Added alert channel: {channel.name}")
    
    def remove_channel(self, channel_name: str) -> bool:
        """
        Remove an alert channel.
        
        Args:
            channel_name: Name of the channel to remove
            
        Returns:
            True if the channel was removed, False otherwise
        """
        if channel_name in self.channels:
            del self.channels[channel_name]
            logger.info(f"Removed alert channel: {channel_name}")
            return True
        
        return False
    
    def evaluate_rule(
        self,
        rule: AlertRule,
        metric_value: float,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> Optional[AlertEvent]:
        """
        Evaluate an alert rule against a metric value.
        
        Args:
            rule: Alert rule to evaluate
            metric_value: Current value of the metric
            additional_info: Additional information to include in the alert
            
        Returns:
            AlertEvent if the rule triggered, None otherwise
        """
        if not rule.enabled:
            return None
        
        # Check if the rule is in cool down period
        now = datetime.now()
        if rule.name in self.last_triggered:
            time_since_last = (now - self.last_triggered[rule.name]).total_seconds()
            if time_since_last < rule.cool_down_seconds:
                return None
        
        # Evaluate the rule
        triggered = False
        
        if rule.operator == ">":
            triggered = metric_value > rule.threshold
        elif rule.operator == "<":
            triggered = metric_value < rule.threshold
        elif rule.operator == ">=":
            triggered = metric_value >= rule.threshold
        elif rule.operator == "<=":
            triggered = metric_value <= rule.threshold
        elif rule.operator == "==":
            triggered = metric_value == rule.threshold
        elif rule.operator == "!=":
            triggered = metric_value != rule.threshold
        else:
            logger.warning(f"Unknown operator: {rule.operator}")
            return None
        
        if triggered:
            # Create alert event
            model_info = {}
            if additional_info:
                model_info = {
                    "model_name": additional_info.get("model_name"),
                    "model_version": additional_info.get("model_version")
                }
            
            message = f"Alert rule '{rule.name}' triggered: {rule.metric_name} {rule.operator} {rule.threshold} (actual: {metric_value})"
            
            event = AlertEvent(
                rule_name=rule.name,
                metric_name=rule.metric_name,
                metric_value=metric_value,
                threshold=rule.threshold,
                operator=rule.operator,
                severity=rule.severity,
                message=message,
                additional_info=additional_info,
                **model_info
            )
            
            # Update last triggered time
            self.last_triggered[rule.name] = now
            
            return event
        
        return None
    
    def check_metric(
        self,
        metric_name: str,
        metric_value: float,
        additional_info: Optional[Dict[str, Any]] = None
    ) -> List[AlertEvent]:
        """
        Check a metric against all applicable rules.
        
        Args:
            metric_name: Name of the metric
            metric_value: Current value of the metric
            additional_info: Additional information to include in the alert
            
        Returns:
            List of triggered alert events
        """
        events = []
        
        for rule in self.rules.values():
            if rule.metric_name == metric_name:
                event = self.evaluate_rule(rule, metric_value, additional_info)
                if event:
                    events.append(event)
                    self.send_alert(event)
        
        return events
    
    def send_alert(self, event: AlertEvent) -> None:
        """
        Send an alert to all enabled channels.
        
        Args:
            event: Alert event to send
        """
        for channel in self.channels.values():
            if channel.enabled:
                try:
                    self._send_to_channel(channel, event)
                except Exception as e:
                    logger.error(f"Failed to send alert to {channel.name}: {e}")
    
    def _send_to_channel(self, channel: AlertChannel, event: AlertEvent) -> None:
        """
        Send an alert to a specific channel.
        
        Args:
            channel: Channel to send to
            event: Alert event to send
        """
        if channel.type == "email":
            self._send_email(channel, event)
        elif channel.type == "slack":
            self._send_slack(channel, event)
        elif channel.type == "webhook":
            self._send_webhook(channel, event)
        elif channel.type == "pagerduty":
            self._send_pagerduty(channel, event)
        else:
            logger.warning(f"Unknown channel type: {channel.type}")
    
    def _send_email(self, channel: AlertChannel, event: AlertEvent) -> None:
        """
        Send an alert via email.
        
        Args:
            channel: Email channel
            event: Alert event to send
        """
        config = channel.config
        
        if not config.get("smtp_server") or not config.get("from_email") or not config.get("to_emails"):
            logger.error("Email channel configuration is incomplete")
            return
        
        # Create message
        msg = MIMEMultipart()
        msg["From"] = config["from_email"]
        msg["To"] = ", ".join(config["to_emails"])
        msg["Subject"] = f"[{event.severity.upper()}] Drift-ML Alert: {event.rule_name}"
        
        # Create the message text
        text = f"""
        Alert: {event.message}
        
        Severity: {event.severity}
        Timestamp: {event.timestamp}
        
        Metric: {event.metric_name}
        Value: {event.metric_value}
        Threshold: {event.threshold} ({event.operator})
        
        """
        
        if event.model_name:
            text += f"Model: {event.model_name}\n"
        
        if event.model_version:
            text += f"Version: {event.model_version}\n"
        
        if event.additional_info:
            text += "\nAdditional Information:\n"
            for key, value in event.additional_info.items():
                if isinstance(value, dict):
                    text += f"{key}:\n"
                    for k, v in value.items():
                        text += f"  {k}: {v}\n"
                else:
                    text += f"{key}: {value}\n"
        
        msg.attach(MIMEText(text, "plain"))
        
        # Connect to SMTP server and send
        try:
            with smtplib.SMTP(config["smtp_server"], config.get("smtp_port", 587)) as server:
                if config.get("use_tls", True):
                    server.starttls()
                
                if config.get("username") and config.get("password"):
                    server.login(config["username"], config["password"])
                
                server.send_message(msg)
                
                logger.info(f"Sent email alert to {', '.join(config['to_emails'])}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def _send_slack(self, channel: AlertChannel, event: AlertEvent) -> None:
        """
        Send an alert to Slack.
        
        Args:
            channel: Slack channel
            event: Alert event to send
        """
        config = channel.config
        
        if not config.get("webhook_url"):
            logger.error("Slack channel configuration is incomplete")
            return
        
        # Create severity color
        color = "#000000"
        if event.severity == "info":
            color = "#2196F3"  # Blue
        elif event.severity == "warning":
            color = "#FFC107"  # Amber
        elif event.severity == "error":
            color = "#F44336"  # Red
        elif event.severity == "critical":
            color = "#9C27B0"  # Purple
        
        # Create message payload
        payload = {
            "attachments": [
                {
                    "fallback": event.message,
                    "color": color,
                    "title": f"Drift-ML Alert: {event.rule_name}",
                    "text": event.message,
                    "fields": [
                        {
                            "title": "Severity",
                            "value": event.severity.upper(),
                            "short": True
                        },
                        {
                            "title": "Metric",
                            "value": event.metric_name,
                            "short": True
                        },
                        {
                            "title": "Value",
                            "value": str(event.metric_value),
                            "short": True
                        },
                        {
                            "title": "Threshold",
                            "value": f"{event.operator} {event.threshold}",
                            "short": True
                        }
                    ],
                    "footer": f"Drift-ML • {event.timestamp.strftime('%Y-%m-%d %H:%M:%S')}",
                    "mrkdwn_in": ["text", "fields"]
                }
            ]
        }
        
        # Add model info if available
        if event.model_name:
            payload["attachments"][0]["fields"].append({
                "title": "Model",
                "value": event.model_name,
                "short": True
            })
        
        if event.model_version:
            payload["attachments"][0]["fields"].append({
                "title": "Version",
                "value": event.model_version,
                "short": True
            })
        
        # Send to Slack
        try:
            response = requests.post(
                config["webhook_url"],
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"Failed to send Slack alert: {response.text}")
            else:
                logger.info(f"Sent Slack alert for {event.rule_name}")
        except Exception as e:
            logger.error(f"Failed to send Slack alert: {e}")
    
    def _send_webhook(self, channel: AlertChannel, event: AlertEvent) -> None:
        """
        Send an alert to a webhook.
        
        Args:
            channel: Webhook channel
            event: Alert event to send
        """
        config = channel.config
        
        if not config.get("url"):
            logger.error("Webhook channel configuration is incomplete")
            return
        
        # Create payload
        payload = {
            "event": event.dict(),
            "timestamp": datetime.now().isoformat()
        }
        
        # Send to webhook
        try:
            response = requests.post(
                config["url"],
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code < 200 or response.status_code >= 300:
                logger.error(f"Failed to send webhook alert: {response.text}")
            else:
                logger.info(f"Sent webhook alert for {event.rule_name}")
        except Exception as e:
            logger.error(f"Failed to send webhook alert: {e}")
    
    def _send_pagerduty(self, channel: AlertChannel, event: AlertEvent) -> None:
        """
        Send an alert to PagerDuty.
        
        Args:
            channel: PagerDuty channel
            event: Alert event to send
        """
        config = channel.config
        
        if not config.get("routing_key") and not config.get("service_key"):
            logger.error("PagerDuty channel configuration is incomplete")
            return
        
        # Determine the key to use
        integration_key = config.get("routing_key") or config.get("service_key")
        
        # Map severity
        severity = "info"
        if event.severity == "warning":
            severity = "warning"
        elif event.severity == "error":
            severity = "error"
        elif event.severity == "critical":
            severity = "critical"
        
        # Create payload
        payload = {
            "routing_key": integration_key,
            "event_action": "trigger",
            "dedup_key": f"drift-ml-{event.rule_name}-{event.metric_name}",
            "payload": {
                "summary": event.message,
                "source": "Drift-ML",
                "severity": severity,
                "custom_details": {
                    "metric_name": event.metric_name,
                    "metric_value": event.metric_value,
                    "threshold": event.threshold,
                    "operator": event.operator,
                    "rule_name": event.rule_name
                }
            }
        }
        
        # Add model info if available
        if event.model_name:
            payload["payload"]["custom_details"]["model_name"] = event.model_name
        
        if event.model_version:
            payload["payload"]["custom_details"]["model_version"] = event.model_version
        
        # Send to PagerDuty
        try:
            response = requests.post(
                "https://events.pagerduty.com/v2/enqueue",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 202:
                logger.error(f"Failed to send PagerDuty alert: {response.text}")
            else:
                logger.info(f"Sent PagerDuty alert for {event.rule_name}")
        except Exception as e:
            logger.error(f"Failed to send PagerDuty alert: {e}")
    
    def start_periodic_evaluation(
        self,
        metrics_provider: Callable[[], Dict[str, float]],
        interval_seconds: int = 60
    ) -> None:
        """
        Start periodic evaluation of metrics against rules.
        
        Args:
            metrics_provider: Function that returns current metrics
            interval_seconds: Interval between evaluations
        """
        def evaluation_loop():
            while not self.stop_event.is_set():
                try:
                    metrics = metrics_provider()
                    for metric_name, metric_value in metrics.items():
                        self.check_metric(metric_name, metric_value)
                except Exception as e:
                    logger.error(f"Error in periodic evaluation: {e}")
                
                # Wait for the next interval
                self.stop_event.wait(interval_seconds)
        
        self.evaluation_thread = threading.Thread(target=evaluation_loop, daemon=True)
        self.evaluation_thread.start()
        logger.info(f"Started periodic alert evaluation (interval: {interval_seconds}s)")
    
    def stop_periodic_evaluation(self) -> None:
        """Stop periodic evaluation of metrics."""
        if self.evaluation_thread:
            self.stop_event.set()
            self.evaluation_thread.join(timeout=10)
            self.evaluation_thread = None
            self.stop_event.clear()
            logger.info("Stopped periodic alert evaluation")

class DriftAlertManager:
    """
    Specialized alert manager for model drift.
    """
    
    def __init__(self, alert_manager: AlertManager):
        """
        Initialize the drift alert manager.
        
        Args:
            alert_manager: Base alert manager to use
        """
        self.alert_manager = alert_manager
    
    def setup_default_drift_alerts(
        self,
        model_name: str,
        model_version: str,
        psi_threshold: float = 0.2,
        drift_proportion_threshold: float = 0.3,
        prediction_drift_threshold: float = 0.2
    ) -> None:
        """
        Set up default drift alerts for a model.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            psi_threshold: Threshold for PSI drift alerts
            drift_proportion_threshold: Threshold for drift proportion
            prediction_drift_threshold: Threshold for prediction drift
        """
        # Feature drift alert
        self.alert_manager.add_rule(AlertRule(
            name=f"{model_name}_{model_version}_feature_drift",
            description=f"Feature drift detected for {model_name} v{model_version}",
            metric_name=f"{model_name}_{model_version}_drift_proportion",
            threshold=drift_proportion_threshold,
            operator=">",
            severity="warning",
            cool_down_seconds=3600  # 1 hour
        ))
        
        # Prediction drift alert
        self.alert_manager.add_rule(AlertRule(
            name=f"{model_name}_{model_version}_prediction_drift",
            description=f"Prediction drift detected for {model_name} v{model_version}",
            metric_name=f"{model_name}_{model_version}_prediction_psi",
            threshold=prediction_drift_threshold,
            operator=">",
            severity="warning",
            cool_down_seconds=3600  # 1 hour
        ))
        
        # Critical drift alert (high drift proportion)
        self.alert_manager.add_rule(AlertRule(
            name=f"{model_name}_{model_version}_critical_drift",
            description=f"Critical drift detected for {model_name} v{model_version}",
            metric_name=f"{model_name}_{model_version}_drift_proportion",
            threshold=0.5,  # 50% of features drifting
            operator=">",
            severity="critical",
            cool_down_seconds=86400  # 1 day
        ))
        
        logger.info(f"Set up default drift alerts for {model_name} v{model_version}")
    
    def process_drift_metrics(
        self,
        model_name: str,
        model_version: str,
        drift_metrics: Dict[str, Any]
    ) -> List[AlertEvent]:
        """
        Process drift metrics and trigger alerts if necessary.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            drift_metrics: Drift metrics from ModelDriftMonitor
            
        Returns:
            List of triggered alert events
        """
        events = []
        
        # Add model info to additional_info
        additional_info = {
            "model_name": model_name,
            "model_version": model_version,
            "drift_metrics": drift_metrics
        }
        
        # Process feature drift metrics
        if "feature_drift" in drift_metrics:
            feature_drift = drift_metrics["feature_drift"]
            
            # Check drift proportion
            drift_proportion = feature_drift.get("drift_proportion", 0)
            events.extend(self.alert_manager.check_metric(
                f"{model_name}_{model_version}_drift_proportion",
                drift_proportion,
                additional_info
            ))
            
            # Check individual feature PSI metrics
            for key, value in feature_drift.items():
                if key.endswith("_psi"):
                    feature_name = key.replace("_psi", "")
                    events.extend(self.alert_manager.check_metric(
                        f"{model_name}_{model_version}_{feature_name}_psi",
                        value,
                        additional_info
                    ))
        
        # Process prediction drift metrics
        if "prediction_drift" in drift_metrics:
            prediction_drift = drift_metrics["prediction_drift"]
            
            # Check prediction PSI
            if "prediction_psi" in prediction_drift:
                events.extend(self.alert_manager.check_metric(
                    f"{model_name}_{model_version}_prediction_psi",
                    prediction_drift["prediction_psi"],
                    additional_info
                ))
        
        # Check overall drift
        if "overall_drift_detected" in drift_metrics and drift_metrics["overall_drift_detected"]:
            events.extend(self.alert_manager.check_metric(
                f"{model_name}_{model_version}_overall_drift",
                1.0,  # Boolean true = 1.0 for threshold comparison
                additional_info
            ))
        
        return events

# Example usage
if __name__ == "__main__":
    # Create alert manager
    alert_manager = AlertManager()
    
    # Add email channel
    alert_manager.add_channel(AlertChannel(
        name="email_alerts",
        type="email",
        config={
            "smtp_server": "smtp.example.com",
            "smtp_port": 587,
            "use_tls": True,
            "username": "alerts@example.com",
            "password": "password123",
            "from_email": "alerts@example.com",
            "to_emails": ["team@example.com"]
        }
    ))
    
    # Add Slack channel
    alert_manager.add_channel(AlertChannel(
        name="slack_alerts",
        type="slack",
        config={
            "webhook_url": "https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX"
        }
    ))
    
    # Create drift alert manager
    drift_alert_manager = DriftAlertManager(alert_manager)
    
    # Set up default drift alerts
    drift_alert_manager.setup_default_drift_alerts(
        model_name="example_model",
        model_version="1"
    )
    
    # Simulate drift metrics
    drift_metrics = {
        "feature_drift": {
            "feature1_psi": 0.15,
            "feature2_psi": 0.25,  # Above threshold
            "drift_proportion": 0.35  # Above threshold
        },
        "prediction_drift": {
            "prediction_psi": 0.18
        },
        "overall_drift_detected": True
    }
    
    # Process drift metrics
    events = drift_alert_manager.process_drift_metrics(
        model_name="example_model",
        model_version="1",
        drift_metrics=drift_metrics
    )
    
    print(f"Triggered {len(events)} alerts")