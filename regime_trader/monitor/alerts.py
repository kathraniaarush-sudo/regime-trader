"""Alerting for critical events (circuit breakers, errors, regime flips).

Optional. If a webhook URL is configured it POSTs a short JSON message;
otherwise it just logs. Never blocks the trading loop on a failed alert.
"""
from __future__ import annotations

import logging

import requests

logger = logging.getLogger("regime.alerts")


class AlertManager:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.webhook_url = cfg.get("webhook_url") or ""
        self.email = cfg.get("email") or ""

    def send(self, level: str, message: str) -> None:
        logger.log(getattr(logging, level.upper(), logging.INFO), "ALERT: %s", message)
        if not (self.enabled and self.webhook_url):
            return
        try:
            requests.post(self.webhook_url, json={"text": f"[{level.upper()}] {message}"}, timeout=10)
        except requests.RequestException as exc:  # never let alerting break trading
            logger.warning("Alert webhook failed: %s", exc)

    def critical(self, message: str) -> None:
        self.send("critical", message)

    def warning(self, message: str) -> None:
        self.send("warning", message)

    def info(self, message: str) -> None:
        self.send("info", message)
