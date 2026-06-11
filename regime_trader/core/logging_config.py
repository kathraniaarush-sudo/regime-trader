"""Structured logging.

Emits human-readable lines to the console and machine-readable JSON lines to
logs/events.jsonl so the dashboard and any external monitor can replay events.
"""
from __future__ import annotations

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

_CONFIGURED = False


class JsonLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        # Anything attached via logger.info(..., extra={"event": {...}})
        event = getattr(record, "event", None)
        if isinstance(event, dict):
            payload.update(event)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def setup_logging(log_dir: str | Path = "logs", level: int = logging.INFO) -> Path:
    """Idempotently configure root logging. Returns the JSONL event-log path."""
    global _CONFIGURED
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)
    events_file = log_path / "events.jsonl"

    if _CONFIGURED:
        return events_file

    root = logging.getLogger()
    root.setLevel(level)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
    root.addHandler(console)

    jsonl = logging.FileHandler(events_file)
    jsonl.setFormatter(JsonLineFormatter())
    root.addHandler(jsonl)

    _CONFIGURED = True
    return events_file


def log_event(logger: logging.Logger, level: int, msg: str, **event) -> None:
    """Log a message plus a structured event dict (lands in the JSONL stream)."""
    logger.log(level, msg, extra={"event": event})
