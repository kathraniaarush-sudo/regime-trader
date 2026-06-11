"""Configuration loading.

Two sources, kept strictly separate:
  * config/settings.yaml -> all non-secret tunables (tickers, params, limits)
  * .env                 -> secrets ONLY (Alpaca API keys)

Secrets are never written to YAML and never logged.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "settings.yaml"


@dataclass(frozen=True)
class BrokerCredentials:
    """Loaded from .env. Repr is redacted so keys never leak into logs."""

    api_key: str
    secret_key: str
    base_url: str
    paper: bool

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return f"BrokerCredentials(base_url={self.base_url!r}, paper={self.paper}, keys=***redacted***)"

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key and self.secret_key)


class Settings:
    """Thin wrapper over the parsed YAML with dotted-path access."""

    def __init__(self, data: dict[str, Any], path: Path):
        self._data = data
        self.path = path

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self._data
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    @property
    def raw(self) -> dict[str, Any]:
        return self._data


@lru_cache(maxsize=1)
def load_settings(path: str | os.PathLike[str] | None = None) -> Settings:
    cfg_path = Path(path) if path else DEFAULT_CONFIG
    with open(cfg_path, "r") as fh:
        data = yaml.safe_load(fh) or {}
    return Settings(data, cfg_path)


def load_credentials() -> BrokerCredentials:
    """Read Alpaca secrets from the environment / .env file."""
    load_dotenv(PROJECT_ROOT / ".env")
    base_url = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets").rstrip("/")
    paper = os.getenv("ALPACA_PAPER", "true").strip().lower() in {"1", "true", "yes"}
    return BrokerCredentials(
        api_key=os.getenv("ALPACA_API_KEY", ""),
        secret_key=os.getenv("ALPACA_SECRET_KEY", ""),
        base_url=base_url,
        paper=paper,
    )
