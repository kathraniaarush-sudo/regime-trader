"""Thin Alpaca REST wrapper.

Implemented directly against the REST API with `requests` (rather than a heavy
SDK) so it is easy to mock in tests and stable across SDK releases. Credentials
come from the environment via core.settings — never hard-coded, never logged.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import requests

from regime_trader.core.settings import BrokerCredentials, load_credentials

logger = logging.getLogger("regime.broker")


class AlpacaError(RuntimeError):
    pass


@dataclass
class Account:
    equity: float
    cash: float
    buying_power: float
    status: str
    raw: dict[str, Any]


class AlpacaClient:
    def __init__(self, creds: BrokerCredentials | None = None, session: requests.Session | None = None):
        self.creds = creds or load_credentials()
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "APCA-API-KEY-ID": self.creds.api_key,
                "APCA-API-SECRET-KEY": self.creds.secret_key,
            }
        )

    @property
    def is_configured(self) -> bool:
        return self.creds.is_configured

    # ----------------------------------------------------------- transport
    def _request(self, method: str, path: str, **kwargs) -> Any:
        if not self.is_configured:
            raise AlpacaError("Alpaca credentials are not set (check your .env file)")
        url = f"{self.creds.base_url}{path}"
        try:
            resp = self.session.request(method, url, timeout=15, **kwargs)
        except requests.RequestException as exc:
            raise AlpacaError(f"Network error talking to Alpaca: {exc}") from exc
        if resp.status_code >= 400:
            raise AlpacaError(f"Alpaca {method} {path} -> {resp.status_code}: {resp.text}")
        if resp.text:
            return resp.json()
        return None

    def get(self, path: str, **kw) -> Any:
        return self._request("GET", path, **kw)

    def post(self, path: str, json: dict | None = None) -> Any:
        return self._request("POST", path, json=json)

    def delete(self, path: str) -> Any:
        return self._request("DELETE", path)

    # ------------------------------------------------------------- account
    def get_account(self) -> Account:
        data = self.get("/v2/account")
        return Account(
            equity=float(data["equity"]),
            cash=float(data["cash"]),
            buying_power=float(data["buying_power"]),
            status=data.get("status", "UNKNOWN"),
            raw=data,
        )

    def is_market_open(self) -> bool:
        clock = self.get("/v2/clock")
        return bool(clock.get("is_open", False))

    def ping(self) -> bool:
        """Lightweight connectivity + auth check used at startup."""
        try:
            acct = self.get_account()
        except AlpacaError as exc:
            logger.error("Alpaca connection check failed: %s", exc)
            return False
        logger.info("Alpaca connected: status=%s equity=%.2f (%s)",
                    acct.status, acct.equity, "paper" if self.creds.paper else "LIVE")
        return True
