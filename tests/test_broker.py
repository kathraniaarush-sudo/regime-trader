"""Broker tests using a fake requests session — no network, no real account."""
from __future__ import annotations

import json

from regime_trader.broker.alpaca_client import AlpacaClient
from regime_trader.broker.order_executor import OrderExecutor
from regime_trader.broker.position_tracker import PositionTracker
from regime_trader.core.settings import BrokerCredentials


class FakeResponse:
    def __init__(self, status_code=200, payload=None, text=None):
        self.status_code = status_code
        self._payload = payload if payload is not None else {}
        self.text = text if text is not None else json.dumps(self._payload)

    def json(self):
        return self._payload


class FakeSession:
    """Records calls and returns canned responses keyed by (method, path-suffix)."""

    def __init__(self, routes):
        self.routes = routes
        self.headers = {}
        self.calls = []

    def request(self, method, url, timeout=None, json=None):
        self.calls.append((method, url, json))
        for (m, suffix), resp in self.routes.items():
            if method == m and url.endswith(suffix):
                return resp
        return FakeResponse(404, {"message": "not found"}, "not found")

    def get(self, url, timeout=None):
        return self.request("GET", url)


def _creds():
    return BrokerCredentials("KEY", "SECRET", "https://paper-api.alpaca.markets", True)


def test_get_account_parses_fields():
    session = FakeSession({
        ("GET", "/v2/account"): FakeResponse(200, {
            "equity": "100000", "cash": "50000", "buying_power": "200000", "status": "ACTIVE"})
    })
    client = AlpacaClient(_creds(), session)
    acct = client.get_account()
    assert acct.equity == 100000
    assert acct.buying_power == 200000
    assert acct.status == "ACTIVE"


def test_ping_succeeds_when_account_active():
    session = FakeSession({
        ("GET", "/v2/account"): FakeResponse(200, {
            "equity": "100000", "cash": "1", "buying_power": "1", "status": "ACTIVE"})
    })
    assert AlpacaClient(_creds(), session).ping() is True


def test_unconfigured_client_raises():
    creds = BrokerCredentials("", "", "https://paper-api.alpaca.markets", True)
    client = AlpacaClient(creds, FakeSession({}))
    assert client.is_configured is False
    try:
        client.get_account()
        assert False, "expected AlpacaError"
    except Exception as exc:
        assert "credentials" in str(exc).lower()


def test_submit_market_order():
    session = FakeSession({
        ("POST", "/v2/orders"): FakeResponse(200, {
            "id": "abc", "symbol": "NVDA", "side": "buy", "qty": "10", "status": "accepted"})
    })
    ex = OrderExecutor(AlpacaClient(_creds(), session))
    res = ex.submit_market_order("NVDA", 10, "buy")
    assert res.id == "abc"
    assert res.status == "accepted"
    # verify the payload we sent
    _, _, payload = session.calls[-1]
    assert payload["symbol"] == "NVDA" and payload["type"] == "market"


def test_submit_rejects_bad_input():
    ex = OrderExecutor(AlpacaClient(_creds(), FakeSession({})))
    for bad in [("NVDA", 0, "buy"), ("NVDA", 5, "hodl")]:
        try:
            ex.submit_market_order(*bad)
            assert False
        except ValueError:
            pass


def test_position_weights():
    session = FakeSession({
        ("GET", "/v2/positions"): FakeResponse(200, [
            {"symbol": "SPY", "qty": "100", "avg_entry_price": "400",
             "market_value": "42000", "unrealized_pl": "2000", "side": "long"}])
    })
    pt = PositionTracker(AlpacaClient(_creds(), session))
    weights = pt.weights(equity=100000)
    assert abs(weights["SPY"] - 0.42) < 1e-9
