"""Order execution: submit, modify, and cancel orders through Alpaca."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from regime_trader.broker.alpaca_client import AlpacaClient

logger = logging.getLogger("regime.executor")


@dataclass
class OrderResult:
    id: str
    symbol: str
    side: str
    qty: float
    status: str
    raw: dict[str, Any]


class OrderExecutor:
    def __init__(self, client: AlpacaClient):
        self.client = client

    def submit_market_order(self, symbol: str, qty: int, side: str,
                            time_in_force: str = "day") -> OrderResult:
        if qty <= 0:
            raise ValueError("qty must be positive")
        if side not in {"buy", "sell"}:
            raise ValueError("side must be 'buy' or 'sell'")
        payload = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
        }
        data = self.client.post("/v2/orders", json=payload)
        logger.info("Submitted %s %d %s -> %s", side, qty, symbol, data.get("status"))
        return self._to_result(data)

    def submit_bracket_order(self, symbol: str, qty: int, side: str,
                             stop_price: float, take_profit: float | None = None,
                             time_in_force: str = "day") -> OrderResult:
        """Market entry with an attached protective stop (and optional target)."""
        payload: dict[str, Any] = {
            "symbol": symbol,
            "qty": str(int(qty)),
            "side": side,
            "type": "market",
            "time_in_force": time_in_force,
            "order_class": "bracket",
            "stop_loss": {"stop_price": round(stop_price, 2)},
        }
        if take_profit is not None:
            payload["take_profit"] = {"limit_price": round(take_profit, 2)}
        data = self.client.post("/v2/orders", json=payload)
        logger.info("Bracket %s %d %s stop=%.2f -> %s",
                    side, qty, symbol, stop_price, data.get("status"))
        return self._to_result(data)

    def cancel_order(self, order_id: str) -> None:
        self.client.delete(f"/v2/orders/{order_id}")
        logger.info("Cancelled order %s", order_id)

    def cancel_all(self) -> None:
        self.client.delete("/v2/orders")
        logger.info("Cancelled all open orders")

    def close_position(self, symbol: str) -> None:
        self.client.delete(f"/v2/positions/{symbol}")
        logger.info("Closed position %s", symbol)

    def close_all_positions(self) -> None:
        self.client.delete("/v2/positions")
        logger.warning("Closed ALL positions (flatten)")

    @staticmethod
    def _to_result(data: dict) -> OrderResult:
        return OrderResult(
            id=data.get("id", ""),
            symbol=data.get("symbol", ""),
            side=data.get("side", ""),
            qty=float(data.get("qty", 0) or 0),
            status=data.get("status", ""),
            raw=data,
        )
