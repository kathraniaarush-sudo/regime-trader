"""Track open positions and current portfolio weights from Alpaca."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from regime_trader.broker.alpaca_client import AlpacaClient

logger = logging.getLogger("regime.positions")


@dataclass
class Position:
    symbol: str
    qty: float
    avg_entry: float
    market_value: float
    unrealized_pl: float
    side: str

    @property
    def unrealized_pl_pct(self) -> float:
        cost = abs(self.avg_entry * self.qty)
        return (self.unrealized_pl / cost) if cost else 0.0


class PositionTracker:
    def __init__(self, client: AlpacaClient):
        self.client = client

    def list_positions(self) -> list[Position]:
        data = self.client.get("/v2/positions") or []
        return [self._to_position(p) for p in data]

    def get_position(self, symbol: str) -> Position | None:
        for p in self.list_positions():
            if p.symbol == symbol:
                return p
        return None

    def weights(self, equity: float) -> dict[str, float]:
        """Signed market-value weight of each holding relative to equity."""
        if equity <= 0:
            return {}
        return {p.symbol: p.market_value / equity for p in self.list_positions()}

    @staticmethod
    def _to_position(p: dict[str, Any]) -> Position:
        qty = float(p.get("qty", 0) or 0)
        return Position(
            symbol=p.get("symbol", ""),
            qty=qty,
            avg_entry=float(p.get("avg_entry_price", 0) or 0),
            market_value=float(p.get("market_value", 0) or 0),
            unrealized_pl=float(p.get("unrealized_pl", 0) or 0),
            side=p.get("side", "long"),
        )
