"""Market data.

Two sources:
  * historical daily bars via yfinance — free, used for fitting the HMM and for
    backtesting, so you do not need a paid Alpaca data subscription to develop.
  * latest trade price via Alpaca — used by the live loop for sizing.

Both return a uniform OHLCV frame indexed by date with lowercase columns.
"""
from __future__ import annotations

import logging

import pandas as pd

from regime_trader.broker.alpaca_client import AlpacaClient

logger = logging.getLogger("regime.data")


def get_history(symbol: str, lookback_days: int = 504, interval: str = "1d") -> pd.DataFrame:
    """Daily OHLCV history from yfinance, normalised to lowercase columns."""
    import yfinance as yf

    # Pad the calendar window so weekends/holidays still leave enough bars.
    period_days = int(lookback_days * 1.6) + 40
    df = yf.download(
        symbol,
        period=f"{period_days}d",
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"No history returned for {symbol}")

    # yfinance may return a MultiIndex column frame for a single ticker.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep].dropna()
    df.index = pd.to_datetime(df.index)

    # Drop the current session's bar. Intraday it is incomplete and updates on
    # every fetch, which would make backtests non-reproducible and would train /
    # detect on a partial bar. We only ever act on completed daily bars.
    if interval.endswith("d"):
        today = pd.Timestamp.now().normalize()
        df = df[df.index.normalize() < today]

    return df.tail(lookback_days)


class MarketData:
    """Live price helper backed by Alpaca's data API."""

    def __init__(self, client: AlpacaClient):
        self.client = client
        # Alpaca data lives on a different host than the trading API.
        self.data_url = "https://data.alpaca.markets"

    def latest_price(self, symbol: str) -> float:
        url = f"{self.data_url}/v2/stocks/{symbol}/trades/latest"
        # Reuse the authenticated session but hit the data host directly.
        resp = self.client.session.get(url, timeout=15)
        if resp.status_code >= 400:
            raise RuntimeError(f"Alpaca data {symbol} -> {resp.status_code}: {resp.text}")
        return float(resp.json()["trade"]["p"])
