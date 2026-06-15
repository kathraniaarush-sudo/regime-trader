"""Cross-sectional momentum: which names to hold.

Ranks a universe by 12-1 momentum (trailing 12-month return, skipping the most
recent month to avoid short-term reversal) and selects the strongest N. This is
the "selection" half of the strategy; the regime overlay and vol targeting in
portfolio.py decide how much of the basket to actually hold.

Two optional "trend-quality" upgrades (off by default):
  * risk_adjusted: rank by momentum / trailing volatility instead of raw return,
    so steady trends beat jumpy high-vol ones (A2).
  * trend_filter: only keep names trading above their own `trend_ma`-day average,
    i.e. individually in an uptrend, not just relatively strong (A3).

All computation is causal: scores at a given point use only bars up to that
point, so the live path and the backtest path agree exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


class MomentumRanker:
    def __init__(self, lookback: int = 252, skip: int = 21, top_n: int = 10,
                 risk_adjusted: bool = False, trend_filter: bool = False,
                 trend_ma: int = 200):
        if skip >= lookback:
            raise ValueError("skip must be smaller than lookback")
        self.lookback = lookback
        self.skip = skip
        self.top_n = top_n
        self.risk_adjusted = risk_adjusted
        self.trend_filter = trend_filter
        self.trend_ma = trend_ma

    def scores(self, closes: pd.DataFrame) -> pd.Series:
        """Momentum score per column at the most recent bar.

        Base = close[-1-skip] / close[-1-lookback] - 1 (the 12-1 return). If
        risk_adjusted, divide by trailing return volatility over the lookback so
        the score rewards trend *quality*, not just magnitude. Names without
        enough history come back as NaN.
        """
        if len(closes) < self.lookback + 1:
            return pd.Series(index=closes.columns, dtype=float)
        recent = closes.iloc[-1 - self.skip]
        past = closes.iloc[-1 - self.lookback]
        mom = recent / past - 1.0
        if self.risk_adjusted:
            vol = closes.pct_change().iloc[-self.lookback:].std()
            mom = mom / vol.replace(0, np.nan)
        return mom.replace([np.inf, -np.inf], np.nan)

    def _above_trend(self, closes: pd.DataFrame) -> pd.Series:
        """Boolean per column: latest close above its own trend_ma-day average."""
        if len(closes) < self.trend_ma:
            return pd.Series(True, index=closes.columns)
        ma = closes.tail(self.trend_ma).mean()
        return closes.iloc[-1] > ma

    def select(self, closes: pd.DataFrame) -> list[str]:
        """Top-N strongest-momentum tickers (fewer if data is missing or, with
        the trend filter on, if fewer names are individually trending up)."""
        ranked = self.scores(closes).dropna().sort_values(ascending=False)
        if self.trend_filter:
            ok = self._above_trend(closes)
            ranked = ranked[[t for t in ranked.index if bool(ok.get(t, False))]]
        return ranked.head(self.top_n).index.tolist()
