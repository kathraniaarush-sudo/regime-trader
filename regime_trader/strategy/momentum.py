"""Cross-sectional momentum: which names to hold.

Ranks a universe by 12-1 momentum (trailing 12-month return, skipping the most
recent month to avoid short-term reversal) and selects the strongest N. This is
the "selection" half of the strategy; the regime overlay and vol targeting in
portfolio.py decide how much of the basket to actually hold.

All computation is causal: scores at a given point use only bars up to that
point, so the live path and the backtest path agree exactly.
"""
from __future__ import annotations

import pandas as pd


class MomentumRanker:
    def __init__(self, lookback: int = 252, skip: int = 21, top_n: int = 10):
        if skip >= lookback:
            raise ValueError("skip must be smaller than lookback")
        self.lookback = lookback
        self.skip = skip
        self.top_n = top_n

    def scores(self, closes: pd.DataFrame) -> pd.Series:
        """12-1 momentum per column at the most recent bar.

        momentum = close[-1-skip] / close[-1-lookback] - 1, i.e. the return from
        `lookback` bars ago up to `skip` bars ago. Names without enough history
        come back as NaN.
        """
        if len(closes) < self.lookback + 1:
            return pd.Series(index=closes.columns, dtype=float)
        recent = closes.iloc[-1 - self.skip]
        past = closes.iloc[-1 - self.lookback]
        return (recent / past - 1.0).replace([float("inf"), float("-inf")], pd.NA)

    def select(self, closes: pd.DataFrame) -> list[str]:
        """The top-N strongest-momentum tickers (fewer if data is missing)."""
        ranked = self.scores(closes).dropna().sort_values(ascending=False)
        return ranked.head(self.top_n).index.tolist()
