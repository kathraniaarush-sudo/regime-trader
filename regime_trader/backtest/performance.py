"""Performance analytics for an equity / returns series."""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class PerformanceMetrics:
    total_return: float
    cagr: float
    ann_vol: float
    sharpe: float
    sortino: float
    max_drawdown: float
    calmar: float
    win_rate: float
    n_periods: int

    def as_dict(self) -> dict:
        return asdict(self)


def _to_returns(equity: pd.Series) -> pd.Series:
    return equity.pct_change().dropna()


def max_drawdown(equity: pd.Series) -> float:
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = (equity - peak) / peak
    return float(dd.min())


def compute_metrics(equity: pd.Series, periods_per_year: int = TRADING_DAYS) -> PerformanceMetrics:
    """Standard risk/return metrics from an equity curve."""
    equity = equity.dropna()
    if len(equity) < 2:
        return PerformanceMetrics(0, 0, 0, 0, 0, 0, 0, 0, len(equity))

    rets = _to_returns(equity)
    total_return = float(equity.iloc[-1] / equity.iloc[0] - 1.0)
    years = len(rets) / periods_per_year
    cagr = float((equity.iloc[-1] / equity.iloc[0]) ** (1 / years) - 1.0) if years > 0 else 0.0

    ann_vol = float(rets.std(ddof=0) * np.sqrt(periods_per_year))
    mean_ann = float(rets.mean() * periods_per_year)
    sharpe = mean_ann / ann_vol if ann_vol > 0 else 0.0

    downside = rets[rets < 0]
    downside_vol = float(downside.std(ddof=0) * np.sqrt(periods_per_year)) if not downside.empty else 0.0
    sortino = mean_ann / downside_vol if downside_vol > 0 else 0.0

    mdd = max_drawdown(equity)
    calmar = cagr / abs(mdd) if mdd < 0 else 0.0
    win_rate = float((rets > 0).mean())

    return PerformanceMetrics(
        total_return=total_return,
        cagr=cagr,
        ann_vol=ann_vol,
        sharpe=sharpe,
        sortino=sortino,
        max_drawdown=mdd,
        calmar=calmar,
        win_rate=win_rate,
        n_periods=len(rets),
    )


def metrics_by_regime(equity_rets: pd.Series, regimes: pd.Series) -> pd.DataFrame:
    """Break daily returns down by the regime that was active that day."""
    df = pd.DataFrame({"ret": equity_rets, "regime": regimes}).dropna()
    if df.empty:
        return pd.DataFrame()
    grouped = df.groupby("regime")["ret"]
    out = pd.DataFrame(
        {
            "days": grouped.count(),
            "mean_ret": grouped.mean(),
            "ann_ret": grouped.mean() * TRADING_DAYS,
            "ann_vol": grouped.std(ddof=0) * np.sqrt(TRADING_DAYS),
            "win_rate": grouped.apply(lambda s: (s > 0).mean()),
        }
    )
    out["sharpe"] = (out["ann_ret"] / out["ann_vol"]).replace([np.inf, -np.inf], 0).fillna(0)
    return out
