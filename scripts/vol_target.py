"""Volatility targeting on the momentum + HMM-regime strategy.

Goal: pull max drawdown under the live -10% circuit breaker while keeping most
of the Sharpe, so the strategy is actually deployable.

How: scale the basket's daily exposure by target_vol / realized_vol, where
realized_vol is the strategy's own trailing 20-day annualised volatility. Calm
stretches get scaled up (toward a leverage cap), wild stretches get scaled down.
Everything is causal: the scalar uses returns through day t and is applied to the
t+1 return (scalar.shift(1)).

We reuse the de-biased pipeline from validate_momentum (broad universe, causal
HMM regime overlay) so the only new variable is the vol-target layer.

Usage:  python scripts/vol_target.py
"""
from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                  # import the sibling research script
sys.path.insert(0, str(HERE.parent))           # and the regime_trader package

import numpy as np
import pandas as pd

from regime_trader.backtest.performance import compute_metrics
from validate_momentum import (
    fetch_closes, compute_regime_gross, momentum_returns, rolling_cum,
    UNIVERSE, WINDOW_30, LOOKBACK,
)

VOL_LOOKBACK = 20
ANN = np.sqrt(252)


def vol_target(raw_ret: pd.Series, target_vol: float, max_lev: float) -> pd.Series:
    """Scale daily returns toward a constant target annualised volatility."""
    realized = raw_ret.rolling(VOL_LOOKBACK).std() * ANN
    scalar = (target_vol / realized.replace(0, np.nan)).clip(0, max_lev).fillna(0.0)
    return (scalar.shift(1) * raw_ret).dropna()


def win_rate(strat: pd.Series, bench: pd.Series) -> float:
    df = pd.DataFrame({"s": rolling_cum(strat, WINDOW_30),
                       "b": rolling_cum(bench, WINDOW_30)}).dropna()
    return ((df["s"] - df["b"]) > 0).mean()


def row(name: str, port: pd.Series, spy: pd.Series) -> None:
    m = compute_metrics(100000 * (1 + port).cumprod())
    breaker = "  HALTS" if m.max_drawdown < -0.10 else "  ok"
    print(f"{name:36s} {m.total_return:+8.1%} {m.sharpe:6.2f} {m.max_drawdown:7.1%} "
          f"{win_rate(port, spy):7.0%}{breaker}")


def main() -> None:
    print("Downloading universe + computing regime overlay (~40s)...")
    closes = fetch_closes(UNIVERSE + ["SPY", "RSP"]).dropna(axis=1)
    stocks = [c for c in closes.columns if c not in ("SPY", "RSP")]
    spy = closes["SPY"].pct_change().loc[closes.index[LOOKBACK + 1]:]

    gross = compute_regime_gross(closes)
    raw = momentum_returns(closes, stocks, gross)   # regime-overlaid, un-vol-targeted

    hdr = f"{'strategy':36s} {'return':>8s} {'sharpe':>6s} {'maxDD':>7s} {'>SPY':>7s}"
    print("\n" + hdr)
    print("-" * (len(hdr) + 8))
    row("SPY buy & hold", spy, spy)
    row("Momentum+HMM (no vol target)", raw, spy)

    print("  -- de-lever only (cap 1.0x): can only reduce exposure --")
    for tv in (0.15, 0.12, 0.10):
        row(f"vol-target {tv:.0%}, cap 1.0x", vol_target(raw, tv, 1.0), spy)

    print("  -- allow modest leverage (cap 1.5x) --")
    for tv in (0.15, 0.12, 0.10):
        row(f"vol-target {tv:.0%}, cap 1.5x", vol_target(raw, tv, 1.5), spy)

    print("\n'maxDD' flagged HALTS if it breaches the live -10% circuit breaker.")
    print("'>SPY' = share of rolling 30-day windows beating SPY. Survivorship caveat still applies.")


if __name__ == "__main__":
    main()
