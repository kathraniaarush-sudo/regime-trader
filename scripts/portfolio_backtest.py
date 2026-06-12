"""Backtest the packaged momentum + regime + vol-target portfolio.

Confirms the in-package PortfolioBacktester (which shares its strategy objects
with the live loop) reproduces the validated research numbers, and reports the
drawdown against the -10% circuit breaker.

Usage:  python scripts/portfolio_backtest.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regime_trader.backtest.portfolio_backtester import PortfolioBacktester
from regime_trader.broker.market_data import get_histories
from regime_trader.core.settings import load_settings


def main() -> None:
    s = load_settings()
    pcfg = s.get("portfolio", {})
    anchor = s.get("universe.regime_anchor", "SPY")
    universe = list(pcfg.get("universe", []))

    print(f"Fetching {len(universe)} names + {anchor} (~30s)...")
    closes = get_histories(universe + [anchor], lookback_days=1500).dropna(axis=1)
    have = [c for c in closes.columns if c != anchor]
    print(f"{len(have)} names with full history, {len(closes)} bars "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})\n")
    print("Running walk-forward (fits the HMM overlay each rebalance, ~60s)...")

    bt = PortfolioBacktester(s.get("hmm", {}), pcfg, s.get("backtest", {}))
    res = bt.run(closes, anchor=anchor)
    m, b = res.metrics, res.benchmark

    halts = "  ⚠ BREACHES -10% breaker" if m.max_drawdown < -0.10 else "  ✓ within -10% breaker"
    print(f"\n=== Momentum + HMM + vol-target  ({res.n_rebalances} rebalances) ===")
    print(f"Total return : {m.total_return:+.1%}   (SPY: {b.total_return:+.1%})")
    print(f"CAGR         : {m.cagr:+.1%}   (SPY: {b.cagr:+.1%})")
    print(f"Sharpe       : {m.sharpe:.2f}   (SPY: {b.sharpe:.2f})")
    print(f"Max drawdown : {m.max_drawdown:.1%}   (SPY: {b.max_drawdown:.1%}){halts}")
    print(f"30-day win rate vs SPY : {res.win_rate_30d:.0%}")
    print("\nSurvivorship caveat applies; treat as plausible, not guaranteed.")


if __name__ == "__main__":
    main()
