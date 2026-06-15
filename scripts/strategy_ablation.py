"""Ablation: does the trend-quality bundle beat plain momentum?

Backtests the baseline (raw 12-1 momentum, equal weight) against each upgrade
added one at a time, and the full bundle, using the SAME walk-forward engine.
Runs on the full universe and again EXCLUDING the mega-cap winners, and reports
vs both SPY and equal-weight RSP, so an improvement has to survive de-biasing —
not just look good on the survivorship-biased sample.

  A2 = risk-adjusted momentum (rank by return / volatility)
  A3 = per-stock trend filter (only hold names above their 200-day average)
  B1 = inverse-volatility weighting

Usage:  python scripts/strategy_ablation.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from regime_trader.backtest.portfolio_backtester import PortfolioBacktester
from regime_trader.broker.market_data import get_histories
from regime_trader.core.settings import load_settings

MEGA7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
WINDOW_30 = 30


def cfg(base: dict, **over) -> dict:
    c = dict(base)
    c.update(over)
    return c


def rolling_cum(r, w=WINDOW_30):
    return (1 + r).rolling(w).apply(np.prod, raw=True) - 1


def run_table(closes, anchor, hmm_cfg, base_pcfg, bt_cfg, spy_ret, rsp_ret, label_suffix=""):
    variants = [
        ("baseline (raw mom, equal wt)", cfg(base_pcfg)),
        ("+A2 risk-adjusted",            cfg(base_pcfg, risk_adjusted_momentum=True)),
        ("+A3 trend filter",             cfg(base_pcfg, trend_filter=True)),
        ("+B1 inverse-vol weight",       cfg(base_pcfg, weighting="inverse_vol")),
        ("BUNDLE A2+A3+B1",              cfg(base_pcfg, risk_adjusted_momentum=True,
                                            trend_filter=True, weighting="inverse_vol")),
    ]
    hdr = f"{'variant'+label_suffix:34s} {'return':>8s} {'sharpe':>6s} {'maxDD':>7s} {'>SPY':>6s} {'>RSP':>6s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for name, pcfg in variants:
        res = PortfolioBacktester(hmm_cfg, pcfg, bt_cfg).run(closes, anchor=anchor)
        m, port = res.metrics, res.returns
        wsp = ((rolling_cum(port) - rolling_cum(spy_ret)).dropna() > 0).mean()
        wrsp = ((rolling_cum(port) - rolling_cum(rsp_ret)).dropna() > 0).mean()
        print(f"{name:34s} {m.total_return:+7.1%} {m.sharpe:6.2f} {m.max_drawdown:7.1%} "
              f"{wsp:6.0%} {wrsp:6.0%}")


def main() -> None:
    s = load_settings()
    hmm_cfg = s.get("hmm", {})
    bt_cfg = s.get("backtest", {})
    base_pcfg = dict(s.get("portfolio", {}))
    # force baseline behaviour as the starting point for the ablation
    base_pcfg.update(risk_adjusted_momentum=False, trend_filter=False, weighting="equal")
    anchor = s.get("universe.regime_anchor", "SPY")
    universe = list(base_pcfg.get("universe", []))

    print("Fetching universe + benchmarks (~40s)...")
    closes = get_histories(universe + [anchor, "RSP"], lookback_days=1500).dropna(axis=1)
    stocks = [c for c in closes.columns if c not in (anchor, "RSP")]
    spy_ret = closes[anchor].pct_change()
    rsp_ret = closes["RSP"].pct_change()
    closes_noidx = closes.drop(columns=["RSP"])
    print(f"{len(stocks)} names, {len(closes)} bars "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})")
    print("Each variant = a full walk-forward (HMM refit per rebalance). Patience.")

    # 1) full universe
    run_table(closes_noidx, anchor, hmm_cfg, base_pcfg, bt_cfg, spy_ret, rsp_ret)

    # 2) de-bias: exclude the mega-cap winners
    keep = [c for c in closes_noidx.columns if c == anchor or c not in MEGA7]
    run_table(closes_noidx[keep], anchor, hmm_cfg, base_pcfg, bt_cfg, spy_ret, rsp_ret,
              label_suffix=" (ex-mega7)")

    print("\n'>SPY'/'>RSP' = share of rolling 30-day windows beating that benchmark.")
    print("A variant only 'wins' if it improves Sharpe AND holds up ex-mega-cap. "
          "Survivorship caveat still applies.")


if __name__ == "__main__":
    main()
