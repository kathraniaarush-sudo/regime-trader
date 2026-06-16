"""A/B test: does the regime cash-gate help or hurt?

Compares the strategy WITH the regime overlay (go to cash in bear/crash) against
WITHOUT it (always invested in the momentum basket at the vol target, regime
ignored). Same selection signal (whatever the config uses, incl. A2) and same
vol targeting in both arms — the ONLY difference is the cash-gate.

Run on the full universe and ex-mega-cap, vs SPY and equal-weight RSP, so the
verdict survives de-biasing.

Usage:  python scripts/regime_ab.py
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
ALWAYS_ON = {"crash": 1.0, "bear": 1.0, "neutral": 1.0, "bull": 1.0, "euphoria": 1.0}


def rolling_cum(r, w=30):
    return (1 + r).rolling(w).apply(np.prod, raw=True) - 1


def row(name, res, spy_ret, rsp_ret):
    m, port = res.metrics, res.returns
    wsp = ((rolling_cum(port) - rolling_cum(spy_ret)).dropna() > 0).mean()
    wrsp = ((rolling_cum(port) - rolling_cum(rsp_ret)).dropna() > 0).mean()
    print(f"{name:30s} {m.total_return:+7.1%} {m.sharpe:6.2f} {m.max_drawdown:7.1%} "
          f"{wsp:6.0%} {wrsp:6.0%}")


def run_pair(closes, anchor, hmm_cfg, pcfg, bt_cfg, spy_ret, rsp_ret, suffix=""):
    with_gate = dict(pcfg)
    no_gate = dict(pcfg, regime_gross=ALWAYS_ON, daily_derisk=False)
    hdr = f"{'arm'+suffix:30s} {'return':>7s} {'sharpe':>6s} {'maxDD':>7s} {'>SPY':>6s} {'>RSP':>6s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    row("WITH regime cash-gate", PortfolioBacktester(hmm_cfg, with_gate, bt_cfg).run(closes, anchor=anchor),
        spy_ret, rsp_ret)
    row("WITHOUT gate (always in)", PortfolioBacktester(hmm_cfg, no_gate, bt_cfg).run(closes, anchor=anchor),
        spy_ret, rsp_ret)


def main() -> None:
    s = load_settings()
    hmm_cfg, bt_cfg = s.get("hmm", {}), s.get("backtest", {})
    pcfg = dict(s.get("portfolio", {}))
    anchor = s.get("universe.regime_anchor", "SPY")
    universe = list(pcfg.get("universe", []))

    print("Fetching universe + benchmarks (~40s)...")
    # ffill intra-series gaps first so a single missing bar doesn't drop a whole
    # column (incl. the anchor); dropna(axis=1) then only removes short-history names.
    closes = get_histories(universe + [anchor, "RSP"], lookback_days=1500).ffill().dropna(axis=1)
    if anchor not in closes.columns or "RSP" not in closes.columns:
        raise SystemExit(f"missing benchmark after fetch (have anchor={anchor in closes.columns}, "
                         f"RSP={'RSP' in closes.columns}) — try re-running")
    stocks = [c for c in closes.columns if c not in (anchor, "RSP")]
    spy_ret, rsp_ret = closes[anchor].pct_change(), closes["RSP"].pct_change()
    base = closes.drop(columns=["RSP"])
    print(f"{len(stocks)} names, {len(closes)} bars "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})  (A2={pcfg.get('risk_adjusted_momentum')})")

    run_pair(base, anchor, hmm_cfg, pcfg, bt_cfg, spy_ret, rsp_ret)
    keep = [c for c in base.columns if c == anchor or c not in MEGA7]
    run_pair(base[keep], anchor, hmm_cfg, pcfg, bt_cfg, spy_ret, rsp_ret, suffix=" (ex-mega7)")

    print("\nIf 'WITHOUT' has the higher Sharpe, the cash-gate is costing risk-adjusted")
    print("return; if 'WITH' has materially lower drawdown, the gate is earning its keep.")


if __name__ == "__main__":
    main()
