"""How often does each profile beat SPY over a rolling 30-trading-day window?

This is the decision-relevant metric for a 30-day challenge. A single multi-year
return tells you little about a one-month bet; the distribution of 30-day
outcomes does. For each profile we slide a 30-trading-day window across the whole
backtest and measure:
  * win rate  -> fraction of windows where the strategy beat SPY buy & hold
  * median / best / worst 30-day excess return vs SPY

Usage:  python scripts/win_rate_30d.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from regime_trader.backtest.backtester import WalkForwardBacktester
from regime_trader.broker.market_data import get_history
from regime_trader.core.settings import load_settings

WINDOW = 30


def strat(neutral: float, bull_lev: float = 1.0, euph_lev: float = 1.25) -> dict:
    return {
        "confidence_scaling": False,
        "min_confidence": 0.50,
        "regimes": {
            "crash": {"gross_exposure": 0.00, "leverage": 1.00, "direction": "flat"},
            "bear": {"gross_exposure": 0.00, "leverage": 1.00, "direction": "flat"},
            "neutral": {"gross_exposure": neutral, "leverage": 1.00, "direction": "long"},
            "bull": {"gross_exposure": 0.95, "leverage": bull_lev, "direction": "long"},
            "euphoria": {"gross_exposure": 0.95, "leverage": euph_lev, "direction": "long"},
        },
    }


def rolling_cum(returns: pd.Series, window: int) -> pd.Series:
    """Trailing `window`-day compounded return for each day."""
    return (1 + returns).rolling(window).apply(np.prod, raw=True) - 1


def main() -> None:
    settings = load_settings()
    hmm_cfg, bt_cfg = settings.get("hmm", {}), settings.get("backtest", {})
    lookback = max(settings.get("hmm.train_lookback_days", 504), 1200)

    # SPY daily buy & hold returns over the common traded window = the benchmark.
    spy_run = WalkForwardBacktester(hmm_cfg, strat(0.40), bt_cfg).run(get_history("SPY", lookback))
    spy_ret = spy_run.daily["asset_ret"]
    spy_roll = rolling_cum(spy_ret, WINDOW)

    candidates = [
        ("SPY  - Sharpe-tuned (current)", "SPY", strat(0.40, 1.00, 1.25)),
        ("SPY  - leveraged 1.25/1.5x",    "SPY", strat(0.60, 1.25, 1.50)),
        ("QQQ  - modest 1.0/1.25x",        "QQQ", strat(0.60, 1.00, 1.25)),
        ("TQQQ - 3x ETF, no extra lev",    "TQQQ", strat(0.50, 1.00, 1.00)),
    ]

    print(f"\nRolling {WINDOW}-trading-day win rate vs SPY buy & hold\n")
    header = f"{'profile':32s} {'win rate':>9s} {'median':>8s} {'best':>8s} {'worst':>8s}"
    print(header)
    print("-" * len(header))

    rows = []
    for label, ticker, strategy in candidates:
        try:
            res = WalkForwardBacktester(hmm_cfg, strategy, bt_cfg).run(get_history(ticker, lookback))
            roll = rolling_cum(res.daily["strat_ret"], WINDOW)
            df = pd.DataFrame({"s": roll, "b": spy_roll}).dropna()
            excess = df["s"] - df["b"]
            rows.append((label, (excess > 0).mean(), excess.median(), excess.max(), excess.min()))
        except Exception as exc:
            print(f"{label:32s}  (skipped: {exc})")

    for label, wr, med, best, worst in sorted(rows, key=lambda r: r[1], reverse=True):
        print(f"{label:32s} {wr:8.0%} {med:+8.1%} {best:+8.1%} {worst:+8.1%}")

    print(f"\n{WINDOW}-day windows sampled across the full backtest. 'win rate' = share of")
    print("windows the profile finished ahead of SPY. ~50% means a coin flip.")


if __name__ == "__main__":
    main()
