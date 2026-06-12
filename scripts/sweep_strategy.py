"""Strategy parameter sweep — find the Sharpe-best allocation, transparently.

Runs the SAME walk-forward backtester used everywhere else across a grid of
candidate allocations and prints a comparison table. Nothing here is hand-tuned:
you see every candidate and the metric that ranks them.

Usage:
    python scripts/sweep_strategy.py            # sweeps the regime-anchor ticker
    python scripts/sweep_strategy.py QQQ         # sweep a different ticker

Principled defaults baked into the grid (driven by the regime breakdown):
  * crash / bear  -> flat (negative historical Sharpe, so zero exposure)
  * bull          -> 0.95 (high Sharpe)
  * euphoria      -> 0.95 @ 1.25x leverage (highest Sharpe)
  * neutral       -> the lever we sweep (it is frequent but mediocre Sharpe)
We also toggle confidence_scaling, which steadily trims exposure below 100%
confidence and therefore trades a little return for a little smoothness.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regime_trader.backtest.backtester import WalkForwardBacktester
from regime_trader.broker.market_data import get_history
from regime_trader.core.settings import load_settings


def make_strategy(neutral: float, confidence_scaling: bool, bear: float = 0.0) -> dict:
    return {
        "rebalance_threshold": 0.05,
        "confidence_scaling": confidence_scaling,
        "min_confidence": 0.50,
        "regimes": {
            "crash": {"gross_exposure": 0.00, "leverage": 1.00, "direction": "flat"},
            "bear": {"gross_exposure": bear, "leverage": 1.00,
                     "direction": "flat" if bear == 0 else "long"},
            "neutral": {"gross_exposure": neutral, "leverage": 1.00, "direction": "long"},
            "bull": {"gross_exposure": 0.95, "leverage": 1.00, "direction": "long"},
            "euphoria": {"gross_exposure": 0.95, "leverage": 1.25, "direction": "long"},
        },
    }


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else None
    settings = load_settings()
    anchor = ticker or settings.get("universe.regime_anchor", "SPY")
    hmm_cfg = settings.get("hmm", {})
    bt_cfg = settings.get("backtest", {})

    prices = get_history(anchor, max(settings.get("hmm.train_lookback_days", 504), 1200))

    # Each candidate: (label, neutral_exposure, confidence_scaling, bear_exposure)
    candidates = [
        ("default (bear .25, neut .50, conf)", 0.50, True, 0.25),
        ("bear flat, neut .25, conf",          0.25, True, 0.0),
        ("bear flat, neut .40, conf",          0.40, True, 0.0),
        ("bear flat, neut .50, conf",          0.50, True, 0.0),
        ("bear flat, neut .65, conf",          0.65, True, 0.0),
        ("bear flat, neut .25, no-conf",       0.25, False, 0.0),
        ("bear flat, neut .40, no-conf",       0.40, False, 0.0),
        ("bear flat, neut .50, no-conf",       0.50, False, 0.0),
    ]

    print(f"\nSweep on {anchor} — ranked by Sharpe (walk-forward, out-of-sample)\n")
    header = f"{'candidate':38s} {'return':>8s} {'sharpe':>7s} {'maxDD':>7s} {'calmar':>7s}"
    print(header)
    print("-" * len(header))

    rows = []
    for label, neutral, conf, bear in candidates:
        bt = WalkForwardBacktester(hmm_cfg, make_strategy(neutral, conf, bear), bt_cfg)
        res = bt.run(prices)
        m = res.metrics
        rows.append((label, m.total_return, m.sharpe, m.max_drawdown, m.calmar))

    for label, ret, sharpe, dd, calmar in sorted(rows, key=lambda r: r[2], reverse=True):
        print(f"{label:38s} {ret:+7.1%} {sharpe:7.2f} {dd:7.1%} {calmar:7.2f}")

    best = max(rows, key=lambda r: r[2])
    print(f"\nBest Sharpe: {best[0]}  (sharpe={best[2]:.2f}, return={best[1]:+.1%}, maxDD={best[3]:.1%})")


if __name__ == "__main__":
    main()
