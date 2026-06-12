"""Instrument comparison for the 'beat the S&P 500' goal.

Backtests several instrument + leverage profiles with the SAME regime/risk
engine, and ranks them by EXCESS RETURN over simply buying and holding SPY over
the identical window. Drawdown and Sharpe are shown so you can see what risk you
take on to chase that excess.

Usage:
    python scripts/compare_instruments.py

IMPORTANT: the walk-forward backtester is a pure allocation simulation. It does
NOT apply the live circuit breakers, so the drawdowns shown for high-octane
instruments (e.g. TQQQ) are the RAW strategy drawdowns. In live trading the risk
manager would halt at the -10% drawdown breaker long before TQQQ's worst.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from regime_trader.backtest.backtester import WalkForwardBacktester
from regime_trader.broker.market_data import get_history
from regime_trader.core.settings import load_settings


def strat(neutral: float, bull_lev: float = 1.0, euph_lev: float = 1.25) -> dict:
    """Return-oriented config: bear/crash flat, confidence-scaling off."""
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


def main() -> None:
    settings = load_settings()
    hmm_cfg = settings.get("hmm", {})
    bt_cfg = settings.get("backtest", {})
    lookback = max(settings.get("hmm.train_lookback_days", 504), 1200)

    # (label, ticker, strategy)
    candidates = [
        ("SPY  - Sharpe-tuned (current)", "SPY", strat(0.40, 1.00, 1.25)),
        ("SPY  - leveraged 1.25/1.5x",    "SPY", strat(0.60, 1.25, 1.50)),
        ("QQQ  - modest 1.0/1.25x",        "QQQ", strat(0.60, 1.00, 1.25)),
        ("TQQQ - 3x ETF, no extra lev",    "TQQQ", strat(0.50, 1.00, 1.00)),
    ]

    # SPY buy-and-hold over the common window = the benchmark to beat.
    spy_prices = get_history("SPY", lookback)
    spy_bt = WalkForwardBacktester(hmm_cfg, candidates[0][2], bt_cfg).run(spy_prices)
    spy_bh = spy_bt.benchmarks["buy_hold"].total_return
    print(f"\nBenchmark to beat: SPY buy & hold = {spy_bh:+.1%} over the test window\n")

    header = f"{'profile':32s} {'return':>8s} {'vs SPY':>8s} {'maxDD':>7s} {'sharpe':>7s}"
    print(header)
    print("-" * len(header))

    rows = []
    for label, ticker, strategy in candidates:
        try:
            prices = get_history(ticker, lookback)
            res = WalkForwardBacktester(hmm_cfg, strategy, bt_cfg).run(prices)
            m = res.metrics
            rows.append((label, m.total_return, m.total_return - spy_bh, m.max_drawdown, m.sharpe))
        except Exception as exc:
            print(f"{label:32s}  (skipped: {exc})")

    for label, ret, excess, dd, sharpe in sorted(rows, key=lambda r: r[2], reverse=True):
        flag = "  <- beats SPY" if excess > 0 else ""
        print(f"{label:32s} {ret:+7.1%} {excess:+7.1%} {dd:7.1%} {sharpe:7.2f}{flag}")

    print("\nNote: 'vs SPY' is total-return excess over buy-and-hold SPY for the SAME window.")
    print("Backtest excludes live circuit breakers; real TQQQ would be halted at the -10% breaker.")


if __name__ == "__main__":
    main()
