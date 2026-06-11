"""Walk-forward backtesting.

This is NOT an in-sample backtest. A naive backtest fits the model on all the
data, finds the settings that look best in hindsight, then pretends it predicted
the future. Here we slide a window: fit the HMM on an in-sample block, then run
it *blind* on the out-of-sample block that follows, and only stitch the
out-of-sample pieces together into the reported equity curve. The model never
sees the data it is judged on.

Causality guarantees:
  * the HMM is fitted only on in-sample bars;
  * regimes in the out-of-sample block come from the forward (filtering) pass, so
    bar t uses bars 0..t only;
  * the weight decided at the close of bar t is applied to the return of t+1.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from regime_trader.brain.hmm_engine import RegimeDetector, RegimeState
from regime_trader.core.features import build_features
from regime_trader.strategy.allocation import RegimeAllocator
from regime_trader.backtest.performance import compute_metrics, metrics_by_regime, PerformanceMetrics

logger = logging.getLogger("regime.backtest")


@dataclass
class BacktestResult:
    equity: pd.Series
    daily: pd.DataFrame                       # weight, asset_ret, strat_ret, regime, confidence
    metrics: PerformanceMetrics
    regime_breakdown: pd.DataFrame
    n_windows: int
    benchmarks: dict[str, PerformanceMetrics] = field(default_factory=dict)


class WalkForwardBacktester:
    def __init__(self, hmm_cfg: dict, strat_cfg: dict, bt_cfg: dict):
        self.hmm_cfg = hmm_cfg
        self.strat_cfg = strat_cfg
        self.in_sample = int(bt_cfg.get("in_sample_days", 252))
        self.out_sample = int(bt_cfg.get("out_sample_days", 126))
        self.step = int(bt_cfg.get("step_days", 126))
        self.slippage = float(bt_cfg.get("slippage_bps", 5)) / 1e4
        self.commission = float(bt_cfg.get("commission_bps", 0)) / 1e4
        self.initial_equity = float(bt_cfg.get("initial_equity", 100000))

    def _new_detector(self) -> RegimeDetector:
        c = self.hmm_cfg
        return RegimeDetector(
            min_regimes=c.get("min_regimes", 3),
            max_regimes=c.get("max_regimes", 7),
            covariance_type=c.get("covariance_type", "diag"),
            n_iter=c.get("n_iter", 200),
            random_state=c.get("random_state", 42),
            min_persistence_bars=c.get("min_persistence_bars", 3),
            max_flips_in_window=c.get("max_flips_in_window", 4),
            flip_window_bars=c.get("flip_window_bars", 20),
        )

    def run(self, prices: pd.DataFrame) -> BacktestResult:
        features = build_features(prices)
        close = prices["close"].reindex(features.index)
        asset_ret = close.pct_change().fillna(0.0)
        allocator = RegimeAllocator(self.strat_cfg)

        n = len(features)
        if n < self.in_sample + self.out_sample:
            raise ValueError(
                f"Not enough data: have {n} bars, need >= {self.in_sample + self.out_sample}"
            )

        weights = pd.Series(0.0, index=features.index)
        regimes = pd.Series(index=features.index, dtype=object)
        confidences = pd.Series(0.0, index=features.index)

        windows = 0
        start = 0
        while start + self.in_sample + self.out_sample <= n:
            is_end = start + self.in_sample
            oos_end = min(is_end + self.out_sample, n)

            detector = self._new_detector()
            try:
                detector.fit(features.iloc[start:is_end])
            except Exception as exc:
                logger.warning("Window %d: HMM fit failed (%s); flat weights", windows, exc)
                start += self.step
                windows += 1
                continue

            # Causal regimes: pass history up to each oos bar, keep the oos slice.
            series = detector.detect_series(features.iloc[start:oos_end])
            oos_idx = features.index[is_end:oos_end]
            for ts in oos_idx:
                row = series.loc[ts]
                state = RegimeState(
                    label=row["label"], canonical=row["canonical"],
                    confidence=float(row["confidence"]), state=int(row["state"]),
                    raw_label=row["raw_label"], uncertain=bool(row["uncertain"]),
                )
                sig = allocator.allocate(state)
                weights.loc[ts] = sig.target_weight
                regimes.loc[ts] = state.canonical
                confidences.loc[ts] = state.confidence

            windows += 1
            start += self.step

        # Apply yesterday's weight to today's return; charge cost on weight changes.
        applied_w = weights.shift(1).fillna(0.0)
        turnover = weights.diff().abs().fillna(0.0)
        cost = turnover * (self.slippage + self.commission)
        strat_ret = applied_w * asset_ret - cost

        traded = strat_ret.loc[regimes.dropna().index]
        equity = self.initial_equity * (1 + traded).cumprod()

        daily = pd.DataFrame(
            {
                "weight": weights,
                "asset_ret": asset_ret,
                "strat_ret": strat_ret,
                "regime": regimes,
                "confidence": confidences,
            }
        ).loc[traded.index]

        metrics = compute_metrics(equity)
        breakdown = metrics_by_regime(traded, regimes.loc[traded.index])
        benchmarks = self._benchmarks(close.loc[traded.index], asset_ret.loc[traded.index])

        logger.info(
            "Backtest: %d windows, return=%.2f%%, sharpe=%.2f, maxDD=%.2f%%",
            windows, metrics.total_return * 100, metrics.sharpe, metrics.max_drawdown * 100,
        )
        return BacktestResult(equity, daily, metrics, breakdown, windows, benchmarks)

    # --------------------------------------------------------- benchmarks
    def _benchmarks(self, close: pd.Series, asset_ret: pd.Series) -> dict[str, PerformanceMetrics]:
        out: dict[str, PerformanceMetrics] = {}

        # Buy & hold
        bh = self.initial_equity * (1 + asset_ret).cumprod()
        out["buy_hold"] = compute_metrics(bh)

        # 200-day SMA trend following: long when close > SMA200, else flat.
        sma = close.rolling(200, min_periods=1).mean()
        sig = (close > sma).astype(float).shift(1).fillna(0.0)
        sma_ret = sig * asset_ret
        out["sma200_trend"] = compute_metrics(self.initial_equity * (1 + sma_ret).cumprod())

        # Random allocation with the same exposure budget (sanity floor).
        rng = np.random.default_rng(0)
        rand_w = pd.Series(rng.uniform(0, 0.95, len(asset_ret)), index=asset_ret.index).shift(1).fillna(0.0)
        rand_ret = rand_w * asset_ret
        out["random"] = compute_metrics(self.initial_equity * (1 + rand_ret).cumprod())
        return out


def stress_test(prices: pd.DataFrame, backtester: WalkForwardBacktester,
                n_crashes: int = 3, crash_pct: float = 0.12, seed: int = 7) -> PerformanceMetrics:
    """Re-run the backtest with synthetic one-day crashes injected.

    Bakes a handful of -10%..-15% single-day shocks into the price path to check
    the strategy (and, in live trading, the risk layer) survives tail events.
    """
    rng = np.random.default_rng(seed)
    shocked = prices.copy()
    idx = rng.choice(np.arange(50, len(shocked) - 5), size=min(n_crashes, len(shocked) - 55), replace=False)
    for i in sorted(idx):
        factor = 1 - crash_pct
        shocked.iloc[i:, shocked.columns.get_loc("close")] *= factor
        if "low" in shocked.columns:
            shocked.iloc[i, shocked.columns.get_loc("low")] *= factor
    result = backtester.run(shocked)
    return result.metrics
