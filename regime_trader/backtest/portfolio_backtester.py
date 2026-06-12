"""Walk-forward backtest for the momentum + regime + vol-target portfolio.

Uses the SAME strategy objects the live loop uses (MomentumRanker,
PortfolioConstructor, RegimeDetector), so a passing backtest reflects live
behaviour rather than a parallel reimplementation. Fully causal: at each
rebalance the regime, the momentum ranking, and the vol estimate use only bars
up to that day; weights are applied to the next day's return.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_trader.brain.hmm_engine import RegimeDetector
from regime_trader.core.features import build_features
from regime_trader.strategy.momentum import MomentumRanker
from regime_trader.strategy.portfolio import PortfolioConstructor
from regime_trader.backtest.performance import compute_metrics, PerformanceMetrics

logger = logging.getLogger("regime.portfolio_bt")


@dataclass
class PortfolioBacktestResult:
    equity: pd.Series
    returns: pd.Series
    weights: pd.DataFrame
    metrics: PerformanceMetrics
    benchmark: PerformanceMetrics       # anchor (SPY) buy & hold over the same window
    win_rate_30d: float                 # share of rolling 30d windows beating the anchor
    n_rebalances: int


class PortfolioBacktester:
    def __init__(self, hmm_cfg: dict, portfolio_cfg: dict, bt_cfg: dict):
        # This strategy's overlay uses its own regime granularity (see config).
        self.hmm_cfg = dict(hmm_cfg or {})
        if portfolio_cfg.get("regime_max_regimes"):
            self.hmm_cfg["max_regimes"] = portfolio_cfg["regime_max_regimes"]
        if portfolio_cfg.get("regime_min_regimes"):
            self.hmm_cfg["min_regimes"] = portfolio_cfg["regime_min_regimes"]
        self.ranker = MomentumRanker(
            lookback=portfolio_cfg.get("momentum_lookback", 252),
            skip=portfolio_cfg.get("momentum_skip", 21),
            top_n=portfolio_cfg.get("top_n", 10),
        )
        self.constructor = PortfolioConstructor(
            top_n=portfolio_cfg.get("top_n", 10),
            regime_gross=portfolio_cfg.get("regime_gross"),
            target_vol=portfolio_cfg.get("target_vol", 0.09),
            vol_lookback=portfolio_cfg.get("vol_lookback", 20),
            max_leverage=portfolio_cfg.get("max_leverage", 1.5),
        )
        self.rebalance_days = portfolio_cfg.get("rebalance_days", 21)
        self.train_window = self.hmm_cfg.get("train_lookback_days", 504)
        self.slippage = float(bt_cfg.get("slippage_bps", 5)) / 1e4
        self.commission = float(bt_cfg.get("commission_bps", 0)) / 1e4
        self.initial_equity = float(bt_cfg.get("initial_equity", 100000))

    def _new_detector(self) -> RegimeDetector:
        c = self.hmm_cfg
        return RegimeDetector(
            min_regimes=c.get("min_regimes", 3),
            max_regimes=c.get("max_regimes", 7),
            covariance_type=c.get("covariance_type", "diag"),
            n_iter=c.get("n_iter", 150),
            random_state=c.get("random_state", 42),
            min_persistence_bars=c.get("min_persistence_bars", 3),
            max_flips_in_window=c.get("max_flips_in_window", 4),
            flip_window_bars=c.get("flip_window_bars", 20),
        )

    def _regime_at(self, anchor_prices: pd.DataFrame) -> str:
        feats = build_features(anchor_prices)
        try:
            det = self._new_detector().fit(feats)
            return det.detect(feats).canonical
        except Exception as exc:  # degenerate window -> treat as neutral
            logger.debug("regime fit failed: %s", exc)
            return "neutral"

    def run(self, closes: pd.DataFrame, anchor: str = "SPY") -> PortfolioBacktestResult:
        if anchor not in closes.columns:
            raise ValueError(f"anchor {anchor!r} missing from price frame")
        stocks = [c for c in closes.columns if c != anchor]
        rets = closes[stocks].pct_change()
        anchor_prices = closes[[anchor]].rename(columns={anchor: "close"})

        warmup = max(self.ranker.lookback + 1, 60)
        if len(closes) <= warmup + self.rebalance_days:
            raise ValueError("not enough history for a portfolio backtest")

        # Pass 1: base weights (momentum + regime overlay), before vol targeting.
        weights = pd.DataFrame(np.nan, index=closes.index, columns=stocks)
        n_rebal = 0
        for i in range(warmup, len(closes), self.rebalance_days):
            sub = closes.iloc[: i + 1]
            selected = self.ranker.select(sub[stocks])
            regime = self._regime_at(anchor_prices.iloc[max(0, i - self.train_window - 80): i + 1])
            base = self.constructor.base_weights(selected, regime)
            weights.iloc[i] = 0.0
            for ticker, w in base.items():
                weights.iloc[i, weights.columns.get_loc(ticker)] = w
            n_rebal += 1

        weights = weights.ffill().fillna(0.0)
        turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
        raw_ret = (weights.shift(1) * rets).sum(axis=1) - turnover * (self.slippage + self.commission)

        # Pass 2: scale by the vol target, using the STRATEGY's own realised vol
        # (causal: scalar from returns up to t, applied to t+1).
        ann = float(np.sqrt(252))
        realised = raw_ret.rolling(self.constructor.vol_lookback).std() * ann
        scalar = (self.constructor.target_vol / realised.replace(0, np.nan)) \
            .clip(0, self.constructor.max_leverage).fillna(0.0)
        port_ret = (scalar.shift(1) * raw_ret).loc[closes.index[warmup]:]

        equity = self.initial_equity * (1 + port_ret).cumprod()
        metrics = compute_metrics(equity)

        bench_ret = closes[anchor].pct_change().loc[port_ret.index]
        bench = compute_metrics(self.initial_equity * (1 + bench_ret).cumprod())

        win = self._rolling_win_rate(port_ret, bench_ret)
        logger.info("Portfolio backtest: %d rebalances, return=%.1f%%, sharpe=%.2f, maxDD=%.1f%%",
                    n_rebal, metrics.total_return * 100, metrics.sharpe, metrics.max_drawdown * 100)
        return PortfolioBacktestResult(equity, port_ret, weights, metrics, bench, win, n_rebal)

    @staticmethod
    def _rolling_win_rate(strat: pd.Series, bench: pd.Series, window: int = 30) -> float:
        s = (1 + strat).rolling(window).apply(np.prod, raw=True) - 1
        b = (1 + bench).rolling(window).apply(np.prod, raw=True) - 1
        df = pd.DataFrame({"s": s, "b": b}).dropna()
        return float(((df["s"] - df["b"]) > 0).mean()) if len(df) else 0.0
