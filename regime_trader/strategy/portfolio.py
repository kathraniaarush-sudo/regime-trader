"""Portfolio construction: how much of the momentum basket to hold.

Two layers, applied in order:
  1. base weights   -> momentum selection + regime overlay. Each selected name
                       gets gross/N, where gross comes from the regime (cash in
                       bear/crash). This is what to hold before risk scaling.
  2. vol targeting  -> scale the whole basket by target_vol / realised_vol, where
                       realised_vol is the STRATEGY's own recent return vol (not
                       the raw basket's). This adapts: calm stretches lever up
                       toward the cap, turbulent stretches scale down, keeping
                       drawdown inside the risk manager's circuit breaker.

Targeting the strategy's own realised vol (rather than the raw basket vol) is the
validated approach: it lets the book lever up when the strategy itself has been
calm, instead of being permanently de-levered by the basket's high gross vol.

The backtester applies layer 2 vectorised across the whole return series; the
live loop calls target_weights() with its own trailing return history.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from regime_trader.brain.hmm_engine import canonical

ANNUALISER = np.sqrt(252)

_DEFAULT_GROSS = {"crash": 0.0, "bear": 0.0, "neutral": 0.6, "bull": 1.0, "euphoria": 1.0}


def realised_vol(returns: pd.Series, lookback: int) -> float:
    """Annualised realised vol from the most recent `lookback` returns."""
    r = pd.Series(returns).dropna().tail(lookback)
    if len(r) < 2:
        return 0.0
    v = r.std()
    return float(v * ANNUALISER) if np.isfinite(v) else 0.0


def vol_target_scalar(realised: float, target_vol: float, max_leverage: float) -> float:
    """target/realised, clamped to [0, max_leverage]. 0 if vol is unusable."""
    if realised <= 0:
        return 0.0
    return float(min(target_vol / realised, max_leverage))


@dataclass
class PortfolioConstructor:
    top_n: int = 10
    regime_gross: dict | None = None
    target_vol: float = 0.09
    vol_lookback: int = 20
    max_leverage: float = 1.5

    def __post_init__(self):
        self.regime_gross = {**_DEFAULT_GROSS, **(self.regime_gross or {})}

    def gross_for(self, regime: str) -> float:
        return float(self.regime_gross.get(canonical(regime), 0.5))

    def base_weights(self, selected: list[str], regime: str) -> dict[str, float]:
        """Regime-overlaid equal weights, before vol targeting. {} = all cash."""
        gross = self.gross_for(regime)
        if gross <= 0 or not selected:
            return {}
        w = gross / len(selected)
        return {ticker: w for ticker in selected}

    def vol_scalar(self, strategy_returns: pd.Series) -> float:
        """Vol-target multiplier from the strategy's own recent returns."""
        return vol_target_scalar(
            realised_vol(strategy_returns, self.vol_lookback), self.target_vol, self.max_leverage
        )

    def target_weights(self, selected: list[str], regime: str,
                       strategy_returns: pd.Series) -> dict[str, float]:
        """Final weights for the live loop: base weights scaled by the vol target.

        `strategy_returns` is the bot's own recent daily return series. Total gross
        is capped at max_leverage.
        """
        base = self.base_weights(selected, regime)
        if not base:
            return {}
        scaled = {t: w * self.vol_scalar(strategy_returns) for t, w in base.items()}
        gross = sum(scaled.values())
        if gross > self.max_leverage:
            f = self.max_leverage / gross
            scaled = {t: w * f for t, w in scaled.items()}
        return {t: w for t, w in scaled.items() if w > 0}
