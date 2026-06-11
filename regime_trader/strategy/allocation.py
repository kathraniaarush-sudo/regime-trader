"""Allocation: turn a detected regime into a target portfolio exposure.

This is the layer you customise to your own edge. The default maps each of the
five canonical regimes to a target gross exposure and leverage:

    crash    -> flat (capital preservation)
    bear     -> small long
    neutral  -> half invested
    bull     -> nearly fully invested
    euphoria -> fully invested, modest leverage

On top of the regime map we optionally scale exposure by the HMM's confidence
and shrink it when the classifier is flickering (uncertain). The risk manager
still holds veto power over whatever this proposes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from regime_trader.brain.hmm_engine import RegimeState, canonical

logger = logging.getLogger("regime.strategy")

_DEFAULT_REGIMES = {
    "crash": {"gross_exposure": 0.00, "leverage": 1.00, "direction": "flat"},
    "bear": {"gross_exposure": 0.25, "leverage": 1.00, "direction": "long"},
    "neutral": {"gross_exposure": 0.50, "leverage": 1.00, "direction": "long"},
    "bull": {"gross_exposure": 0.95, "leverage": 1.00, "direction": "long"},
    "euphoria": {"gross_exposure": 0.95, "leverage": 1.25, "direction": "long"},
}


@dataclass
class AllocationSignal:
    """A target the executor / risk manager can act on."""

    regime: str
    target_weight: float       # signed target weight of equity (+long / -short)
    leverage: float
    confidence: float
    uncertain: bool
    reason: str

    @property
    def direction(self) -> str:
        if self.target_weight > 0:
            return "long"
        if self.target_weight < 0:
            return "short"
        return "flat"


class RegimeAllocator:
    def __init__(self, config: dict | None = None):
        cfg = config or {}
        self.regimes = {**_DEFAULT_REGIMES, **(cfg.get("regimes") or {})}
        self.rebalance_threshold = float(cfg.get("rebalance_threshold", 0.05))
        self.confidence_scaling = bool(cfg.get("confidence_scaling", True))
        self.min_confidence = float(cfg.get("min_confidence", 0.50))

    def _regime_cfg(self, regime: str) -> dict:
        return self.regimes.get(canonical(regime), self.regimes["neutral"])

    def allocate(self, state: RegimeState) -> AllocationSignal:
        """Compute the target weight for a detected regime."""
        regime = state.canonical
        cfg = self._regime_cfg(regime)
        base = float(cfg["gross_exposure"])
        leverage = float(cfg["leverage"])
        direction = cfg.get("direction", "long")

        sign = -1.0 if direction == "short" else (0.0 if direction == "flat" else 1.0)
        weight = base * sign

        reason_bits = [f"regime={regime}", f"base={base:.2f}"]

        # Below the confidence floor we de-risk toward flat.
        if state.confidence < self.min_confidence:
            weight *= state.confidence / max(self.min_confidence, 1e-9)
            reason_bits.append(f"low_conf({state.confidence:.2f})")
        elif self.confidence_scaling:
            weight *= state.confidence
            reason_bits.append(f"conf_scaled({state.confidence:.2f})")

        # A flickering classifier means we are uncertain — halve the bet.
        if state.uncertain:
            weight *= 0.5
            reason_bits.append("uncertain_halved")

        return AllocationSignal(
            regime=regime,
            target_weight=round(weight, 4),
            leverage=leverage,
            confidence=state.confidence,
            uncertain=state.uncertain,
            reason=", ".join(reason_bits),
        )

    def needs_rebalance(self, current_weight: float, signal: AllocationSignal) -> bool:
        """Only trade when the target drifts beyond the rebalance threshold."""
        return abs(signal.target_weight - current_weight) >= self.rebalance_threshold
