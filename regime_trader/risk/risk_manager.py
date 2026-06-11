"""The safety net.

This layer is deliberately dumb and hard-coded. It runs INDEPENDENTLY of the
HMM and the strategy and holds absolute veto power. A mediocre strategy with a
good risk layer bleeds slowly; a good strategy with a bad risk layer can blow up
the account. So the risk manager always gets the last word.

Circuit breakers (drawdown measured from the running equity peak / period open):
    daily   -2%  -> halve all new position sizes
    daily   -3%  -> flatten everything, no new entries today
    weekly  -5%  -> halve all new position sizes
    peak   -10%  -> HARD STOP: write a block file that a human must delete

The 10% block file is intentional friction: the bot stops, writes a file
explaining why, and refuses to trade until you read it and delete it by hand.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path

from regime_trader.core.settings import PROJECT_ROOT

logger = logging.getLogger("regime.risk")


@dataclass
class RiskDecision:
    """Outcome of evaluating the account against every breaker."""

    allow_new_entries: bool
    flatten_all: bool
    size_multiplier: float          # scales every proposed position size
    halted: bool                    # hard stop engaged (block file present)
    reasons: list[str] = field(default_factory=list)

    @property
    def tradable(self) -> bool:
        return self.allow_new_entries and not self.halted and not self.flatten_all


class RiskManager:
    def __init__(self, config: dict | None = None, block_file: str | Path | None = None):
        cfg = config or {}
        self.max_risk_per_trade = float(cfg.get("max_risk_per_trade", 0.01))
        self.max_leverage = float(cfg.get("max_leverage", 1.25))
        self.max_position_weight = float(cfg.get("max_position_weight", 1.0))
        self.max_correlation = float(cfg.get("max_correlation", 0.80))

        self.daily_loss_halve = float(cfg.get("daily_loss_halve", 0.02))
        self.daily_loss_flatten = float(cfg.get("daily_loss_flatten", 0.03))
        self.weekly_loss_halve = float(cfg.get("weekly_loss_halve", 0.05))
        self.max_drawdown_stop = float(cfg.get("max_drawdown_stop", 0.10))

        name = cfg.get("block_file", block_file or "TRADING_BLOCKED")
        self.block_path = Path(name)
        if not self.block_path.is_absolute():
            self.block_path = PROJECT_ROOT / self.block_path

        # Equity reference marks. In production these are persisted; here they are
        # initialised on first evaluation.
        self.equity_peak: float | None = None
        self.day_open_equity: float | None = None
        self.week_open_equity: float | None = None
        self._marks_date: date | None = None
        self._marks_week: tuple[int, int] | None = None

    # ------------------------------------------------------- reference marks
    def _roll_marks(self, equity: float, now: datetime) -> None:
        today = now.date()
        iso_week = now.isocalendar()[:2]
        if self._marks_date != today:
            self.day_open_equity = equity
            self._marks_date = today
        if self._marks_week != iso_week:
            self.week_open_equity = equity
            self._marks_week = iso_week
        if self.equity_peak is None or equity > self.equity_peak:
            self.equity_peak = equity

    # ---------------------------------------------------------- block file
    def is_blocked(self) -> bool:
        return self.block_path.exists()

    def _write_block_file(self, equity: float, drawdown: float) -> None:
        if self.block_path.exists():
            return
        self.block_path.write_text(
            "TRADING HALTED BY RISK MANAGER\n"
            "================================\n"
            f"Time:           {datetime.now(timezone.utc).isoformat()}\n"
            f"Equity:         {equity:,.2f}\n"
            f"Peak equity:    {self.equity_peak:,.2f}\n"
            f"Drawdown:       {drawdown:.2%} (limit {self.max_drawdown_stop:.2%})\n\n"
            "The maximum drawdown circuit breaker tripped. The bot will not place\n"
            "any trades while this file exists.\n\n"
            "ACTION REQUIRED:\n"
            "  1. Review what happened (logs/events.jsonl, the dashboard signal feed).\n"
            "  2. Decide whether your strategy is still valid.\n"
            "  3. Delete this file by hand to resume trading.\n",
            encoding="utf-8",
        )
        logger.critical("Max drawdown hit (%.2f%%). Wrote block file: %s", drawdown * 100, self.block_path)

    # ------------------------------------------------------------- evaluate
    def evaluate(self, equity: float, now: datetime | None = None) -> RiskDecision:
        """Run every circuit breaker against current account equity."""
        now = now or datetime.now(timezone.utc)
        self._roll_marks(equity, now)

        reasons: list[str] = []
        size_mult = 1.0
        allow_new = True
        flatten = False

        # Hard stop is sticky: once the block file exists, nothing trades.
        if self.is_blocked():
            return RiskDecision(False, False, 0.0, True, ["halted: block file present"])

        peak_dd = (self.equity_peak - equity) / self.equity_peak if self.equity_peak else 0.0
        if peak_dd >= self.max_drawdown_stop:
            self._write_block_file(equity, peak_dd)
            return RiskDecision(False, True, 0.0, True,
                                [f"max drawdown {peak_dd:.2%} -> HARD STOP"])

        day_dd = (self.day_open_equity - equity) / self.day_open_equity if self.day_open_equity else 0.0
        week_dd = (self.week_open_equity - equity) / self.week_open_equity if self.week_open_equity else 0.0

        if day_dd >= self.daily_loss_flatten:
            allow_new = False
            flatten = True
            size_mult = 0.0
            reasons.append(f"daily loss {day_dd:.2%} -> flatten")
        elif day_dd >= self.daily_loss_halve:
            size_mult = min(size_mult, 0.5)
            reasons.append(f"daily loss {day_dd:.2%} -> half size")

        if week_dd >= self.weekly_loss_halve:
            size_mult = min(size_mult, 0.5)
            reasons.append(f"weekly loss {week_dd:.2%} -> half size")

        if not reasons:
            reasons.append("ok")
        return RiskDecision(allow_new, flatten, size_mult, False, reasons)

    # -------------------------------------------------------- position sizing
    def size_position(
        self,
        equity: float,
        price: float,
        stop_price: float,
        size_multiplier: float = 1.0,
        target_weight: float | None = None,
    ) -> int:
        """Shares to trade so the loss to `stop_price` is <= max_risk_per_trade.

        The result is additionally capped by the target weight (if given) and the
        max single-name weight, then scaled by the breaker `size_multiplier`.
        """
        if price <= 0 or equity <= 0:
            return 0
        risk_per_share = abs(price - stop_price)
        if risk_per_share <= 0:
            return 0

        dollar_risk = equity * self.max_risk_per_trade
        shares_by_risk = dollar_risk / risk_per_share

        # Weight caps
        weight_cap = self.max_position_weight
        if target_weight is not None:
            weight_cap = min(weight_cap, abs(target_weight))
        shares_by_weight = (equity * weight_cap) / price

        shares = min(shares_by_risk, shares_by_weight) * size_multiplier
        return int(max(0, shares))

    def within_leverage(self, equity: float, gross_exposure_value: float) -> bool:
        if equity <= 0:
            return False
        return (gross_exposure_value / equity) <= self.max_leverage + 1e-9

    # --------------------------------------------------------- correlation
    def correlation_ok(self, new_symbol: str, open_symbols, corr_matrix) -> bool:
        """Reject a new position too correlated with something already held.

        `corr_matrix` is a pandas DataFrame of pairwise correlations (or None to
        skip the check, e.g. when only one name is traded).
        """
        if corr_matrix is None or not open_symbols:
            return True
        for sym in open_symbols:
            if sym == new_symbol:
                continue
            try:
                rho = abs(float(corr_matrix.loc[new_symbol, sym]))
            except (KeyError, ValueError):
                continue
            if rho > self.max_correlation:
                logger.info("Blocking %s: corr %.2f with held %s", new_symbol, rho, sym)
                return False
        return True
