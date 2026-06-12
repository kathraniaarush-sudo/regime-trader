"""Live portfolio trader: momentum + regime + vol-target, on Alpaca.

Each rebalance:
  1. read account equity, run the risk circuit breakers FIRST
     (halt -> stop; daily-flatten -> close everything; else get a size multiplier)
  2. fetch universe history, detect the regime (causal HMM on the anchor)
  3. rank momentum, select the top N, compute vol-targeted target weights
  4. diff target vs current positions and submit the rebalancing orders
     (sells first to free buying power, then buys)

Rebalances on the configured cadence (monthly by default); the breakers are
checked every loop iteration so a drawdown can flatten the book intraday. Uses
the SAME strategy objects as the backtester, so live behaviour matches the test.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from regime_trader.brain.hmm_engine import RegimeDetector
from regime_trader.broker.alpaca_client import AlpacaClient
from regime_trader.broker.market_data import get_histories
from regime_trader.broker.order_executor import OrderExecutor
from regime_trader.broker.position_tracker import PositionTracker
from regime_trader.core.features import build_features
from regime_trader.core.logging_config import setup_logging, log_event
from regime_trader.core.settings import load_settings, load_credentials, PROJECT_ROOT
from regime_trader.monitor.alerts import AlertManager
from regime_trader.risk.risk_manager import RiskManager
from regime_trader.strategy.momentum import MomentumRanker
from regime_trader.strategy.portfolio import PortfolioConstructor

logger = logging.getLogger("regime.portfolio")

MIN_ORDER_NOTIONAL = 25.0   # skip dust trades smaller than this


@dataclass
class TargetBasket:
    regime: str
    selected: list[str]
    weights: dict[str, float]    # ticker -> target weight of equity


class PortfolioTrader:
    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        s = self.settings
        setup_logging(s.get("monitoring.log_dir", "logs"))

        p = s.get("portfolio", {})
        self.universe = list(p.get("universe", []))
        self.anchor = s.get("universe.regime_anchor", "SPY")
        self.rebalance_days = int(p.get("rebalance_days", 21))
        self.vol_lookback = int(p.get("vol_lookback", 20))
        self.train_lookback = s.get("hmm.train_lookback_days", 504)

        self.ranker = MomentumRanker(
            lookback=p.get("momentum_lookback", 252),
            skip=p.get("momentum_skip", 21),
            top_n=p.get("top_n", 10),
        )
        self.constructor = PortfolioConstructor(
            top_n=p.get("top_n", 10),
            regime_gross=p.get("regime_gross"),
            target_vol=p.get("target_vol", 0.09),
            vol_lookback=self.vol_lookback,
            max_leverage=p.get("max_leverage", 1.5),
        )
        # This strategy's overlay uses its own regime granularity (see config).
        self._hmm_cfg = dict(s.get("hmm", {}))
        if p.get("regime_max_regimes"):
            self._hmm_cfg["max_regimes"] = p["regime_max_regimes"]

        self.creds = load_credentials()
        self.client = AlpacaClient(self.creds)
        self.executor = OrderExecutor(self.client)
        self.positions = PositionTracker(self.client)
        self.risk = RiskManager(s.get("risk", {}))
        self.alerts = AlertManager(s.get("monitoring.alerts", {}))

        self.state_path = PROJECT_ROOT / "state" / "portfolio_state.json"

    # -------------------------------------------------------------- helpers
    def connect(self) -> bool:
        if not self.client.is_configured:
            logger.error("No Alpaca credentials — copy .env.example to .env and fill it in.")
            return False
        return self.client.ping()

    def _new_detector(self) -> RegimeDetector:
        c = self._hmm_cfg
        return RegimeDetector(
            min_regimes=c.get("min_regimes", 3),
            max_regimes=c.get("max_regimes", 5),
            covariance_type=c.get("covariance_type", "diag"),
            n_iter=c.get("n_iter", 200),
            random_state=c.get("random_state", 42),
            min_persistence_bars=c.get("min_persistence_bars", 3),
            max_flips_in_window=c.get("max_flips_in_window", 4),
            flip_window_bars=c.get("flip_window_bars", 20),
        )

    def compute_target(self, closes: pd.DataFrame) -> TargetBasket:
        """Momentum selection + regime overlay + vol target -> target weights."""
        stocks = [c for c in closes.columns if c != self.anchor]
        selected = self.ranker.select(closes[stocks])

        anchor_prices = closes[[self.anchor]].rename(columns={self.anchor: "close"})
        try:
            det = self._new_detector().fit(build_features(anchor_prices))
            regime = det.detect(build_features(anchor_prices)).canonical
        except Exception as exc:
            logger.warning("regime detection failed (%s); treating as neutral", exc)
            regime = "neutral"

        base = self.constructor.base_weights(selected, regime)
        if not base:
            return TargetBasket(regime, selected, {})

        # Reconstruct the strategy's recent returns (base-weighted basket) so the
        # vol target uses the same basis as the backtest.
        recent = (closes[selected].pct_change().tail(self.vol_lookback) * pd.Series(base)).sum(axis=1)
        weights = self.constructor.target_weights(selected, regime, recent)
        return TargetBasket(regime, selected, weights)

    @staticmethod
    def plan_orders(target: dict[str, float], equity: float, prices,
                    current_shares: dict[str, int]) -> list[tuple[str, str, int]]:
        """Diff target weights against current shares -> (symbol, side, qty).

        Pure function: no broker calls, so it is directly unit-testable.
        """
        orders: list[tuple[str, str, int]] = []
        for sym in set(target) | set(current_shares):
            price = float(prices.get(sym, 0) or 0)
            if price <= 0:
                continue
            target_sh = int((target.get(sym, 0.0) * equity) / price)
            cur_sh = int(current_shares.get(sym, 0))
            delta = target_sh - cur_sh
            if abs(delta) * price < MIN_ORDER_NOTIONAL:
                continue
            orders.append((sym, "buy" if delta > 0 else "sell", abs(delta)))
        # Sells first to free up buying power, then buys.
        orders.sort(key=lambda o: 0 if o[1] == "sell" else 1)
        return orders

    # -------------------------------------------------------------- rebalance
    def rebalance(self, dry_run: bool = False) -> list[tuple[str, str, int]]:
        account = self.client.get_account()
        equity = account.equity
        decision = self.risk.evaluate(equity)
        log_event(logger, logging.INFO, f"Risk: {', '.join(decision.reasons)}",
                  event="risk_check", equity=equity, size_mult=decision.size_multiplier)

        if decision.halted:
            self.alerts.critical("Trading halted (block file). Manual reset required.")
            return []
        if decision.flatten_all:
            self.alerts.critical(f"Daily flatten breaker at equity {equity:,.0f}.")
            if not dry_run:
                self.executor.cancel_all()
                self.executor.close_all_positions()
            return []

        closes = get_histories(self.universe + [self.anchor], self.train_lookback + 60).dropna(axis=1)
        basket = self.compute_target(closes)
        # Circuit-breaker size multiplier scales the whole book.
        weights = {t: w * decision.size_multiplier for t, w in basket.weights.items()}
        prices = closes.iloc[-1]
        current_shares = {p.symbol: int(float(p.qty)) for p in self.positions.list_positions()}
        orders = self.plan_orders(weights, equity, prices, current_shares)

        log_event(logger, logging.INFO,
                  f"Rebalance: regime={basket.regime} names={len(basket.weights)} orders={len(orders)}",
                  event="rebalance", regime=basket.regime,
                  weights={k: round(v, 3) for k, v in weights.items()}, n_orders=len(orders))

        if dry_run:
            self._print_plan(basket, weights, orders, equity)
            return orders

        for sym, side, qty in orders:
            try:
                self.executor.submit_market_order(sym, qty, side)
                log_event(logger, logging.INFO, f"{side} {qty} {sym}",
                          event="order", symbol=sym, side=side, qty=qty, regime=basket.regime)
            except Exception as exc:
                logger.error("order failed %s %s %s: %s", side, qty, sym, exc)
                self.alerts.warning(f"Order failed {side} {qty} {sym}: {exc}")
        self._save_state(basket)
        return orders

    def _print_plan(self, basket, weights, orders, equity):
        print(f"\n=== Rebalance plan (DRY RUN) — equity ${equity:,.0f} ===")
        print(f"Regime: {basket.regime}")
        if not weights:
            print("Target: CASH (regime says flat)")
        else:
            print("Target basket:")
            for t, w in sorted(weights.items(), key=lambda x: -x[1]):
                print(f"  {t:6s} {w:6.1%}  (${w*equity:,.0f})")
        print(f"Orders ({len(orders)}):" if orders else "Orders: none (already aligned)")
        for sym, side, qty in orders:
            print(f"  {side.upper():4s} {qty:>5d} {sym}")

    # -------------------------------------------------------------- schedule
    def _load_state(self) -> dict:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text())
            except json.JSONDecodeError:
                return {}
        return {}

    def _save_state(self, basket: TargetBasket) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps({
            "last_rebalance": datetime.now(timezone.utc).isoformat(),
            "regime": basket.regime,
            "weights": {k: round(v, 4) for k, v in basket.weights.items()},
        }, indent=2))

    def _due_for_rebalance(self, now: datetime) -> bool:
        state = self._load_state()
        last = state.get("last_rebalance")
        if not last:
            return True
        try:
            elapsed = (now - datetime.fromisoformat(last)).days
        except ValueError:
            return True
        return elapsed >= self.rebalance_days

    # -------------------------------------------------------------- loop
    def run(self, once: bool = False, dry_run: bool = False) -> None:
        if not self.connect():
            return
        poll = self.settings.get("execution.poll_seconds", 60)
        logger.info("Portfolio loop started (rebalance every %dd, paper=%s).",
                    self.rebalance_days, self.creds.paper)
        while True:
            try:
                now = datetime.now(timezone.utc)
                equity = self.client.get_account().equity
                decision = self.risk.evaluate(equity, now)
                # React to breakers every tick; only re-rank on the cadence.
                if decision.halted or decision.flatten_all or self._due_for_rebalance(now) or dry_run:
                    self.rebalance(dry_run=dry_run)
                else:
                    logger.info("No rebalance due (equity %.0f). Next in <= %dd.",
                                equity, self.rebalance_days)
            except Exception as exc:
                logger.exception("Portfolio loop iteration failed: %s", exc)
                self.alerts.warning(f"Loop error: {exc}")
            if once:
                break
            time.sleep(poll)
