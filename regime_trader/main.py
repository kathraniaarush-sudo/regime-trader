"""Orchestration: wire every component together and run the trading loop.

Startup sequence:
  1. load config + credentials
  2. connect to Alpaca, verify the account
  3. fit the HMM on recent history
  4. initialise risk manager, allocator, executor, position tracker

Each iteration of the loop (one per bar):
  * pull account equity and check every circuit breaker FIRST
  * if halted/flatten -> close out and stop trading
  * detect the current regime (forward-filtered, causal)
  * ask the allocator for a target weight
  * size the order under the risk budget and submit it

Commands:
  python -m regime_trader.main status     # one-shot health + regime read
  python -m regime_trader.main backtest    # walk-forward backtest on the anchor
  python -m regime_trader.main run         # live loop (paper by default)
  python -m regime_trader.main run --once  # single iteration then exit
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone

from regime_trader.brain.hmm_engine import RegimeDetector
from regime_trader.broker.alpaca_client import AlpacaClient
from regime_trader.broker.market_data import MarketData, get_history
from regime_trader.broker.order_executor import OrderExecutor
from regime_trader.broker.position_tracker import PositionTracker
from regime_trader.core.features import build_features
from regime_trader.core.logging_config import setup_logging, log_event
from regime_trader.core.settings import load_settings, load_credentials
from regime_trader.monitor.alerts import AlertManager
from regime_trader.risk.risk_manager import RiskManager
from regime_trader.strategy.allocation import RegimeAllocator

logger = logging.getLogger("regime.main")


class TradingBot:
    def __init__(self, settings=None):
        self.settings = settings or load_settings()
        s = self.settings
        setup_logging(s.get("monitoring.log_dir", "logs"))

        self.creds = load_credentials()
        self.client = AlpacaClient(self.creds)
        self.executor = OrderExecutor(self.client)
        self.positions = PositionTracker(self.client)
        self.market = MarketData(self.client)

        self.detector = RegimeDetector(
            min_regimes=s.get("hmm.min_regimes", 3),
            max_regimes=s.get("hmm.max_regimes", 7),
            covariance_type=s.get("hmm.covariance_type", "diag"),
            n_iter=s.get("hmm.n_iter", 200),
            random_state=s.get("hmm.random_state", 42),
            min_persistence_bars=s.get("hmm.min_persistence_bars", 3),
            max_flips_in_window=s.get("hmm.max_flips_in_window", 4),
            flip_window_bars=s.get("hmm.flip_window_bars", 20),
        )
        self.allocator = RegimeAllocator(s.get("strategy", {}))
        self.risk = RiskManager(s.get("risk", {}))
        self.alerts = AlertManager(s.get("monitoring.alerts", {}))

        self.anchor = s.get("universe.regime_anchor", "SPY")
        self.train_lookback = s.get("hmm.train_lookback_days", 504)
        self._last_regime: str | None = None

    # --------------------------------------------------------------- setup
    def connect(self) -> bool:
        if not self.client.is_configured:
            logger.error("No Alpaca credentials — copy .env.example to .env and fill it in.")
            return False
        return self.client.ping()

    def train(self) -> None:
        logger.info("Fitting HMM on %d days of %s history...", self.train_lookback, self.anchor)
        prices = get_history(self.anchor, self.train_lookback)
        self._train_prices = prices
        self.detector.fit(build_features(prices))

    def current_regime(self):
        prices = get_history(self.anchor, self.train_lookback)
        return self.detector.detect(build_features(prices))

    # ---------------------------------------------------------------- loop
    def step(self) -> None:
        """One iteration: breakers -> regime -> allocate -> size -> trade."""
        account = self.client.get_account()
        equity = account.equity
        now = datetime.now(timezone.utc)

        decision = self.risk.evaluate(equity, now)
        log_event(logger, logging.INFO, f"Risk: {', '.join(decision.reasons)}",
                  event="risk_check", equity=equity, size_mult=decision.size_multiplier,
                  tradable=decision.tradable)

        if decision.halted:
            self.alerts.critical("Trading halted (block file present). Manual reset required.")
            return
        if decision.flatten_all:
            self.alerts.critical(f"Daily flatten breaker hit at equity {equity:,.0f}.")
            self.executor.cancel_all()
            self.executor.close_all_positions()
            return

        state = self.current_regime()
        if state.canonical != self._last_regime:
            log_event(logger, logging.WARNING, f"Regime change -> {state.label}",
                      event="regime_change", regime=state.canonical,
                      confidence=state.confidence, uncertain=state.uncertain)
            self.alerts.info(f"Regime is now {state.label} ({state.confidence:.0%} conf)")
            self._last_regime = state.canonical

        signal = self.allocator.allocate(state)
        current_w = self.positions.weights(equity).get(self.anchor, 0.0)
        if not self.allocator.needs_rebalance(current_w, signal):
            logger.info("No rebalance: target %.2f vs current %.2f", signal.target_weight, current_w)
            return

        if not self.client.is_market_open():
            logger.info("Market closed — skipping order submission.")
            return

        price = self.market.latest_price(self.anchor)
        # Protective stop derived from recent volatility (2x daily vol below entry).
        stop = price * (1 - 0.04)
        qty = self.risk.size_position(
            equity, price, stop,
            size_multiplier=decision.size_multiplier,
            target_weight=signal.target_weight,
        )
        if qty <= 0:
            logger.info("Sized to 0 shares (risk/exposure caps).")
            return

        side = "buy" if signal.target_weight > current_w else "sell"
        order = self.executor.submit_bracket_order(self.anchor, qty, side, stop_price=stop)
        log_event(logger, logging.INFO, f"Order {side} {qty} {self.anchor}",
                  event="order", symbol=self.anchor, side=side, qty=qty,
                  price=price, stop=stop, regime=state.canonical, order_id=order.id)

    def run(self, once: bool = False) -> None:
        if not self.connect():
            return
        self.train()
        poll = self.settings.get("execution.poll_seconds", 60)
        logger.info("Entering trading loop (poll=%ss, paper=%s). Ctrl-C to stop.",
                    poll, self.creds.paper)
        while True:
            try:
                self.step()
            except Exception as exc:  # keep the loop alive through transient errors
                logger.exception("Loop iteration failed: %s", exc)
                self.alerts.warning(f"Loop error: {exc}")
            if once:
                break
            time.sleep(poll)


# ------------------------------------------------------------------- CLI
def _cmd_status(bot: TradingBot) -> None:
    if not bot.connect():
        return
    bot.train()
    state = bot.current_regime()
    signal = bot.allocator.allocate(state)
    acct = bot.client.get_account()
    print("\n=== Regime Trader status ===")
    print(f"Account     : {acct.status}  equity={acct.equity:,.2f}  buying_power={acct.buying_power:,.2f}")
    print(f"Regime      : {state.label} (canonical={state.canonical})  conf={state.confidence:.0%}"
          f"{'  [UNCERTAIN]' if state.uncertain else ''}")
    print(f"Posteriors  : " + ", ".join(f"{k}={v:.0%}" for k, v in state.posteriors.items()))
    print(f"Target wt   : {signal.target_weight:+.2f}  ({signal.reason})")
    print(f"Risk blocked: {bot.risk.is_blocked()}")


def _cmd_backtest(bot: TradingBot) -> None:
    from regime_trader.backtest.backtester import WalkForwardBacktester, stress_test
    s = bot.settings
    prices = get_history(bot.anchor, max(bot.train_lookback, 1200))
    bt = WalkForwardBacktester(s.get("hmm", {}), s.get("strategy", {}), s.get("backtest", {}))
    res = bt.run(prices)
    m = res.metrics
    print(f"\n=== Walk-forward backtest: {bot.anchor} ({res.n_windows} windows) ===")
    print(f"Total return : {m.total_return:+.2%}")
    print(f"CAGR         : {m.cagr:+.2%}")
    print(f"Sharpe       : {m.sharpe:.2f}   Sortino: {m.sortino:.2f}")
    print(f"Max drawdown : {m.max_drawdown:.2%}   Calmar: {m.calmar:.2f}")
    print(f"Win rate     : {m.win_rate:.1%}   ({m.n_periods} days)")
    print("\nBenchmarks (total return / sharpe):")
    for name, bm in res.benchmarks.items():
        print(f"  {name:14s}: {bm.total_return:+.2%} / {bm.sharpe:.2f}")
    print("\nBy regime:")
    print(res.regime_breakdown.round(3).to_string() if not res.regime_breakdown.empty else "  (none)")
    stressed = stress_test(prices, bt)
    print(f"\nStress test (crashes injected): return={stressed.total_return:+.2%} "
          f"maxDD={stressed.max_drawdown:.2%}")


def _cmd_portfolio_backtest() -> None:
    from regime_trader.backtest.portfolio_backtester import PortfolioBacktester
    from regime_trader.broker.market_data import get_histories
    s = load_settings()
    p = s.get("portfolio", {})
    anchor = s.get("universe.regime_anchor", "SPY")
    universe = list(p.get("universe", []))
    print(f"Fetching {len(universe)} names + {anchor}...")
    closes = get_histories(universe + [anchor], lookback_days=1500).dropna(axis=1)
    bt = PortfolioBacktester(s.get("hmm", {}), p, s.get("backtest", {}))
    res = bt.run(closes, anchor=anchor)
    m, b = res.metrics, res.benchmark
    halt = "  BREACHES -10% breaker" if m.max_drawdown < -0.10 else "  within breaker"
    print(f"\n=== Momentum + HMM + vol-target ({res.n_rebalances} rebalances) ===")
    print(f"Total return : {m.total_return:+.1%}   (SPY {b.total_return:+.1%})")
    print(f"Sharpe       : {m.sharpe:.2f}   (SPY {b.sharpe:.2f})")
    print(f"Max drawdown : {m.max_drawdown:.1%}   (SPY {b.max_drawdown:.1%}){halt}")
    print(f"30-day win rate vs SPY : {res.win_rate_30d:.0%}")
    print("\nSurvivorship caveat applies; treat as plausible, not guaranteed.")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Regime Trader")
    parser.add_argument(
        "command", nargs="?", default="status",
        choices=["status", "backtest", "run", "portfolio", "portfolio-backtest"],
    )
    parser.add_argument("--once", action="store_true", help="run a single loop iteration and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="portfolio: print the rebalance plan without sending orders")
    args = parser.parse_args(argv)

    if args.command == "portfolio-backtest":
        _cmd_portfolio_backtest()
        return
    if args.command == "portfolio":
        from regime_trader.portfolio_trader import PortfolioTrader
        PortfolioTrader().run(once=args.once, dry_run=args.dry_run)
        return

    bot = TradingBot()
    if args.command == "status":
        _cmd_status(bot)
    elif args.command == "backtest":
        _cmd_backtest(bot)
    elif args.command == "run":
        bot.run(once=args.once)


if __name__ == "__main__":
    main()
