# Regime Trader

A fully-automated, **regime-aware** trading bot. It detects what kind of market
you are in using Hidden Markov Models, sizes a portfolio allocation to match,
enforces hard risk limits, and (optionally) places real orders through Alpaca —
all driven from a single config file and visualised in a live dashboard.

This is an open-source reimplementation of the system described in the YouTube
video *"How To Actually Build a Trading Bot With Claude Code"*
([y_bsjZThP0o](https://www.youtube.com/watch?v=y_bsjZThP0o)), built from the
transcript and structured into a clean, tested Python package.

> ⚠️ **Not financial advice.** Trading involves real risk of loss. This is an
> educational framework for building a *disciplined, systematic* process. It
> ships configured for **paper trading**. Validate any strategy extensively
> before risking real capital. The authors guarantee nothing.

---

## The five components

The system mirrors the architecture from the video:

| Component | Module | Role |
|-----------|--------|------|
| 🧠 **Brain** | `brain/hmm_engine.py` | HMMs classify the market into volatility regimes (crash / bear / neutral / bull / euphoria). Auto-selects the regime count by BIC; labels states by return; detects causally with a forward-only filter. |
| ⚖️ **Allocation** | `strategy/allocation.py` | Maps each regime to a target exposure and leverage, scaled by confidence. **This is the layer you customise.** |
| 🛡️ **Safety** | `risk/risk_manager.py` | Hard-coded circuit breakers, position sizing, leverage & correlation caps. Runs independently of the model with veto power. |
| 🏦 **Broker** | `broker/` | Alpaca REST wrapper: account, orders, positions, market data. |
| 📊 **Dashboard** | `dashboard/app.py` | Streamlit UI: regime, confidence, P&L, risk status, signal feed. |

Plus a **walk-forward backtester** (`backtest/`) that validates a strategy on
blind out-of-sample data against buy-and-hold, 200-day SMA, and random
benchmarks, with synthetic-crash stress tests.

## Why Hidden Markov Models?

The HMM does **not** predict prices. It infers the hidden *state* of the market
from volatility-flavoured features. Calm uptrends, choppy ranges, and crashes
have distinct statistical signatures; the model learns to separate them so the
allocation layer can act differently in each.

Two correctness details the video stresses, both implemented here:

- **No look-ahead bias.** Live detection uses the **forward (filtering)
  algorithm only** (`_forward_filter`), so the regime at bar *t* depends solely
  on bars `0..t`. We deliberately avoid `hmm.predict`, whose forward-backward
  smoother peeks at future bars. Tests assert this (`test_forward_filter_is_causal`).
- **Stability filter.** A new regime must persist `min_persistence_bars`
  consecutive bars before the system acts, so a single flickering bar can't
  trigger a trade. Excessive flipping flags the market as *uncertain* and shrinks
  position sizes.

## Risk: the circuit breakers

These are hard-coded and independent of the strategy (config in `settings.yaml`):

| Trigger | Action |
|---------|--------|
| Down 2% on the day | Halve all new position sizes |
| Down 3% on the day | Flatten everything, no new entries |
| Down 5% on the week | Halve all new position sizes |
| Down 10% from peak | **Hard stop** — write `TRADING_BLOCKED`; a human must delete it to resume |

Position sizing risks at most 1% of equity per trade by default.

---

## Quick start

```bash
git clone <your-repo-url> regime-trader && cd regime-trader
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Validate everything works (no broker or network needed)
pytest -q

# See the current regime and a backtest using FREE yfinance data
python -m regime_trader.main backtest

# Launch the dashboard (works offline)
streamlit run regime_trader/dashboard/app.py
```

### Connecting Alpaca (paper trading)

1. Create a free account at [alpaca.markets](https://alpaca.markets) and open
   **Paper Trading → API Keys**.
2. Copy `.env.example` to `.env` and fill in your keys:
   ```bash
   cp .env.example .env
   ```
   ```
   ALPACA_API_KEY=...
   ALPACA_SECRET_KEY=...
   ALPACA_BASE_URL=https://paper-api.alpaca.markets
   ALPACA_PAPER=true
   ```
   `.env` is git-ignored — **never commit or share your keys.**
3. Check the connection and current regime:
   ```bash
   python -m regime_trader.main status
   ```
4. Run the live loop (paper):
   ```bash
   python -m regime_trader.main run          # continuous
   python -m regime_trader.main run --once    # single iteration
   ```

## Running the 30-day challenge

The momentum strategy (cross-sectional momentum + HMM regime overlay + vol
targeting) is the "beat the index" mode. Backtest it, run it on paper, and watch
yourself versus the S&P 500.

```bash
# 1. Backtest the strategy (free yfinance data)
python -m regime_trader.main portfolio-backtest

# 2. See exactly what it would trade, without sending orders
python -m regime_trader.main portfolio --once --dry-run

# 3. Start the challenge — first rebalance into the basket (paper account)
python -m regime_trader.main portfolio --once

# 4. Keep it running. Either a terminal:
python -m regime_trader.main portfolio
#    ...or the always-on launchd runner (survives reboots):
cp deploy/com.regimetrader.bot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.regimetrader.bot.plist   # see deploy/README.md

# 5. Watch the scoreboard
streamlit run regime_trader/dashboard/app.py     # "You vs S&P 500" updates daily
```

Tune the strategy in [`config/settings.yaml`](config/settings.yaml) under
`portfolio` (universe, `top_n`, `target_vol`, `regime_gross`). Re-run the
backtest after each change. Honest expectations: over a single 30-day window
beating the index is close to a coin flip; the strategy's durable edge is
*risk-adjusted* return and shallower drawdowns, and it shines in choppy or
falling markets. Not financial advice.

## Customising it to your strategy

Everything tunable lives in [`config/settings.yaml`](config/settings.yaml):

- `universe.tickers` / `regime_anchor` — what you trade and what defines the regime.
- `strategy.regimes` — the exposure/leverage map. **Spend your time here.** Edit
  it, re-run `python -m regime_trader.main backtest`, and iterate until it beats
  the benchmarks on *out-of-sample* data and survives the stress test.
- `risk.*` — circuit-breaker thresholds and sizing limits.
- `hmm.*` — feature window, regime-count search range, stability filter.

The recommended workflow (straight from the video): **paper trade for at least a
month**, review every rebalance, backtest across multiple tickers and periods,
then — only after it has demonstrated an edge live — consider a funded account.

## Project layout

```
regime_trader/
  core/        settings, structured logging, feature engineering
  brain/       HMM regime detection (the "brain")
  strategy/    regime → allocation
  risk/        circuit breakers, position sizing (the "safety net")
  broker/      Alpaca client, order executor, position tracker, market data
  backtest/    walk-forward engine + performance analytics
  monitor/     alerting
  dashboard/   Streamlit UI
  main.py      orchestration + CLI (status / backtest / run)
config/        settings.yaml
tests/         one suite per phase
```

## Tests

`pytest -q` runs the full suite with **no network and no real account** (the
broker is exercised against a fake HTTP session). The tests assert the things
that actually matter for a trading bot: feature causality, no-look-ahead regime
detection, return-ordered labels, the stability filter, every circuit breaker,
risk-budgeted sizing, and walk-forward weight lagging.

## License

MIT — see [LICENSE](LICENSE). Provided for educational purposes, with no warranty
and no guarantee of profit. Use at your own risk.
