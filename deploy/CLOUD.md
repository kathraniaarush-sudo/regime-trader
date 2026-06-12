# Running in the cloud (GitHub Actions)

Trade without keeping your laptop on. A scheduled GitHub Actions workflow
(`.github/workflows/trade.yml`) runs the bot a few times each trading day on
GitHub's servers, for free.

## One-time setup

### 1. Add your Alpaca keys as repository secrets

GitHub → your repo → **Settings** → **Secrets and variables** → **Actions** →
**New repository secret**. Add these four (same values as your local `.env`):

| Name | Value |
|------|-------|
| `ALPACA_API_KEY` | your paper API key id |
| `ALPACA_SECRET_KEY` | your paper secret key |
| `ALPACA_BASE_URL` | `https://paper-api.alpaca.markets` |
| `ALPACA_PAPER` | `true` |

Secrets are encrypted and never printed in logs. They are NOT the same as your
local `.env` (which stays on your machine and is git-ignored).

### 2. Turn it on

The workflow is already scheduled. To test it immediately:
GitHub → **Actions** tab → **Trade (paper)** → **Run workflow**.

Watch the run; the last step's log shows the regime, the rebalance decision, and
any orders. It will also run automatically on the schedule (weekday market hours).

## Important: don't double-run

Once the cloud is trading your paper account, **stop running the bot on your
laptop** — otherwise both would trade the same account. That means:

- do NOT load the launchd runner (`deploy/README.md`), and
- do NOT run `python -m regime_trader.main portfolio` locally.

Local commands you DO still use:
- `python -m regime_trader.main portfolio-backtest` — backtesting
- `streamlit run regime_trader/dashboard/app.py` — the dashboard reads Alpaca
  live, so it shows your positions and the scoreboard no matter where the bot runs

## How it behaves

- Runs `portfolio --once` ~4×/day on weekdays: checks the risk circuit breakers
  every run and rebalances on the monthly cadence.
- State (last rebalance date, challenge start) persists between runs via the
  Actions cache. If a run is ever missed or state is lost, the bot self-heals —
  it reconciles against your actual Alpaca positions, so it won't double-buy.
- GitHub cron is UTC and can be delayed a few minutes; breakers therefore react
  within ~an hour, not instantly. Fine for a paper challenge.

## Watching the dashboard from anywhere (optional)

The dashboard is a local app. To get an always-on URL without your laptop, deploy
it free to **Streamlit Community Cloud** (share.streamlit.io): point it at this
repo and `regime_trader/dashboard/app.py`, and add the same four Alpaca values as
Streamlit secrets. It reads Alpaca live, so it mirrors whatever the cloud bot does.

## Pausing

- Disable the schedule: GitHub → Actions → Trade (paper) → **⋯** → **Disable workflow**.
- Or trip the kill-switch: the risk manager writes `TRADING_BLOCKED` on a 10%
  drawdown and stops until you remove it.
