# Always-on runner (macOS launchd)

Keeps the paper trading loop running unattended — it survives terminal closes,
logouts, and reboots, and restarts itself if it crashes.

## Install / start

```bash
cp deploy/com.regimetrader.bot.plist ~/Library/LaunchAgents/
launchctl load ~/Library/LaunchAgents/com.regimetrader.bot.plist
```

It starts immediately and on every login.

## Check it's running

```bash
launchctl list | grep regimetrader      # shows a PID if alive
tail -f logs/runner.log                  # live output
```

## Stop / uninstall

```bash
launchctl unload ~/Library/LaunchAgents/com.regimetrader.bot.plist
rm ~/Library/LaunchAgents/com.regimetrader.bot.plist
```

## Notes

- The loop polls every `execution.poll_seconds` (default 60s): it checks the risk
  circuit breakers every tick and rebalances on the `portfolio.rebalance_days`
  cadence (monthly by default).
- Paper trading only while `ALPACA_PAPER=true` in `.env`.
- The plist hard-codes the project path (`/Users/aarushkathrani/code/regime-trader`).
  If you move the repo, edit the plist and reload.
- To pause trading entirely, either unload the agent (above) or let the risk
  manager's `TRADING_BLOCKED` kill-switch file stop it.
