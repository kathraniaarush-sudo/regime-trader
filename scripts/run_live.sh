#!/usr/bin/env bash
# Always-on launcher for the paper trading loop (invoked by launchd).
# Resolves its own project root so it works regardless of where it's called from.
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR"
exec .venv/bin/python -m regime_trader.main portfolio
