"""Prototype: cross-sectional momentum + trend overlay ('dual momentum').

The idea you asked to test: instead of timing the index, hold the strongest
trending stocks WITHIN the S&P, and step aside when the market itself rolls over.

  * Relative momentum (selection): each month, rank the universe by 12-1 momentum
    (trailing 12-month return, skipping the most recent month to dodge short-term
    reversal) and hold the top N equal-weighted.
  * Absolute momentum (overlay): only hold stocks while SPY is above its 200-day
    moving average; otherwise go to cash. This is the 'regime / trend' switch.

Everything is causal: weights decided on day t use only data up to t and are
applied to t+1 returns. Monthly rebalance keeps turnover and overfitting down.

HONEST CAVEAT: the universe below is a FIXED list of today's large-caps, so the
backtest has survivorship bias (these are known winners). Real results would be
lower. This prototype tests whether the IDEA has legs, not a deployable number.

Usage:  python scripts/momentum_prototype.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from regime_trader.backtest.performance import compute_metrics, max_drawdown

# Fixed liquid large-cap universe (survivorship-biased — see caveat above).
UNIVERSE = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "JNJ",
    "WMT", "MA", "PG", "HD", "AVGO", "XOM", "UNH", "KO", "PEP", "COST",
    "ABBV", "ADBE", "CRM", "NFLX", "AMD", "INTC", "CSCO", "MCD", "TMO", "ABT",
    "WFC", "DIS", "BAC", "ORCL", "ACN", "LLY", "CVX", "MRK", "NKE", "QCOM",
]
TOP_N = 10
REBAL = 21          # trading days between rebalances (~monthly)
LOOKBACK = 252      # momentum lookback
SKIP = 21           # skip most recent month
SLIPPAGE = 5 / 1e4
WINDOW_30 = 30


def fetch_closes(tickers, period_days=1500) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(tickers, period=f"{int(period_days*1.5)}d", interval="1d",
                      auto_adjust=True, progress=False)
    # Multi-ticker downloads use a (field, ticker) column MultiIndex with the
    # field capitalised ("Close"); single-ticker downloads are flat.
    if isinstance(raw.columns, pd.MultiIndex):
        df = raw["Close"]
    else:
        df = raw[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(df, pd.Series):
        df = df.to_frame()
    today = pd.Timestamp.now().normalize()
    df = df[df.index.normalize() < today]    # drop the incomplete current bar
    return df.dropna(how="all").tail(period_days)


def run_strategy(closes: pd.DataFrame, use_overlay: bool) -> pd.Series:
    spy = closes["SPY"]
    sma200 = spy.rolling(200, min_periods=50).mean()
    stocks = [c for c in closes.columns if c != "SPY"]
    rets = closes[stocks].pct_change()

    # Weights are NaN except on rebalance days, where we write the FULL target
    # row (zeros for everything not selected). Forward-filling then carries each
    # rebalance's target until the next one, so names that leave the top N are
    # correctly exited rather than held forever.
    weights = pd.DataFrame(np.nan, index=closes.index, columns=stocks)
    for i in range(LOOKBACK + 1, len(closes), REBAL):
        mom = closes[stocks].iloc[i - SKIP] / closes[stocks].iloc[i - LOOKBACK] - 1.0
        top = mom.dropna().nlargest(TOP_N).index
        if len(top) == 0:
            continue
        gross = 1.0
        if use_overlay and not (spy.iloc[i] > sma200.iloc[i]):
            gross = 0.0       # market in downtrend -> cash
        weights.iloc[i] = 0.0
        weights.iloc[i, weights.columns.get_indexer(top)] = gross / len(top)

    weights = weights.ffill().fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    port_ret = (weights.shift(1) * rets).sum(axis=1) - turnover * SLIPPAGE
    return port_ret.loc[closes.index[LOOKBACK + 1]:]


# Regime -> gross exposure for the momentum basket. The basket is the alpha;
# the regime decides how much of it to hold (and when to sit in cash).
GROSS_MAP = {"crash": 0.0, "bear": 0.0, "neutral": 0.6, "bull": 1.0, "euphoria": 1.0}


def run_strategy_hmm(closes: pd.DataFrame, train: int = 504) -> pd.Series:
    """Momentum selection, but the gross exposure overlay is the HMM regime brain.

    At each rebalance we fit the regime detector on a trailing window of SPY
    ending that day and read the current regime via the causal forward filter
    (no look-ahead), then scale the basket by GROSS_MAP[regime].
    """
    from regime_trader.brain.hmm_engine import RegimeDetector
    from regime_trader.core.features import build_features

    spy_prices = closes[["SPY"]].rename(columns={"SPY": "close"})
    stocks = [c for c in closes.columns if c != "SPY"]
    rets = closes[stocks].pct_change()

    weights = pd.DataFrame(np.nan, index=closes.index, columns=stocks)
    for i in range(LOOKBACK + 1, len(closes), REBAL):
        mom = closes[stocks].iloc[i - SKIP] / closes[stocks].iloc[i - LOOKBACK] - 1.0
        top = mom.dropna().nlargest(TOP_N).index
        if len(top) == 0:
            continue
        # Causal regime read: only SPY bars up to and including day i.
        feats = build_features(spy_prices.iloc[max(0, i - train - 80):i + 1])
        try:
            det = RegimeDetector(min_regimes=3, max_regimes=5, n_iter=150).fit(feats)
            gross = GROSS_MAP.get(det.detect(feats).canonical, 0.5)
        except Exception:
            gross = 0.5
        weights.iloc[i] = 0.0
        weights.iloc[i, weights.columns.get_indexer(top)] = gross / len(top)

    weights = weights.ffill().fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    port_ret = (weights.shift(1) * rets).sum(axis=1) - turnover * SLIPPAGE
    return port_ret.loc[closes.index[LOOKBACK + 1]:]


def rolling_cum(returns: pd.Series, window: int) -> pd.Series:
    return (1 + returns).rolling(window).apply(np.prod, raw=True) - 1


def summarize(name: str, port_ret: pd.Series, spy_ret: pd.Series) -> None:
    equity = 100000 * (1 + port_ret).cumprod()
    m = compute_metrics(equity)
    df = pd.DataFrame({"s": rolling_cum(port_ret, WINDOW_30),
                       "b": rolling_cum(spy_ret, WINDOW_30)}).dropna()
    win = ((df["s"] - df["b"]) > 0).mean()
    print(f"{name:34s} {m.total_return:+8.1%} {m.cagr:+7.1%} {m.sharpe:6.2f} "
          f"{m.max_drawdown:7.1%} {win:8.0%}")


def main() -> None:
    print("Downloading universe (this can take ~20s)...")
    closes = fetch_closes(UNIVERSE + ["SPY"])
    closes = closes.dropna(axis=1)        # keep only full-history names
    have = [c for c in closes.columns if c != "SPY"]
    print(f"Universe with full history: {len(have)} names, {len(closes)} bars "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})\n")

    spy_ret = closes["SPY"].pct_change().loc[closes.index[LOOKBACK + 1]:]

    header = f"{'strategy':34s} {'return':>8s} {'cagr':>7s} {'sharpe':>6s} {'maxDD':>7s} {'30d win':>8s}"
    print(header)
    print("-" * len(header))

    summarize("SPY buy & hold", spy_ret, spy_ret)
    summarize("Momentum only (always invested)", run_strategy(closes, use_overlay=False), spy_ret)
    summarize("Momentum + 200-SMA overlay (dumb)", run_strategy(closes, use_overlay=True), spy_ret)
    print("  fitting HMM overlay (refits each rebalance, ~30s)...")
    summarize("Momentum + HMM regime (smart)", run_strategy_hmm(closes), spy_ret)

    print("\n'30d win' = share of rolling 30-day windows that beat SPY.")
    print("Survivorship-biased universe; treat as idea-validation, not a deployable result.")


if __name__ == "__main__":
    main()
