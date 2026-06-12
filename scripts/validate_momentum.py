"""De-bias and stress-test the momentum + HMM-regime strategy.

The prototype looked great, but on a survivorship-biased universe of today's
winners. This script attacks that:

  1. Broader, diversified universe that deliberately INCLUDES laggards
     (Intel, IBM, Pfizer, CVS, GM, Ford, AT&T, Paramount, Kraft...), not just
     the obvious winners.
  2. A head-to-head run that EXCLUDES the 7 mega-cap winners entirely. If the
     edge survives without AAPL/MSFT/NVDA/AMZN/GOOGL/META/TSLA, it is not just
     "you rode NVDA".
  3. Benchmarks against BOTH cap-weighted SPY and equal-weight S&P (RSP). Beating
     RSP is the harder, fairer test, since cap-weighted SPY was itself a mega-cap
     bet this era.
  4. A sub-period breakdown to check the edge is steady, not one lucky stretch.

Residual bias we CANNOT remove with free data: truly delisted/bankrupt names are
absent. For large-cap US over this window that effect is small, but it is not
zero, so treat results as "edge is plausibly real", not "guaranteed".

Usage:  python scripts/validate_momentum.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from regime_trader.backtest.performance import compute_metrics

TOP_N = 10
REBAL = 21
LOOKBACK = 252
SKIP = 21
SLIPPAGE = 5 / 1e4
WINDOW_30 = 30
GROSS_MAP = {"crash": 0.0, "bear": 0.0, "neutral": 0.6, "bull": 1.0, "euphoria": 1.0}

MEGA7 = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA"]
UNIVERSE = MEGA7 + [
    # tech / telecom, winners and laggards
    "ORCL", "CSCO", "INTC", "IBM", "ADBE", "CRM", "AMD", "QCOM", "TXN", "HPQ", "T", "VZ",
    # financials
    "JPM", "BAC", "WFC", "C", "GS", "MS", "V", "MA", "AXP",
    # healthcare, winners and laggards
    "JNJ", "UNH", "PFE", "MRK", "ABBV", "ABT", "TMO", "LLY", "CVS", "BMY",
    # consumer
    "PG", "KO", "PEP", "WMT", "COST", "MCD", "NKE", "SBUX", "TGT", "HD", "LOW", "DIS",
    # industrials / energy / value laggards
    "GE", "CAT", "BA", "HON", "UPS", "MMM", "XOM", "CVX", "SLB", "GM", "F", "KHC", "PARA", "MO",
]


def fetch_closes(tickers, period_days=1500) -> pd.DataFrame:
    import yfinance as yf
    raw = yf.download(tickers, period=f"{int(period_days*1.5)}d", interval="1d",
                      auto_adjust=True, progress=False)
    df = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    today = pd.Timestamp.now().normalize()
    df = df[df.index.normalize() < today]
    return df.dropna(how="all").tail(period_days)


def rebal_positions(n: int):
    return range(LOOKBACK + 1, n, REBAL)


def compute_regime_gross(closes: pd.DataFrame, train: int = 504) -> dict:
    """Causal HMM regime -> gross exposure at each rebalance (computed once)."""
    from regime_trader.brain.hmm_engine import RegimeDetector
    from regime_trader.core.features import build_features
    spy = closes[["SPY"]].rename(columns={"SPY": "close"})
    out = {}
    for i in rebal_positions(len(closes)):
        feats = build_features(spy.iloc[max(0, i - train - 80):i + 1])
        try:
            det = RegimeDetector(min_regimes=3, max_regimes=5, n_iter=150).fit(feats)
            out[i] = GROSS_MAP.get(det.detect(feats).canonical, 0.5)
        except Exception:
            out[i] = 0.5
    return out


def momentum_returns(closes: pd.DataFrame, stocks, gross_lookup: dict) -> pd.Series:
    rets = closes[stocks].pct_change()
    weights = pd.DataFrame(np.nan, index=closes.index, columns=stocks)
    for i in rebal_positions(len(closes)):
        mom = closes[stocks].iloc[i - SKIP] / closes[stocks].iloc[i - LOOKBACK] - 1.0
        top = mom.dropna().nlargest(TOP_N).index
        if len(top) == 0:
            continue
        gross = gross_lookup.get(i, 0.5)
        weights.iloc[i] = 0.0
        weights.iloc[i, weights.columns.get_indexer(top)] = gross / len(top)
    weights = weights.ffill().fillna(0.0)
    turnover = weights.diff().abs().sum(axis=1).fillna(0.0)
    port = (weights.shift(1) * rets).sum(axis=1) - turnover * SLIPPAGE
    return port.loc[closes.index[LOOKBACK + 1]:]


def rolling_cum(r: pd.Series, w: int) -> pd.Series:
    return (1 + r).rolling(w).apply(np.prod, raw=True) - 1


def win_rate(strat: pd.Series, bench: pd.Series) -> float:
    df = pd.DataFrame({"s": rolling_cum(strat, WINDOW_30),
                       "b": rolling_cum(bench, WINDOW_30)}).dropna()
    return ((df["s"] - df["b"]) > 0).mean()


def row(name, port, spy, rsp):
    m = compute_metrics(100000 * (1 + port).cumprod())
    print(f"{name:34s} {m.total_return:+8.1%} {m.sharpe:6.2f} {m.max_drawdown:7.1%} "
          f"{win_rate(port, spy):7.0%} {win_rate(port, rsp):7.0%}")


def main() -> None:
    print("Downloading broad universe + benchmarks (~30s)...")
    closes = fetch_closes(UNIVERSE + ["SPY", "RSP"]).dropna(axis=1)
    stocks = [c for c in closes.columns if c not in ("SPY", "RSP")]
    ex_mega = [s for s in stocks if s not in MEGA7]
    print(f"Full universe: {len(stocks)} names ({len(stocks)-len(ex_mega)} mega-caps, "
          f"{len(ex_mega)} others), {len(closes)} bars "
          f"({closes.index[0].date()} -> {closes.index[-1].date()})\n")

    spy = closes["SPY"].pct_change()
    rsp = closes["RSP"].pct_change()
    start = closes.index[LOOKBACK + 1]
    spy, rsp = spy.loc[start:], rsp.loc[start:]

    print("  computing causal HMM regime overlay (~30s)...")
    gross = compute_regime_gross(closes)

    hdr = f"{'strategy':34s} {'return':>8s} {'sharpe':>6s} {'maxDD':>7s} {'>SPY':>7s} {'>RSP':>7s}"
    print("\n" + hdr)
    print("-" * len(hdr))
    row("SPY buy & hold", spy, spy, rsp)
    row("RSP (equal-weight S&P)", rsp, spy, rsp)
    full = momentum_returns(closes, stocks, gross)
    row("Momentum+HMM (full universe)", full, spy, rsp)
    row("Momentum+HMM (EXCLUDING mega-7)", momentum_returns(closes, ex_mega, gross), spy, rsp)

    # Sub-period stability for the main strategy
    print("\nSub-period stability (Momentum+HMM full) — return vs SPY, 30d win vs SPY:")
    idx = full.index
    thirds = np.array_split(idx, 3)
    for k, seg in enumerate(thirds, 1):
        seg = pd.DatetimeIndex(seg)
        s_ret = (1 + full.loc[seg]).prod() - 1
        b_ret = (1 + spy.loc[seg]).prod() - 1
        wr = win_rate(full.loc[seg], spy.loc[seg])
        print(f"  period {k} ({seg[0].date()} -> {seg[-1].date()}): "
              f"strat {s_ret:+6.1%} vs SPY {b_ret:+6.1%}   30d-win {wr:.0%}")

    print("\n'>SPY' / '>RSP' = share of rolling 30-day windows beating that benchmark.")
    print("Residual survivorship bias remains (delisted names absent); treat as plausible, not proven.")


if __name__ == "__main__":
    main()
