"""Feature engineering for the regime model.

The HMM does not see raw prices. It sees a small set of stationary,
volatility-flavoured features so it can separate calm trending markets from
choppy/crashing ones. Every feature uses ONLY past data at each row (no
forward fills, no centred windows) so the live and backtest paths agree and
there is no look-ahead bias.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# Column names the HMM engine expects, in order.
FEATURE_COLUMNS = ["log_return", "vol_20", "vol_ratio", "abs_return", "range_pct"]


def _true_range_pct(df: pd.DataFrame) -> pd.Series:
    """High-low range as a fraction of the close (intraday volatility proxy)."""
    if {"high", "low", "close"}.issubset(df.columns):
        rng = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
        return rng
    # Fall back to absolute return if OHLC is unavailable.
    return df["close"].pct_change().abs()


def build_features(prices: pd.DataFrame) -> pd.DataFrame:
    """Turn an OHLC(V) frame into the HMM feature matrix.

    Parameters
    ----------
    prices : DataFrame indexed by date with at least a 'close' column
             (and ideally 'high'/'low'/'volume').

    Returns
    -------
    DataFrame with FEATURE_COLUMNS, rows containing NaNs from the warm-up
    window dropped. Index is preserved so callers can align with prices.
    """
    if "close" not in prices.columns:
        raise ValueError("prices must contain a 'close' column")

    close = prices["close"].astype(float)
    log_ret = np.log(close / close.shift(1))

    vol_20 = log_ret.rolling(20).std()
    vol_60 = log_ret.rolling(60).std()
    # Short vs long vol: > 1 means volatility is expanding (stress building).
    vol_ratio = vol_20 / vol_60.replace(0, np.nan)

    feats = pd.DataFrame(
        {
            "log_return": log_ret,
            "vol_20": vol_20,
            "vol_ratio": vol_ratio,
            "abs_return": log_ret.abs(),
            "range_pct": _true_range_pct(prices),
        },
        index=prices.index,
    )
    return feats[FEATURE_COLUMNS].dropna()


def standardize(features: pd.DataFrame, stats: tuple[pd.Series, pd.Series] | None = None):
    """Z-score features. Returns (standardized, (mean, std)).

    Pass `stats` from a training window to apply the SAME transform to
    out-of-sample data — never re-fit scaling on test data.
    """
    if stats is None:
        mean = features.mean()
        std = features.std(ddof=0).replace(0, 1.0)
    else:
        mean, std = stats
    return (features - mean) / std, (mean, std)
