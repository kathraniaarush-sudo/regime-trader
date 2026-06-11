"""Shared fixtures: a synthetic two-regime price series with known structure."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def synthetic_prices() -> pd.DataFrame:
    """A price path with alternating calm-up and volatile-down regimes.

    Deterministic (seeded) so HMM tests are reproducible. Built so a regime
    detector should clearly separate the two environments.
    """
    rng = np.random.default_rng(123)
    segments = []
    price = 100.0
    # 6 alternating blocks: calm bull, then volatile bear, etc.
    for i in range(6):
        n = 120
        if i % 2 == 0:  # calm uptrend
            rets = rng.normal(0.0008, 0.006, n)
        else:           # volatile downtrend
            rets = rng.normal(-0.0010, 0.025, n)
        for r in rets:
            price *= (1 + r)
            segments.append(price)
    close = np.array(segments)
    idx = pd.bdate_range("2019-01-01", periods=len(close))
    high = close * (1 + np.abs(rng.normal(0, 0.004, len(close))))
    low = close * (1 - np.abs(rng.normal(0, 0.004, len(close))))
    vol = rng.integers(1_000_000, 5_000_000, len(close))
    return pd.DataFrame({"open": close, "high": high, "low": low, "close": close, "volume": vol}, index=idx)
