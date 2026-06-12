import numpy as np
import pandas as pd
import pytest

from regime_trader.strategy.momentum import MomentumRanker
from regime_trader.strategy.portfolio import (
    PortfolioConstructor, realised_vol, vol_target_scalar,
)


def _trending_prices(n=300, seed=0):
    """Three names: WIN trends up hardest, MID mild, LOSE drifts down."""
    rng = np.random.default_rng(seed)
    idx = pd.bdate_range("2021-01-01", periods=n)
    out = {}
    for name, drift in [("WIN", 0.0015), ("MID", 0.0003), ("LOSE", -0.0010)]:
        rets = rng.normal(drift, 0.005, n)
        out[name] = 100 * np.cumprod(1 + rets)
    return pd.DataFrame(out, index=idx)


# ----------------------------------------------------------------- momentum
def test_ranker_orders_by_momentum():
    prices = _trending_prices()
    ranker = MomentumRanker(lookback=252, skip=21, top_n=3)
    ranked = ranker.select(prices)
    assert ranked[0] == "WIN"
    assert ranked[-1] == "LOSE"


def test_ranker_top_n_limit():
    prices = _trending_prices()
    assert len(MomentumRanker(top_n=2).select(prices)) == 2


def test_ranker_insufficient_history_returns_empty():
    prices = _trending_prices(n=100)  # < lookback + 1
    assert MomentumRanker(lookback=252).select(prices) == []


# --------------------------------------------------------------- vol target
def test_vol_target_scalar_clamps():
    # low realised vol -> want to lever up, but capped
    assert vol_target_scalar(0.02, target_vol=0.09, max_leverage=1.5) == 1.5
    # high realised vol -> scale down below 1
    assert vol_target_scalar(0.36, target_vol=0.09, max_leverage=1.5) == pytest.approx(0.25)
    # unusable vol -> flat
    assert vol_target_scalar(0.0, 0.09, 1.5) == 0.0


def test_realised_vol_positive():
    rng = np.random.default_rng(4)
    rets = pd.Series(rng.normal(0, 0.01, 40))
    assert realised_vol(rets, lookback=20) > 0
    assert realised_vol(pd.Series([0.0]), 20) == 0.0   # too few points


# ----------------------------------------------------------- base weights
def test_base_weights_flat_regime_is_cash():
    pc = PortfolioConstructor(top_n=2)
    assert pc.base_weights(["WIN", "MID"], "crash") == {}
    assert pc.base_weights(["WIN", "MID"], "bear") == {}


def test_base_weights_bull_equal_weight():
    pc = PortfolioConstructor(top_n=2, regime_gross={"bull": 1.0})
    w = pc.base_weights(["A", "B"], "bull")
    assert w == {"A": 0.5, "B": 0.5}


def test_base_weights_neutral_partial():
    pc = PortfolioConstructor(top_n=2, regime_gross={"neutral": 0.6})
    w = pc.base_weights(["A", "B"], "neutral")
    assert sum(w.values()) == pytest.approx(0.6)


# ----------------------------------------------------- target weights (live)
def _calm_returns(seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.001, 40))   # tiny vol -> vol target wants leverage


def _wild_returns(seed=2):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0, 0.04, 40))    # high vol -> scale down


def test_target_weights_low_vol_levers_to_cap():
    pc = PortfolioConstructor(top_n=2, regime_gross={"bull": 1.0}, target_vol=0.09, max_leverage=1.5)
    w = pc.target_weights(["A", "B"], "bull", _calm_returns())
    assert set(w) == {"A", "B"}
    assert sum(w.values()) == pytest.approx(1.5)        # pinned to max_leverage
    assert w["A"] == pytest.approx(0.75)


def test_target_weights_high_vol_scales_down():
    pc = PortfolioConstructor(top_n=2, regime_gross={"bull": 1.0}, target_vol=0.09, max_leverage=1.5)
    w = pc.target_weights(["A", "B"], "bull", _wild_returns())
    assert 0 < sum(w.values()) < 1.0


def test_target_weights_flat_regime_is_cash():
    pc = PortfolioConstructor(top_n=2)
    assert pc.target_weights(["A", "B"], "crash", _calm_returns()) == {}
