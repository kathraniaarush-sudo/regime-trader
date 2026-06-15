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


def test_risk_adjusted_momentum_penalises_volatility():
    # two names, same total momentum but B is far more volatile -> A scores higher
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2021-01-01", periods=300)
    steady = 100 * np.cumprod(1 + rng.normal(0.0008, 0.004, 300))
    jumpy = 100 * np.cumprod(1 + rng.normal(0.0008, 0.030, 300))
    prices = pd.DataFrame({"STEADY": steady, "JUMPY": jumpy}, index=idx)
    sc = MomentumRanker(risk_adjusted=True).scores(prices)
    assert sc["STEADY"] > sc["JUMPY"]


def test_trend_filter_excludes_downtrending_names():
    idx = pd.bdate_range("2021-01-01", periods=300)
    up = pd.Series(range(300), index=idx).astype(float) + 100        # clean uptrend
    down = pd.Series(range(300, 0, -1), index=idx).astype(float) + 100  # downtrend
    prices = pd.DataFrame({"UP": up, "DOWN": down}, index=idx)
    picked = MomentumRanker(trend_filter=True, trend_ma=200, top_n=2).select(prices)
    assert "UP" in picked and "DOWN" not in picked


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


def test_inverse_vol_weighting_favours_the_calmer_name():
    rng = np.random.default_rng(11)
    idx = pd.bdate_range("2022-01-01", periods=90)
    calm = 100 * np.cumprod(1 + rng.normal(0, 0.004, 90))
    wild = 100 * np.cumprod(1 + rng.normal(0, 0.030, 90))
    closes = pd.DataFrame({"CALM": calm, "WILD": wild}, index=idx)
    pc = PortfolioConstructor(top_n=2, regime_gross={"bull": 1.0}, weighting="inverse_vol",
                              vol_weight_lookback=60)
    w = pc.base_weights(["CALM", "WILD"], "bull", closes)
    assert w["CALM"] > w["WILD"]
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-6)   # gross 1.0 fully allocated


def test_equal_weighting_default():
    pc = PortfolioConstructor(top_n=2, regime_gross={"bull": 1.0})
    w = pc.base_weights(["A", "B"], "bull")   # no closes -> equal
    assert w == {"A": 0.5, "B": 0.5}
