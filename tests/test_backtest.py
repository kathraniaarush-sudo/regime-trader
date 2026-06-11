import pandas as pd
import pytest

from regime_trader.backtest.backtester import WalkForwardBacktester, stress_test
from regime_trader.backtest.performance import compute_metrics, max_drawdown


HMM_CFG = {"min_regimes": 2, "max_regimes": 4, "n_iter": 60}
STRAT_CFG = {}
BT_CFG = {"in_sample_days": 200, "out_sample_days": 100, "step_days": 100,
          "slippage_bps": 5, "initial_equity": 100000}


def test_metrics_on_known_curve():
    eq = pd.Series([100, 110, 121, 133.1])  # +10% each step
    m = compute_metrics(eq, periods_per_year=4)
    assert m.total_return == pytest.approx(0.331, rel=1e-3)
    assert m.win_rate == 1.0
    assert m.max_drawdown == 0.0


def test_max_drawdown_sign():
    eq = pd.Series([100, 120, 90, 95])
    assert max_drawdown(eq) == pytest.approx((90 - 120) / 120)


def test_walkforward_runs_and_is_consistent(synthetic_prices):
    bt = WalkForwardBacktester(HMM_CFG, STRAT_CFG, BT_CFG)
    res = bt.run(synthetic_prices)
    assert res.n_windows >= 1
    assert len(res.equity) > 0
    # equity curve and daily frame must align
    assert len(res.equity) == len(res.daily)
    # benchmarks present
    assert "buy_hold" in res.benchmarks
    assert "sma200_trend" in res.benchmarks


def test_weights_are_lagged_no_lookahead(synthetic_prices):
    """Strategy return on day t must use the weight decided on day t-1."""
    bt = WalkForwardBacktester(HMM_CFG, STRAT_CFG, BT_CFG)
    res = bt.run(synthetic_prices)
    d = res.daily
    recomputed = d["weight"].shift(1).fillna(0.0) * d["asset_ret"]
    # strat_ret = lagged weight * asset_ret minus costs, so it must be <= gross
    assert (d["strat_ret"] <= recomputed + 1e-9).all()


def test_stress_test_runs(synthetic_prices):
    bt = WalkForwardBacktester(HMM_CFG, STRAT_CFG, BT_CFG)
    m = stress_test(synthetic_prices, bt, n_crashes=2)
    assert m.n_periods > 0
