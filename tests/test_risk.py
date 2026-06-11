from datetime import datetime, timezone

import pytest

from regime_trader.risk.risk_manager import RiskManager


@pytest.fixture
def rm(tmp_path):
    cfg = {
        "max_risk_per_trade": 0.01,
        "max_leverage": 1.25,
        "daily_loss_halve": 0.02,
        "daily_loss_flatten": 0.03,
        "weekly_loss_halve": 0.05,
        "max_drawdown_stop": 0.10,
        "block_file": str(tmp_path / "TRADING_BLOCKED"),
    }
    return RiskManager(cfg)


def _t(h=12):
    return datetime(2024, 1, 2, h, 0, tzinfo=timezone.utc)


def test_clean_account_is_tradable(rm):
    d = rm.evaluate(100_000, _t())
    assert d.tradable
    assert d.size_multiplier == 1.0


def test_daily_2pct_halves(rm):
    rm.evaluate(100_000, _t(9))
    d = rm.evaluate(97_900, _t(12))  # -2.1% on the day
    assert d.size_multiplier == 0.5
    assert d.allow_new_entries


def test_daily_3pct_flattens(rm):
    rm.evaluate(100_000, _t(9))
    d = rm.evaluate(96_500, _t(12))  # -3.5% on the day
    assert d.flatten_all
    assert not d.allow_new_entries


def test_max_drawdown_writes_block_file(rm):
    rm.evaluate(100_000, _t(9))
    d = rm.evaluate(89_000, _t(12))  # -11% from peak
    assert d.halted
    assert rm.is_blocked()
    # sticky: still halted on the next evaluation even if equity recovers
    d2 = rm.evaluate(100_000, _t(13))
    assert d2.halted


def test_position_sizing_caps_risk(rm):
    # risk 1% of 100k = $1000; stop is $2 away -> 500 shares max
    qty = rm.size_position(100_000, price=100, stop_price=98)
    assert qty == 500


def test_position_sizing_respects_weight_cap(rm):
    # target weight 0.1 of 100k at $100 -> 100 shares cap, below risk-based size
    qty = rm.size_position(100_000, price=100, stop_price=99.9, target_weight=0.1)
    assert qty == 100


def test_size_multiplier_scales_position(rm):
    full = rm.size_position(100_000, 100, 98, size_multiplier=1.0)
    half = rm.size_position(100_000, 100, 98, size_multiplier=0.5)
    assert half == full // 2


def test_leverage_check(rm):
    assert rm.within_leverage(100_000, 120_000)
    assert not rm.within_leverage(100_000, 130_000)


def test_correlation_block():
    import pandas as pd
    rm = RiskManager({"max_correlation": 0.8})
    corr = pd.DataFrame(
        [[1.0, 0.95], [0.95, 1.0]], index=["AAA", "BBB"], columns=["AAA", "BBB"]
    )
    assert rm.correlation_ok("AAA", ["BBB"], corr) is False
    assert rm.correlation_ok("AAA", [], None) is True
