from regime_trader.brain.hmm_engine import RegimeState
from regime_trader.strategy.allocation import RegimeAllocator


def _state(canon, conf=0.9, uncertain=False):
    return RegimeState(label=canon, canonical=canon, confidence=conf,
                       state=0, raw_label=canon, uncertain=uncertain)


def test_crash_goes_flat():
    alloc = RegimeAllocator()
    sig = alloc.allocate(_state("crash"))
    assert sig.target_weight == 0.0
    assert sig.direction == "flat"


def test_bull_more_invested_than_bear():
    alloc = RegimeAllocator()
    bull = alloc.allocate(_state("bull"))
    bear = alloc.allocate(_state("bear"))
    assert bull.target_weight > bear.target_weight > 0


def test_euphoria_allows_leverage():
    alloc = RegimeAllocator()
    sig = alloc.allocate(_state("euphoria"))
    assert sig.leverage > 1.0


def test_confidence_scaling_reduces_weight():
    alloc = RegimeAllocator({"confidence_scaling": True, "min_confidence": 0.5})
    high = alloc.allocate(_state("bull", conf=0.95))
    low = alloc.allocate(_state("bull", conf=0.55))
    assert low.target_weight < high.target_weight


def test_uncertainty_halves_weight():
    alloc = RegimeAllocator({"confidence_scaling": False})
    certain = alloc.allocate(_state("bull", uncertain=False))
    flicker = alloc.allocate(_state("bull", uncertain=True))
    assert abs(flicker.target_weight - certain.target_weight * 0.5) < 1e-9


def test_rebalance_threshold():
    alloc = RegimeAllocator({"rebalance_threshold": 0.05})
    sig = alloc.allocate(_state("bull", conf=1.0))
    assert alloc.needs_rebalance(0.0, sig) is True
    assert alloc.needs_rebalance(sig.target_weight - 0.01, sig) is False
