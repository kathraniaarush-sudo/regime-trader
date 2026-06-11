import numpy as np

from regime_trader.brain.hmm_engine import RegimeDetector, regime_labels, canonical
from regime_trader.core.features import build_features


def test_label_sets_are_ordered():
    assert regime_labels(3) == ["bear", "neutral", "bull"]
    assert regime_labels(5) == ["crash", "bear", "neutral", "bull", "euphoria"]
    assert canonical("mania") == "euphoria"
    assert canonical("severe_bear") == "bear"


def test_fit_selects_regime_count_in_range(synthetic_prices):
    feats = build_features(synthetic_prices)
    det = RegimeDetector(min_regimes=2, max_regimes=5, n_iter=50).fit(feats)
    assert det.n_regimes is not None
    assert 2 <= det.n_regimes <= 5


def test_labels_sorted_by_return(synthetic_prices):
    """Higher-return states must get higher-ranked labels."""
    feats = build_features(synthetic_prices)
    det = RegimeDetector(min_regimes=2, max_regimes=4, n_iter=80).fit(feats)
    order = ["crash", "severe_bear", "bear", "neutral", "bull", "euphoria", "mania"]
    # mean standardized return per state should be non-decreasing along label order
    std = (feats - feats.mean()) / feats.std(ddof=0).replace(0, 1)
    X = std.to_numpy()
    states = det.model.predict(X)
    means = {det.state_to_label[s]: X[states == s, 0].mean() for s in range(det.n_regimes)}
    ranked = sorted(means, key=lambda lab: order.index(lab))
    vals = [means[r] for r in ranked]
    assert vals == sorted(vals)


def test_forward_filter_is_causal(synthetic_prices):
    """The regime at bar t must not change when later bars are added.

    This proves we use forward filtering, not the look-ahead-prone smoother.
    """
    feats = build_features(synthetic_prices)
    det = RegimeDetector(min_regimes=2, max_regimes=4, n_iter=80).fit(feats)

    series_full = det.detect_series(feats)
    series_short = det.detect_series(feats.iloc[:-40])
    common = series_short.index
    # raw (pre-stability) states must be identical on the shared prefix
    assert (series_full.loc[common, "state"].to_numpy()
            == series_short.loc[common, "state"].to_numpy()).all()


def test_detect_returns_valid_state(synthetic_prices):
    feats = build_features(synthetic_prices)
    det = RegimeDetector(min_regimes=2, max_regimes=4, n_iter=80).fit(feats)
    state = det.detect(feats)
    assert state.canonical in {"crash", "bear", "neutral", "bull", "euphoria"}
    assert 0.0 <= state.confidence <= 1.0
    assert abs(sum(state.posteriors.values()) - 1.0) < 1e-6


def test_stability_filter_resists_single_bar_flips():
    det = RegimeDetector(min_persistence_bars=3)
    raw = ["bull", "bull", "bear", "bull", "bull", "bull", "bull"]
    out = det._apply_stability(raw)
    # a lone 'bear' bar must not flip the confirmed regime
    assert out == ["bull"] * 7


def test_stability_filter_confirms_persistent_change():
    det = RegimeDetector(min_persistence_bars=3)
    raw = ["bull", "bull", "bear", "bear", "bear", "bear"]
    out = det._apply_stability(raw)
    assert out[-1] == "bear"
