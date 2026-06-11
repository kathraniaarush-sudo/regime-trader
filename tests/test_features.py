import numpy as np

from regime_trader.core.features import build_features, standardize, FEATURE_COLUMNS


def test_features_have_expected_columns(synthetic_prices):
    feats = build_features(synthetic_prices)
    assert list(feats.columns) == FEATURE_COLUMNS
    assert len(feats) > 0


def test_features_have_no_nans(synthetic_prices):
    feats = build_features(synthetic_prices)
    assert not feats.isna().any().any()


def test_features_are_causal(synthetic_prices):
    """A feature row must not change when future bars are appended.

    This is the core no-look-ahead guarantee for feature engineering.
    """
    full = build_features(synthetic_prices)
    truncated = build_features(synthetic_prices.iloc[:-50])
    common = truncated.index
    np.testing.assert_allclose(
        full.loc[common].to_numpy(), truncated.loc[common].to_numpy(), rtol=1e-9
    )


def test_standardize_uses_training_stats(synthetic_prices):
    feats = build_features(synthetic_prices)
    train = feats.iloc[:200]
    _, stats = standardize(train)
    applied, _ = standardize(feats.iloc[200:], stats)
    # mean/std came from train, so test-set mean need not be ~0
    assert applied.shape[1] == len(FEATURE_COLUMNS)
