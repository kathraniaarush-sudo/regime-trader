"""The brain: Hidden Markov Model regime detection.

What it does
------------
The HMM does NOT predict future prices. It looks at volatility-flavoured
features and classifies the *current market environment* into a small number of
hidden states (regimes), e.g. crash / bear / neutral / bull / euphoria.

Three design choices matter for correctness:

1. The number of regimes is not hard-coded. We fit candidates from
   `min_regimes`..`max_regimes` and pick the best by BIC.

2. States are labelled by their empirical mean return so "bull" always means the
   high-return state regardless of how hmmlearn happened to number them.

3. Live detection uses a **forward (filtering) pass only** — the regime at bar t
   is inferred from bars 0..t exclusively. hmmlearn's `predict` runs the full
   forward-backward smoother, which peeks at future bars and injects look-ahead
   bias into a backtest. We never use it for inference.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.special import logsumexp

from regime_trader.core.features import standardize

logger = logging.getLogger("regime.hmm")

# Ordered, low-return -> high-return label sets for each supported regime count.
_LABELS: dict[int, list[str]] = {
    3: ["bear", "neutral", "bull"],
    4: ["crash", "bear", "bull", "euphoria"],
    5: ["crash", "bear", "neutral", "bull", "euphoria"],
    6: ["crash", "bear", "neutral", "bull", "euphoria", "mania"],
    7: ["crash", "severe_bear", "bear", "neutral", "bull", "euphoria", "mania"],
}

# Collapse any label down to the five canonical buckets the strategy understands.
_CANONICAL = {
    "crash": "crash",
    "severe_bear": "bear",
    "bear": "bear",
    "neutral": "neutral",
    "bull": "bull",
    "euphoria": "euphoria",
    "mania": "euphoria",
}


def regime_labels(n: int) -> list[str]:
    if n in _LABELS:
        return _LABELS[n]
    raise ValueError(f"Unsupported regime count: {n} (supported: {sorted(_LABELS)})")


def canonical(label: str) -> str:
    """Map any regime label to one of crash/bear/neutral/bull/euphoria."""
    return _CANONICAL.get(label, "neutral")


@dataclass
class RegimeState:
    """The regime decision for a single bar."""

    label: str                 # stability-filtered, return-sorted label
    canonical: str             # collapsed to the 5 standard buckets
    confidence: float          # filtered posterior prob of the active state
    state: int                 # raw hidden-state index
    raw_label: str             # label before the stability filter
    uncertain: bool = False    # True when the classifier is flickering
    posteriors: dict[str, float] = field(default_factory=dict)


class RegimeDetector:
    """Fit once on a training window, then detect causally on new bars."""

    def __init__(
        self,
        min_regimes: int = 3,
        max_regimes: int = 7,
        covariance_type: str = "diag",
        n_iter: int = 200,
        random_state: int = 42,
        min_persistence_bars: int = 3,
        max_flips_in_window: int = 4,
        flip_window_bars: int = 20,
    ):
        self.min_regimes = max(2, min_regimes)
        self.max_regimes = min(max(_LABELS), max_regimes)
        self.covariance_type = covariance_type
        self.n_iter = n_iter
        self.random_state = random_state
        self.min_persistence_bars = min_persistence_bars
        self.max_flips_in_window = max_flips_in_window
        self.flip_window_bars = flip_window_bars

        self.model: GaussianHMM | None = None
        self.n_regimes: int | None = None
        self.state_to_label: dict[int, str] = {}
        self._scale_stats = None  # (mean, std) from training features

    # ------------------------------------------------------------------ fit
    @staticmethod
    def _bic(model: GaussianHMM, X: np.ndarray) -> float:
        n, d = X.shape
        k = model.n_components
        # free params: transmat k*(k-1) + startprob (k-1) + means k*d + diag vars k*d
        n_params = k * (k - 1) + (k - 1) + k * d + k * d
        log_likelihood = model.score(X)
        return -2.0 * log_likelihood + n_params * np.log(n)

    def _fit_one(self, X: np.ndarray, k: int) -> GaussianHMM:
        model = GaussianHMM(
            n_components=k,
            covariance_type=self.covariance_type,
            n_iter=self.n_iter,
            random_state=self.random_state,
            tol=1e-4,
        )
        model.fit(X)
        return model

    def fit(self, features: pd.DataFrame) -> "RegimeDetector":
        """Select the regime count by BIC, fit the model, and label states."""
        if len(features) < 60:
            raise ValueError(f"Need >=60 rows to fit; got {len(features)}")

        std_feats, self._scale_stats = standardize(features)
        X = std_feats.to_numpy()

        best_model, best_bic, best_k = None, np.inf, None
        for k in range(self.min_regimes, self.max_regimes + 1):
            try:
                model = self._fit_one(X, k)
                bic = self._bic(model, X)
            except Exception as exc:  # degenerate fit for this k — skip it
                logger.debug("HMM fit failed for k=%d: %s", k, exc)
                continue
            logger.debug("k=%d BIC=%.1f", k, bic)
            if bic < best_bic:
                best_model, best_bic, best_k = model, bic, k

        if best_model is None:
            raise RuntimeError("HMM failed to fit for every candidate regime count")

        self.model = best_model
        self.n_regimes = best_k
        self._label_states(X)
        logger.info(
            "HMM fitted: %d regimes (BIC=%.1f) labels=%s",
            best_k, best_bic, list(self.state_to_label.values()),
        )
        return self

    def _label_states(self, X: np.ndarray) -> None:
        """Sort states by empirical mean log-return and assign ordered labels."""
        assert self.model is not None and self.n_regimes is not None
        # Viterbi path is fine here: labelling is a static training-time property,
        # not a live inference used for trading decisions.
        states = self.model.predict(X)
        mean_ret = []
        for s in range(self.n_regimes):
            mask = states == s
            # feature column 0 is the (standardized) log return
            mean_ret.append(X[mask, 0].mean() if mask.any() else 0.0)
        order = np.argsort(mean_ret)  # ascending: worst -> best
        labels = regime_labels(self.n_regimes)
        self.state_to_label = {int(state): labels[rank] for rank, state in enumerate(order)}

    # --------------------------------------------------------------- detect
    def _forward_filter(self, X: np.ndarray) -> np.ndarray:
        """Causal forward (filtering) recursion in log space.

        Returns the filtered posterior P(state_t | obs_0..t) for every t.
        Row t depends only on observations up to and including t — no peeking
        ahead. This is the key difference from `model.predict`.
        """
        assert self.model is not None
        model = self.model
        framelogprob = model._compute_log_likelihood(X)  # (T, k) emission logprobs
        log_startprob = np.log(model.startprob_ + 1e-300)
        log_transmat = np.log(model.transmat_ + 1e-300)
        T, k = framelogprob.shape

        log_alpha = np.empty((T, k))
        log_alpha[0] = log_startprob + framelogprob[0]
        for t in range(1, T):
            # alpha_t(j) = emit_t(j) + logsumexp_i(alpha_{t-1}(i) + A(i,j))
            log_alpha[t] = framelogprob[t] + logsumexp(
                log_alpha[t - 1][:, None] + log_transmat, axis=0
            )
        # normalise each row to a proper posterior
        log_post = log_alpha - logsumexp(log_alpha, axis=1, keepdims=True)
        return np.exp(log_post)

    def _filtered_states(self, features: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        std_feats, _ = standardize(features, self._scale_stats)
        post = self._forward_filter(std_feats.to_numpy())
        return post.argmax(axis=1), post

    def detect_series(self, features: pd.DataFrame) -> pd.DataFrame:
        """Causal regime per bar (for backtesting and dashboards).

        Returns a frame indexed like `features` with columns:
        state, raw_label, label (stability-filtered), canonical, confidence,
        uncertain.
        """
        self._require_fitted()
        raw_states, post = self._filtered_states(features)
        raw_labels = [self.state_to_label[int(s)] for s in raw_states]
        confidences = post[np.arange(len(post)), raw_states]

        filtered = self._apply_stability(raw_labels)
        uncertain = self._flip_flags(raw_labels)

        return pd.DataFrame(
            {
                "state": raw_states,
                "raw_label": raw_labels,
                "label": filtered,
                "canonical": [canonical(l) for l in filtered],
                "confidence": confidences,
                "uncertain": uncertain,
            },
            index=features.index,
        )

    def detect(self, features: pd.DataFrame) -> RegimeState:
        """Regime decision for the most recent bar."""
        series = self.detect_series(features)
        last = series.iloc[-1]
        _, post = self._filtered_states(features)
        posteriors = {
            self.state_to_label[s]: float(post[-1, s]) for s in range(self.n_regimes)
        }
        state = RegimeState(
            label=last["label"],
            canonical=last["canonical"],
            confidence=float(last["confidence"]),
            state=int(last["state"]),
            raw_label=last["raw_label"],
            uncertain=bool(last["uncertain"]),
            posteriors=posteriors,
        )
        return state

    # ------------------------------------------------------- stability layer
    def _apply_stability(self, raw_labels: list[str]) -> list[str]:
        """A new regime is only confirmed after persisting N consecutive bars."""
        n = self.min_persistence_bars
        confirmed = raw_labels[0]
        out = [confirmed]
        run_label = raw_labels[0]
        run_len = 1
        for lab in raw_labels[1:]:
            if lab == run_label:
                run_len += 1
            else:
                run_label, run_len = lab, 1
            if run_len >= n and run_label != confirmed:
                confirmed = run_label
            out.append(confirmed)
        return out

    def _flip_flags(self, raw_labels: list[str]) -> list[bool]:
        """Flag bars where the raw classifier flipped too often recently."""
        w = self.flip_window_bars
        flags = []
        for i in range(len(raw_labels)):
            window = raw_labels[max(0, i - w + 1): i + 1]
            flips = sum(1 for a, b in zip(window, window[1:]) if a != b)
            flags.append(flips > self.max_flips_in_window)
        return flags

    # --------------------------------------------------------------- helpers
    def _require_fitted(self) -> None:
        if self.model is None:
            raise RuntimeError("RegimeDetector.fit() must be called before detection")
