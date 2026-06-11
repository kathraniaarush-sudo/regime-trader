"""Streamlit dashboard for the regime trader.

Run with:  streamlit run regime_trader/dashboard/app.py

Shows the live-detected regime, confidence, account value (if Alpaca is
configured), a price chart with regime overlay, confidence over time, and the
signal feed reconstructed from logs/events.jsonl. Works offline using free
yfinance data even when no broker is connected.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from regime_trader.brain.hmm_engine import RegimeDetector
from regime_trader.core.features import build_features
from regime_trader.core.settings import PROJECT_ROOT, load_settings
from regime_trader.broker.market_data import get_history

REGIME_COLORS = {
    "crash": "#7f1d1d",
    "bear": "#dc2626",
    "neutral": "#a3a3a3",
    "bull": "#16a34a",
    "euphoria": "#15803d",
}

st.set_page_config(page_title="Regime Trader", layout="wide", page_icon="📈")


@st.cache_data(ttl=900)
def load_prices(symbol: str, lookback: int) -> pd.DataFrame:
    return get_history(symbol, lookback)


@st.cache_resource(show_spinner=False)
def fit_detector(symbol: str, lookback: int, hmm_cfg: tuple):
    cfg = dict(hmm_cfg)
    prices = load_prices(symbol, lookback)
    feats = build_features(prices)
    det = RegimeDetector(**cfg)
    det.fit(feats)
    series = det.detect_series(feats)
    return det, prices.reindex(feats.index), series


def read_account():
    try:
        from regime_trader.broker.alpaca_client import AlpacaClient
        client = AlpacaClient()
        if not client.is_configured:
            return None
        return client.get_account()
    except Exception:
        return None


def read_signal_feed(limit: int = 50) -> pd.DataFrame:
    path = PROJECT_ROOT / "logs" / "events.jsonl"
    if not path.exists():
        return pd.DataFrame()
    rows = []
    for line in path.read_text().splitlines()[-2000:]:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") in {"order", "regime_change", "risk_check"}:
            rows.append(ev)
    return pd.DataFrame(rows[-limit:][::-1])


def main():
    settings = load_settings()
    anchor = settings.get("universe.regime_anchor", "SPY")
    lookback = settings.get("hmm.train_lookback_days", 504)
    hmm_cfg = (
        ("min_regimes", settings.get("hmm.min_regimes", 3)),
        ("max_regimes", settings.get("hmm.max_regimes", 7)),
        ("covariance_type", settings.get("hmm.covariance_type", "diag")),
        ("min_persistence_bars", settings.get("hmm.min_persistence_bars", 3)),
        ("max_flips_in_window", settings.get("hmm.max_flips_in_window", 4)),
        ("flip_window_bars", settings.get("hmm.flip_window_bars", 20)),
    )

    st.title("📈 Regime Trader")
    st.caption("Hidden Markov regime detection · regime-driven allocation · Alpaca execution. "
               "Paper trade first — not financial advice.")

    with st.spinner(f"Detecting regimes for {anchor}…"):
        det, prices, series = fit_detector(anchor, lookback, hmm_cfg)

    last = series.iloc[-1]
    regime = last["label"]
    canonical = last["canonical"]
    conf = float(last["confidence"])
    account = read_account()

    # --- top metrics row -----------------------------------------------------
    c1, c2, c3, c4 = st.columns(4)
    color = REGIME_COLORS.get(canonical, "#a3a3a3")
    c1.markdown(f"### Regime\n<span style='color:{color};font-size:1.6rem;font-weight:700'>"
                f"{regime.upper()}</span>", unsafe_allow_html=True)
    c2.metric("Confidence", f"{conf:.0%}", "uncertain" if last["uncertain"] else "stable")
    if account:
        c3.metric("Portfolio value", f"${account.equity:,.0f}")
        c4.metric("Buying power", f"${account.buying_power:,.0f}")
    else:
        c3.metric("Portfolio value", "— (no broker)")
        c4.metric("Regimes detected", str(det.n_regimes))

    # --- price chart with regime overlay ------------------------------------
    st.subheader("Price & detected regime")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices["close"], name="Close",
                             line=dict(color="#e5e7eb", width=1.5)))
    for canon, col in REGIME_COLORS.items():
        mask = series["canonical"] == canon
        if mask.any():
            fig.add_trace(go.Scatter(
                x=prices.index[mask], y=prices["close"][mask], mode="markers",
                name=canon, marker=dict(color=col, size=5)))
    fig.update_layout(height=420, template="plotly_dark", legend_orientation="h",
                      margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # --- confidence over time ------------------------------------------------
    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Confidence over time")
        cfig = go.Figure(go.Scatter(x=series.index, y=series["confidence"],
                                    line=dict(color="#60a5fa")))
        cfig.update_layout(height=260, template="plotly_dark",
                           margin=dict(l=10, r=10, t=10, b=10), yaxis_range=[0, 1])
        st.plotly_chart(cfig, use_container_width=True)
    with col_b:
        st.subheader("Regime distribution")
        counts = series["canonical"].value_counts()
        dfig = go.Figure(go.Bar(x=counts.index, y=counts.values,
                                marker_color=[REGIME_COLORS.get(r, "#888") for r in counts.index]))
        dfig.update_layout(height=260, template="plotly_dark",
                           margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(dfig, use_container_width=True)

    # --- risk controls -------------------------------------------------------
    st.subheader("Risk controls")
    rc = settings.get("risk", {})
    blocked = (PROJECT_ROOT / rc.get("block_file", "TRADING_BLOCKED")).exists()
    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Kill switch", "🔴 BLOCKED" if blocked else "🟢 Armed")
    r2.metric("Daily flatten", f"-{rc.get('daily_loss_flatten', 0.03):.0%}")
    r3.metric("Max drawdown stop", f"-{rc.get('max_drawdown_stop', 0.10):.0%}")
    r4.metric("Max leverage", f"{rc.get('max_leverage', 1.25):.2f}x")

    # --- signal feed ---------------------------------------------------------
    st.subheader("Signal feed")
    feed = read_signal_feed()
    if feed.empty:
        st.info("No events yet. Run `python -m regime_trader.main run --once` to generate activity.")
    else:
        cols = [c for c in ["ts", "event", "msg", "regime", "confidence", "equity"] if c in feed.columns]
        st.dataframe(feed[cols], use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
