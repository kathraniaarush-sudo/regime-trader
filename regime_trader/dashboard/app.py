"""Streamlit dashboard for the regime trader — Robinhood-style dark UI.

Run with:  streamlit run regime_trader/dashboard/app.py

Design language: near-black surface, one green/red gain-loss accent, a clean
grotesk typeface, oversized portfolio number, a single smooth area chart with a
regime ribbon beneath it. Works offline using free yfinance data; shows live
Alpaca account figures when credentials are configured.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Streamlit runs this file as a standalone script, so the project root is not on
# the import path by default. Add it so `import regime_trader` resolves whether
# launched via `streamlit run` or `python -m`.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from regime_trader.brain.hmm_engine import RegimeDetector
from regime_trader.core.features import build_features
from regime_trader.core.settings import PROJECT_ROOT, load_settings
from regime_trader.broker.market_data import get_history

# ----------------------------------------------------------------- palette
BG = "#0a0b0e"
SURFACE = "#14161b"
HAIRLINE = "rgba(255,255,255,0.07)"
TEXT = "#f4f5f6"
MUTED = "#8b9097"
UP = "#00c805"        # Robinhood green: gains / risk-on regimes
DOWN = "#ff5a5f"      # losses / risk-off regimes
FONT = "Manrope, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

REGIME_COLORS = {
    "crash": "#c4102e",
    "bear": "#ff5a5f",
    "neutral": "#7d838c",
    "bull": "#00c805",
    "euphoria": "#36e27b",
}
RISK_ON = {"bull", "euphoria"}

st.set_page_config(page_title="Regime Trader", layout="wide", page_icon="📈",
                   initial_sidebar_state="collapsed")


# ----------------------------------------------------------------- styling
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');

        .stApp {{ background: {BG}; }}
        html, body, [data-testid="stAppViewContainer"], [class*="css"] {{
            font-family: {FONT};
            color: {TEXT};
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        #MainMenu, footer, [data-testid="stToolbar"] {{ visibility: hidden; }}
        .block-container {{ max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }}

        .rt-brand {{ display:flex; align-items:center; gap:.6rem; margin-bottom:2.1rem; }}
        .rt-brand .mark {{
            width:30px; height:30px; border-radius:9px;
            background: linear-gradient(150deg, {UP}, #00e676);
            display:flex; align-items:center; justify-content:center;
            color:#06210b; font-weight:800; font-size:18px;
        }}
        .rt-brand .name {{ font-weight:700; font-size:1.05rem; letter-spacing:-.01em; }}
        .rt-brand .status {{
            margin-left:auto; font-size:.78rem; color:{MUTED}; font-weight:600;
            display:flex; align-items:center; gap:.45rem;
        }}
        .rt-brand .live {{ width:8px; height:8px; border-radius:50%; }}

        .rt-label {{ color:{MUTED}; font-size:.82rem; font-weight:600; letter-spacing:.01em; }}
        .rt-value {{
            font-size:3.4rem; font-weight:800; line-height:1.02; letter-spacing:-.03em;
            font-variant-numeric: tabular-nums; margin:.15rem 0 .35rem;
        }}
        .rt-delta {{ font-size:1.0rem; font-weight:700; font-variant-numeric: tabular-nums; }}
        .rt-delta.up {{ color:{UP}; }}
        .rt-delta.down {{ color:{DOWN}; }}
        .rt-delta .sub {{ color:{MUTED}; font-weight:600; margin-left:.4rem; }}

        .rt-pill {{
            display:inline-flex; align-items:center; gap:.55rem;
            padding:.5rem .95rem; border-radius:999px;
            background:var(--c-soft); border:1px solid var(--c-line);
            font-weight:800; font-size:.95rem; letter-spacing:.02em;
        }}
        .rt-pill .swatch {{ width:9px; height:9px; border-radius:50%; background:var(--c); }}
        .rt-pill .conf {{ color:{MUTED}; font-weight:700; }}
        .rt-right {{ text-align:right; }}
        .rt-right .bp {{ color:{MUTED}; font-size:.86rem; font-weight:600; margin-top:.7rem; }}
        .rt-right .bp b {{ color:{TEXT}; font-variant-numeric: tabular-nums; }}

        .rt-section {{ font-size:.82rem; color:{MUTED}; font-weight:700; letter-spacing:.06em;
            text-transform:uppercase; margin:2.4rem 0 .9rem; }}

        .rt-tiles {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; }}
        .rt-tile {{
            background:{SURFACE}; border:1px solid {HAIRLINE}; border-radius:14px;
            padding:1.0rem 1.1rem;
        }}
        .rt-tile .k {{ color:{MUTED}; font-size:.78rem; font-weight:600; }}
        .rt-tile .v {{ font-size:1.35rem; font-weight:800; margin-top:.35rem;
            font-variant-numeric: tabular-nums; letter-spacing:-.01em; }}

        .rt-feed {{ background:{SURFACE}; border:1px solid {HAIRLINE}; border-radius:14px; overflow:hidden; }}
        .rt-row {{ display:flex; align-items:center; gap:1rem; padding:.85rem 1.15rem;
            border-top:1px solid {HAIRLINE}; }}
        .rt-row:first-child {{ border-top:none; }}
        .rt-row .ic {{ width:34px; height:34px; border-radius:10px; flex:0 0 auto;
            display:flex; align-items:center; justify-content:center; font-weight:800; font-size:.95rem; }}
        .rt-row .body {{ flex:1; min-width:0; }}
        .rt-row .t {{ font-weight:700; font-size:.92rem; }}
        .rt-row .s {{ color:{MUTED}; font-size:.79rem; margin-top:.12rem;
            white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
        .rt-row .when {{ color:{MUTED}; font-size:.78rem; font-variant-numeric:tabular-nums; flex:0 0 auto; }}
        .rt-empty {{ color:{MUTED}; padding:1.6rem 1.2rem; font-size:.9rem; }}
        .rt-legend {{ display:flex; flex-wrap:wrap; gap:1.1rem; margin-top:.6rem; }}
        .rt-legend span {{ color:{MUTED}; font-size:.8rem; font-weight:600;
            display:inline-flex; align-items:center; gap:.4rem; }}
        .rt-legend i {{ width:10px; height:10px; border-radius:3px; display:inline-block; }}

        @media (max-width: 760px) {{
            .rt-value {{ font-size:2.5rem; }}
            .rt-tiles {{ grid-template-columns:repeat(2,1fr); }}
            .rt-right {{ text-align:left; margin-top:1rem; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------------- data
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


def fetch_live() -> dict:
    """Best-effort Alpaca read: account, open positions, market status."""
    out = {"account": None, "open_pl": 0.0, "positions": 0, "market_open": None}
    try:
        from regime_trader.broker.alpaca_client import AlpacaClient
        from regime_trader.broker.position_tracker import PositionTracker
        client = AlpacaClient()
        if not client.is_configured:
            return out
        out["account"] = client.get_account()
        try:
            out["market_open"] = client.is_market_open()
        except Exception:
            pass
        positions = PositionTracker(client).list_positions()
        out["positions"] = len(positions)
        out["open_pl"] = sum(p.unrealized_pl for p in positions)
    except Exception:
        pass
    return out


def read_signal_feed(limit: int = 10) -> list[dict]:
    path = PROJECT_ROOT / "logs" / "events.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines()[-2000:]:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") in {"order", "regime_change", "risk_check"}:
            rows.append(ev)
    return rows[-limit:][::-1]


# ----------------------------------------------------------------- helpers
def money(x: float) -> str:
    return f"${x:,.2f}"


def contiguous_runs(series: pd.Series):
    vals, idx = series.tolist(), series.index
    runs, start = [], 0
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            runs.append((idx[start], idx[i - 1], vals[start]))
            start = i
    runs.append((idx[start], idx[-1], vals[start]))
    return runs


def price_chart(prices: pd.DataFrame) -> go.Figure:
    close = prices["close"]
    rising = close.iloc[-1] >= close.iloc[0]
    color = UP if rising else DOWN
    fill = "rgba(0,200,5,0.10)" if rising else "rgba(255,90,95,0.10)"
    lo, hi = close.min(), close.max()
    pad = (hi - lo) * 0.08 or 1.0

    fig = go.Figure(go.Scatter(
        x=close.index, y=close, mode="lines",
        line=dict(color=color, width=2.2, shape="spline", smoothing=0.4),
        fill="tozeroy", fillcolor=fill,
        hovertemplate="%{x|%b %d, %Y}   $%{y:.2f}<extra></extra>",
    ))
    fig.update_layout(
        height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=8, b=0), font=dict(family=FONT, color=MUTED),
        hovermode="x", hoverlabel=dict(bgcolor=SURFACE, bordercolor=HAIRLINE,
                                       font=dict(family=FONT, color=TEXT)),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=False, zeroline=False, showspikes=True,
                     spikecolor=MUTED, spikethickness=1, spikedash="dot",
                     spikemode="across", tickfont=dict(size=11), color=MUTED)
    fig.update_yaxes(visible=False, range=[lo - pad, hi + pad])
    return fig


def regime_ribbon(prices: pd.DataFrame, series: pd.DataFrame) -> go.Figure:
    idx = prices.index
    fig = go.Figure(go.Scatter(x=[idx[0], idx[-1]], y=[0, 0], mode="lines",
                               line=dict(width=0), hoverinfo="skip", showlegend=False))
    for x0, x1, reg in contiguous_runs(series["canonical"]):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=REGIME_COLORS.get(reg, MUTED),
                      opacity=0.92, line_width=0, layer="below")
    fig.update_layout(height=34, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=0, b=0))
    fig.update_xaxes(visible=False, range=[idx[0], idx[-1]])
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig


def mini_line(series: pd.Series, color: str) -> go.Figure:
    fig = go.Figure(go.Scatter(x=series.index, y=series, mode="lines",
                               line=dict(color=color, width=2),
                               hovertemplate="%{y:.0%}<extra></extra>"))
    fig.update_layout(height=150, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=0, r=0, t=6, b=0), showlegend=False,
                      font=dict(family=FONT, color=MUTED), hovermode="x")
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False, range=[0, 1.02])
    return fig


# ----------------------------------------------------------------- view
def main():
    inject_css()
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

    with st.spinner(""):
        det, prices, series = fit_detector(anchor, lookback, hmm_cfg)
    live = fetch_live()

    last = series.iloc[-1]
    regime = str(last["label"])
    canon = str(last["canonical"])
    conf = float(last["confidence"])
    c = REGIME_COLORS.get(canon, MUTED)

    # --- brand bar ---------------------------------------------------------
    if live["market_open"] is True:
        status, dot = "Market open", UP
    elif live["market_open"] is False:
        status, dot = "Market closed", MUTED
    else:
        status, dot = "Paper account not connected", MUTED
    st.markdown(
        f"""<div class="rt-brand">
            <div class="mark">M</div><div class="name">Regime Trader</div>
            <div class="status"><span class="live" style="background:{dot}"></span>{status}</div>
        </div>""", unsafe_allow_html=True)

    # --- hero: portfolio value + regime -----------------------------------
    acct = live["account"]
    left, right = st.columns([3, 2], gap="large")
    with left:
        if acct:
            pl = live["open_pl"]
            base = acct.equity - pl
            pct = (pl / base * 100) if base else 0.0
            up = pl >= 0
            arrow = "▲" if up else "▼"
            cls = "up" if up else "down"
            sub = (f"{live['positions']} open position" + ("s" if live["positions"] != 1 else "")
                   if live["positions"] else "no open positions")
            st.markdown(
                f"""<div class="rt-label">Portfolio value</div>
                <div class="rt-value">{money(acct.equity)}</div>
                <div class="rt-delta {cls}">{arrow} {money(abs(pl))} ({pct:+.2f}%)
                    <span class="sub">Open P/L · {sub}</span></div>""",
                unsafe_allow_html=True)
        else:
            st.markdown(
                f"""<div class="rt-label">Anchor price · {anchor}</div>
                <div class="rt-value">{money(prices['close'].iloc[-1])}</div>
                <div class="rt-delta"><span class="sub">Add Alpaca keys to .env for live portfolio value</span></div>""",
                unsafe_allow_html=True)
    with right:
        bp = f"<div class='bp'>Buying power <b>{money(acct.buying_power)}</b></div>" if acct else ""
        st.markdown(
            f"""<div class="rt-right">
                <div class="rt-pill" style="--c:{c}; --c-soft:{c}1f; --c-line:{c}3d">
                    <span class="swatch"></span>{regime.upper()}
                    <span class="conf">{conf:.0%}</span>
                </div>{bp}
            </div>""", unsafe_allow_html=True)

    # --- price chart + regime ribbon --------------------------------------
    st.plotly_chart(price_chart(prices), use_container_width=True,
                    config={"displayModeBar": False})
    st.plotly_chart(regime_ribbon(prices, series), use_container_width=True,
                    config={"displayModeBar": False})
    present = [r for r in REGIME_COLORS if (series["canonical"] == r).any()]
    st.markdown(
        "<div class='rt-legend'>" + "".join(
            f"<span><i style='background:{REGIME_COLORS[r]}'></i>{r}</span>" for r in present
        ) + "</div>", unsafe_allow_html=True)

    # --- confidence + distribution ----------------------------------------
    a, b = st.columns(2, gap="large")
    with a:
        st.markdown("<div class='rt-section'>Confidence</div>", unsafe_allow_html=True)
        st.plotly_chart(mini_line(series["confidence"], c), use_container_width=True,
                        config={"displayModeBar": False})
    with b:
        st.markdown("<div class='rt-section'>Days in each regime</div>", unsafe_allow_html=True)
        counts = series["canonical"].value_counts()
        bar = go.Figure(go.Bar(
            x=counts.index, y=counts.values,
            marker_color=[REGIME_COLORS.get(r, MUTED) for r in counts.index],
            hovertemplate="%{x}: %{y} days<extra></extra>"))
        bar.update_layout(height=150, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                          margin=dict(l=0, r=0, t=6, b=0), showlegend=False,
                          font=dict(family=FONT, color=MUTED))
        bar.update_xaxes(showgrid=False, tickfont=dict(size=11), color=MUTED)
        bar.update_yaxes(visible=False)
        st.plotly_chart(bar, use_container_width=True, config={"displayModeBar": False})

    # --- risk controls -----------------------------------------------------
    st.markdown("<div class='rt-section'>Risk controls</div>", unsafe_allow_html=True)
    rc = settings.get("risk", {})
    blocked = (PROJECT_ROOT / rc.get("block_file", "TRADING_BLOCKED")).exists()
    kill_v = (f"<span style='color:{DOWN}'>Blocked</span>" if blocked
              else f"<span style='color:{UP}'>Armed</span>")
    tiles = [
        ("Kill switch", kill_v),
        ("Daily flatten", f"-{rc.get('daily_loss_flatten', 0.03):.0%}"),
        ("Max drawdown stop", f"-{rc.get('max_drawdown_stop', 0.10):.0%}"),
        ("Max leverage", f"{rc.get('max_leverage', 1.25):.2f}x"),
    ]
    st.markdown(
        "<div class='rt-tiles'>" + "".join(
            f"<div class='rt-tile'><div class='k'>{k}</div><div class='v'>{v}</div></div>"
            for k, v in tiles
        ) + "</div>", unsafe_allow_html=True)

    # --- signal feed -------------------------------------------------------
    st.markdown("<div class='rt-section'>Signal feed</div>", unsafe_allow_html=True)
    feed = read_signal_feed()
    if not feed:
        st.markdown(
            "<div class='rt-feed'><div class='rt-empty'>No activity yet. "
            "Run <code>python -m regime_trader.main run --once</code> to generate signals.</div></div>",
            unsafe_allow_html=True)
    else:
        rows = []
        for ev in feed:
            kind = ev.get("event")
            if kind == "order":
                side = str(ev.get("side", "")).upper()
                col = UP if side == "BUY" else DOWN
                ic = "▲" if side == "BUY" else "▼"
                title = f"{side} {ev.get('qty','')} {ev.get('symbol','')}"
            elif kind == "regime_change":
                col = REGIME_COLORS.get(ev.get("regime", ""), MUTED)
                ic, title = "R", f"Regime -> {ev.get('regime','')}"
            else:
                col, ic, title = MUTED, "•", "Risk check"
            ts = str(ev.get("ts", ""))[:19].replace("T", " ")
            rows.append(
                f"""<div class="rt-row">
                    <div class="ic" style="background:{col}22;color:{col}">{ic}</div>
                    <div class="body"><div class="t">{title}</div>
                        <div class="s">{ev.get('msg','')}</div></div>
                    <div class="when">{ts}</div></div>""")
        st.markdown("<div class='rt-feed'>" + "".join(rows) + "</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
