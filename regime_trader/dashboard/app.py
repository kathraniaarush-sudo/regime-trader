"""Streamlit dashboard for the regime trader — Yahoo Finance-style data display.

Run with:  streamlit run regime_trader/dashboard/app.py

Design language borrowed from Yahoo Finance: labeled market-summary tiles, a
clean formatted price chart with a regime overlay, and a watchlist-style basket
table (ticker + company + sparkline + price + % badge + target weight). Dark
theme, teal-green brand accent, green/red gain-loss semantics. Works offline
using free yfinance data; shows live Alpaca figures when configured.
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from regime_trader.brain.hmm_engine import RegimeDetector
from regime_trader.core.features import build_features
from regime_trader.core.settings import PROJECT_ROOT, load_settings
from regime_trader.broker.market_data import get_history

# ----------------------------------------------------------- Yahoo-ish palette
PAGE = "#0f1318"
SURFACE = "#191e25"
SURFACE2 = "#222933"
BORDER = "#2b323c"
TEXT = "#e7eaee"
MUTED = "#98a1ac"
UP = "#15c784"          # gains
DOWN = "#f0616d"        # losses
TEAL = "#13b29a"        # brand accent (Yahoo Finance teal-green)
FONT = "Inter, -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

REGIME_COLORS = {
    "crash": "#b22a3a", "bear": "#f0616d", "neutral": "#8a93a0",
    "bull": "#15c784", "euphoria": "#10e0a0",
}

NAMES = {
    "AAPL": "Apple", "MSFT": "Microsoft", "NVDA": "NVIDIA", "AMZN": "Amazon",
    "GOOGL": "Alphabet", "META": "Meta Platforms", "TSLA": "Tesla", "ORCL": "Oracle",
    "CSCO": "Cisco Systems", "INTC": "Intel", "IBM": "IBM", "ADBE": "Adobe",
    "CRM": "Salesforce", "AMD": "AMD", "QCOM": "Qualcomm", "TXN": "Texas Instruments",
    "HPQ": "HP Inc.", "T": "AT&T", "VZ": "Verizon", "JPM": "JPMorgan Chase",
    "BAC": "Bank of America", "WFC": "Wells Fargo", "C": "Citigroup", "GS": "Goldman Sachs",
    "MS": "Morgan Stanley", "V": "Visa", "MA": "Mastercard", "AXP": "American Express",
    "JNJ": "Johnson & Johnson", "UNH": "UnitedHealth", "PFE": "Pfizer", "MRK": "Merck",
    "ABBV": "AbbVie", "ABT": "Abbott", "TMO": "Thermo Fisher", "LLY": "Eli Lilly",
    "CVS": "CVS Health", "BMY": "Bristol-Myers Squibb", "PG": "Procter & Gamble",
    "KO": "Coca-Cola", "PEP": "PepsiCo", "WMT": "Walmart", "COST": "Costco",
    "MCD": "McDonald's", "NKE": "Nike", "SBUX": "Starbucks", "TGT": "Target",
    "HD": "Home Depot", "LOW": "Lowe's", "DIS": "Disney", "GE": "GE Aerospace",
    "CAT": "Caterpillar", "BA": "Boeing", "HON": "Honeywell", "UPS": "UPS",
    "MMM": "3M", "XOM": "Exxon Mobil", "CVX": "Chevron", "SLB": "Schlumberger",
    "GM": "General Motors", "F": "Ford", "KHC": "Kraft Heinz", "MO": "Altria",
    "SPY": "S&P 500 ETF",
}

st.set_page_config(page_title="Regime Trader", layout="wide", page_icon="📈",
                   initial_sidebar_state="collapsed")


# ----------------------------------------------------------------- styling
def inject_css() -> None:
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        .stApp {{ background:{PAGE}; }}
        html, body, [data-testid="stAppViewContainer"], [class*="css"] {{
            font-family:{FONT}; color:{TEXT}; }}
        [data-testid="stHeader"] {{ background:transparent; }}
        #MainMenu, footer, [data-testid="stToolbar"] {{ visibility:hidden; }}
        .block-container {{ max-width:1180px; padding-top:1.6rem; padding-bottom:4rem; }}

        .yf-bar {{ display:flex; align-items:center; gap:.55rem; margin-bottom:1.4rem; }}
        .yf-bar .mark {{ width:28px; height:28px; border-radius:7px;
            background:{TEAL}; display:flex; align-items:center; justify-content:center;
            color:#04201b; font-weight:800; font-size:16px; }}
        .yf-bar .name {{ font-weight:700; font-size:1.02rem; }}
        .yf-bar .status {{ margin-left:auto; font-size:.78rem; color:{MUTED}; font-weight:600;
            display:flex; align-items:center; gap:.4rem; }}
        .yf-bar .dot {{ width:8px; height:8px; border-radius:50%; }}

        /* market summary tiles */
        .yf-tiles {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; margin-bottom:.4rem; }}
        .yf-tile {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; padding:.85rem 1rem;
            box-shadow:rgba(0,0,0,0.18) 0px 4px 12px 0px; }}
        .yf-tile .k {{ color:{MUTED}; font-size:.72rem; font-weight:700; letter-spacing:.05em;
            text-transform:uppercase; }}
        .yf-tile .v {{ font-size:1.5rem; font-weight:800; margin-top:.3rem; letter-spacing:-.02em;
            font-variant-numeric:tabular-nums; line-height:1.1; }}
        .yf-tile .d {{ font-size:.82rem; font-weight:700; margin-top:.2rem; font-variant-numeric:tabular-nums; }}
        .yf-tile .d.sub {{ color:{MUTED}; font-weight:600; }}
        .up {{ color:{UP}; }} .down {{ color:{DOWN}; }}

        .yf-h {{ font-size:1.02rem; font-weight:700; margin:1.9rem 0 .7rem; }}
        .yf-h .sub {{ color:{MUTED}; font-weight:500; font-size:.84rem; margin-left:.5rem; }}

        /* watchlist */
        .yf-list {{ background:{SURFACE}; border:1px solid {BORDER}; border-radius:12px; overflow:hidden; }}
        .yf-row {{ display:grid; grid-template-columns:1.7fr 84px 1.1fr 1.4fr; align-items:center;
            gap:.6rem; padding:.72rem 1rem; border-top:1px solid {BORDER}; }}
        .yf-row:first-child {{ border-top:none; }}
        .yf-row .sym {{ font-weight:700; font-size:.95rem; }}
        .yf-row .co {{ color:{MUTED}; font-size:.76rem; white-space:nowrap; overflow:hidden;
            text-overflow:ellipsis; }}
        .yf-row .px {{ text-align:right; font-weight:700; font-variant-numeric:tabular-nums; font-size:.92rem; }}
        .yf-badge {{ display:inline-block; min-width:62px; text-align:center; padding:.18rem .4rem;
            border-radius:6px; font-size:.78rem; font-weight:700; font-variant-numeric:tabular-nums; }}
        .yf-wt {{ display:flex; align-items:center; gap:.5rem; justify-content:flex-end; }}
        .yf-wt .track {{ flex:1; max-width:120px; height:7px; border-radius:4px; background:{SURFACE2}; overflow:hidden; }}
        .yf-wt .fill {{ height:100%; background:{TEAL}; }}
        .yf-wt .pct {{ font-size:.82rem; font-weight:700; color:{TEXT}; font-variant-numeric:tabular-nums;
            min-width:42px; text-align:right; }}
        .yf-head {{ display:grid; grid-template-columns:1.7fr 84px 1.1fr 1.4fr; gap:.6rem;
            padding:.5rem 1rem; color:{MUTED}; font-size:.7rem; font-weight:700; letter-spacing:.05em;
            text-transform:uppercase; }}
        .yf-head .r {{ text-align:right; }}

        .yf-tile-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; }}
        .yf-empty {{ color:{MUTED}; padding:1.4rem 1.1rem; font-size:.9rem; }}
        .yf-legend {{ display:flex; flex-wrap:wrap; gap:1rem; margin-top:.5rem; }}
        .yf-legend span {{ color:{MUTED}; font-size:.78rem; font-weight:600;
            display:inline-flex; align-items:center; gap:.4rem; }}
        .yf-legend i {{ width:10px; height:10px; border-radius:3px; }}

        @media (max-width:760px) {{
            .yf-tiles, .yf-tile-row {{ grid-template-columns:repeat(2,1fr); }}
            .yf-row, .yf-head {{ grid-template-columns:1.4fr 60px 1fr; }}
            .yf-row .wtcol, .yf-head .wtcol {{ display:none; }}
        }}
        </style>
        """, unsafe_allow_html=True)


# ----------------------------------------------------------------- data
@st.cache_data(ttl=900)
def load_prices(symbol: str, lookback: int) -> pd.DataFrame:
    return get_history(symbol, lookback)


@st.cache_resource(show_spinner=False)
def fit_detector(symbol: str, lookback: int, hmm_cfg: tuple):
    prices = load_prices(symbol, lookback)
    feats = build_features(prices)
    det = RegimeDetector(**dict(hmm_cfg)).fit(feats)
    return det, prices.reindex(feats.index), det.detect_series(feats)


@st.cache_data(ttl=900, show_spinner=False)
def load_basket():
    """Target basket with per-name price, day change, weight, sparkline, and the
    momentum score + rank that justify why each name was selected."""
    from regime_trader.portfolio_trader import PortfolioTrader
    from regime_trader.broker.market_data import get_histories
    pt = PortfolioTrader()
    closes = get_histories(pt.universe + [pt.anchor], pt.train_lookback + 60).dropna(axis=1)
    stocks = [c for c in closes.columns if c != pt.anchor]
    b = pt.compute_target(closes)

    # Momentum scores across the whole universe -> the selection rationale.
    scores = pt.ranker.scores(closes[stocks]).dropna().sort_values(ascending=False)
    rank_of = {sym: i + 1 for i, sym in enumerate(scores.index)}

    rows = []
    for t in b.selected:
        s = closes[t].dropna()
        chg = float(s.iloc[-1] / s.iloc[-2] - 1) if len(s) > 1 else 0.0
        rows.append({"ticker": t, "name": NAMES.get(t, t), "weight": float(b.weights.get(t, 0.0)),
                     "price": float(s.iloc[-1]), "chg": chg, "spark": s.tail(30).tolist(),
                     "momentum": float(scores.get(t, 0.0)), "rank": rank_of.get(t, 0)})
    return b.regime, sum(b.weights.values()), rows, len(stocks)


@st.cache_data(ttl=600, show_spinner=False)
def load_challenge(anchor: str):
    """Your paper equity vs SPY since the challenge started (both rebased to 100).

    Uses Alpaca's native portfolio-history endpoint, clipped to the challenge
    start recorded in the trader state. Returns (bot, spy) normalised series or
    None if the challenge hasn't produced two data points yet.
    """
    # Demo mode: RT_DEMO_CHALLENGE=1 feeds the scoreboard sample data so you can
    # preview the populated state before a real trading day has completed.
    import os
    if os.getenv("RT_DEMO_CHALLENGE"):
        idx = pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=15)
        you = [100, 100.3, 99.8, 100.6, 101.1, 100.7, 101.5, 102.1, 101.7, 102.3,
               102.9, 102.5, 103.1, 102.2, 102.34]
        spy = [100, 100.1, 99.6, 100.0, 100.4, 100.2, 100.7, 101.1, 100.6, 101.0,
               101.4, 101.1, 101.5, 100.9, 101.10]
        return pd.Series(you, index=idx), pd.Series(spy, index=idx)

    try:
        from regime_trader.broker.alpaca_client import AlpacaClient
        client = AlpacaClient()
        if not client.is_configured:
            return None
    except Exception:
        return None

    # Challenge start, in order of preference:
    #   1. local trader state file (set on the first rebalance)
    #   2. CHALLENGE_START env / Streamlit secret
    #   3. the date of the oldest Alpaca order (works on Streamlit Cloud, which
    #      has no local state file)
    start = None
    state_path = PROJECT_ROOT / "state" / "portfolio_state.json"
    if state_path.exists():
        try:
            st_ = json.loads(state_path.read_text())
            start = st_.get("challenge_start") or st_.get("last_rebalance")
        except json.JSONDecodeError:
            pass
    if not start:
        start = os.getenv("CHALLENGE_START")
    if not start:
        try:
            oldest = client.get("/v2/orders",
                                params={"status": "all", "direction": "asc", "limit": 1})
            if oldest:
                start = oldest[0].get("submitted_at") or oldest[0].get("created_at")
        except Exception:
            pass
    if not start:
        return None

    try:
        h = client.get("/v2/account/portfolio/history",
                       params={"period": "1M", "timeframe": "1D", "extended_hours": "false"})
    except Exception:
        return None

    pairs = [(t, e) for t, e in zip(h.get("timestamp", []), h.get("equity", [])) if e]
    if len(pairs) < 2:
        return None
    bot = pd.Series([e for _, e in pairs], index=pd.to_datetime([t for t, _ in pairs], unit="s"))

    cutoff = pd.to_datetime(start)
    cutoff = (cutoff.tz_convert("UTC").tz_localize(None) if cutoff.tzinfo else cutoff).normalize()
    bot = bot[bot.index >= cutoff]
    if len(bot) < 2:
        return None

    spy = load_prices(anchor, 120)["close"]
    spy = spy[spy.index >= cutoff]
    if len(spy) < 2:
        return None
    return bot / bot.iloc[0] * 100, spy / spy.iloc[0] * 100


def fetch_live() -> dict:
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
        pos = PositionTracker(client).list_positions()
        out["positions"] = len(pos)
        out["open_pl"] = sum(p.unrealized_pl for p in pos)
    except Exception:
        pass
    return out


@st.cache_data(ttl=120, show_spinner=False)
def load_orders(limit: int = 12) -> list[dict]:
    """Recent orders straight from Alpaca, so the feed works on any host
    (the local event log only exists where the bot itself ran)."""
    try:
        from regime_trader.broker.alpaca_client import AlpacaClient
        c = AlpacaClient()
        if not c.is_configured:
            return []
        return c.get("/v2/orders", params={"status": "all", "limit": limit, "direction": "desc"}) or []
    except Exception:
        return []


def read_signal_feed(limit: int = 8) -> list[dict]:
    path = PROJECT_ROOT / "logs" / "events.jsonl"
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines()[-2000:]:
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") in {"order", "regime_change", "rebalance", "risk_check"}:
            rows.append(ev)
    return rows[-limit:][::-1]


# ----------------------------------------------------------------- helpers
def money(x):
    return f"${x:,.2f}"


def pct(x):
    return f"{x:+.2f}%"


def sparkline(values, color, w=76, h=22):
    if not values or len(values) < 2:
        return ""
    lo, hi = min(values), max(values)
    rng = (hi - lo) or 1.0
    n = len(values)
    pts = " ".join(f"{i/(n-1)*w:.1f},{h - (v-lo)/rng*(h-2) - 1:.1f}" for i, v in enumerate(values))
    return (f'<svg width="{w}" height="{h}" viewBox="0 0 {w} {h}" preserveAspectRatio="none">'
            f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6" '
            f'stroke-linejoin="round" stroke-linecap="round"/></svg>')


def tile(label, value, delta_html=""):
    return (f"<div class='yf-tile'><div class='k'>{label}</div>"
            f"<div class='v'>{value}</div>{delta_html}</div>")


# ----------------------------------------------------------------- charts
def price_chart(prices, series):
    close = prices["close"]
    rising = close.iloc[-1] >= close.iloc[0]
    line = UP if rising else DOWN
    fill = "rgba(21,199,132,0.10)" if rising else "rgba(240,97,109,0.10)"
    lo, hi = close.min(), close.max()
    pad = (hi - lo) * 0.06 or 1.0
    fig = go.Figure(go.Scatter(
        x=close.index, y=close, mode="lines", name="Close",
        line=dict(color=line, width=2), fill="tozeroy", fillcolor=fill,
        hovertemplate="%{x|%b %d, %Y}<br>$%{y:.2f}<extra></extra>"))
    # regime overlay bands
    for x0, x1, reg in _runs(series["canonical"]):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=REGIME_COLORS.get(reg, MUTED),
                      opacity=0.06, line_width=0, layer="below")
    fig.update_layout(
        height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=8, r=8, t=8, b=8), font=dict(family=FONT, color=MUTED, size=12),
        hovermode="x", hoverlabel=dict(bgcolor=SURFACE2, bordercolor=BORDER,
                                       font=dict(family=FONT, color=TEXT)), showlegend=False)
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.4, color=MUTED,
                     showspikes=True, spikecolor=MUTED, spikethickness=1, spikedash="dot")
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.4, color=MUTED,
                     tickprefix="$", range=[lo - pad, hi + pad], side="right")
    return fig


def regime_ribbon(prices, series):
    idx = prices.index
    fig = go.Figure(go.Scatter(x=[idx[0], idx[-1]], y=[0, 0], mode="lines",
                               line=dict(width=0), hoverinfo="skip", showlegend=False))
    for x0, x1, reg in _runs(series["canonical"]):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=REGIME_COLORS.get(reg, MUTED),
                      opacity=0.92, line_width=0, layer="below")
    fig.update_layout(height=30, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      margin=dict(l=8, r=8, t=0, b=0))
    fig.update_xaxes(visible=False, range=[idx[0], idx[-1]])
    fig.update_yaxes(visible=False, range=[0, 1])
    return fig


def _runs(series):
    vals, idx, runs, start = series.tolist(), series.index, [], 0
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            runs.append((idx[start], idx[i - 1], vals[start]))
            start = i
    runs.append((idx[start], idx[-1], vals[start]))
    return runs


# ----------------------------------------------------------------- view
def main():
    inject_css()
    s = load_settings()
    anchor = s.get("universe.regime_anchor", "SPY")
    lookback = s.get("hmm.train_lookback_days", 504)
    max_regimes = s.get("portfolio.regime_max_regimes", s.get("hmm.max_regimes", 5))
    hmm_cfg = (("min_regimes", s.get("hmm.min_regimes", 3)), ("max_regimes", max_regimes),
               ("covariance_type", s.get("hmm.covariance_type", "diag")),
               ("min_persistence_bars", s.get("hmm.min_persistence_bars", 3)),
               ("max_flips_in_window", s.get("hmm.max_flips_in_window", 4)),
               ("flip_window_bars", s.get("hmm.flip_window_bars", 20)))

    with st.spinner("Loading market data…"):
        det, prices, series = fit_detector(anchor, lookback, hmm_cfg)
    live = fetch_live()

    last = series.iloc[-1]
    regime, conf = str(last["canonical"]), float(last["confidence"])
    rc = REGIME_COLORS.get(regime, MUTED)

    # SPY day change
    spy_px = float(prices["close"].iloc[-1])
    spy_chg = float(prices["close"].iloc[-1] / prices["close"].iloc[-2] - 1) * 100

    # --- top bar ---
    if live["market_open"] is True:
        status, dot = "Market open", UP
    elif live["market_open"] is False:
        status, dot = "Market closed", MUTED
    else:
        status, dot = "Paper account not connected", MUTED
    st.markdown(f"""<div class="yf-bar"><div class="mark">R</div>
        <div class="name">Regime Trader</div>
        <div class="status"><span class="dot" style="background:{dot}"></span>{status}</div></div>""",
        unsafe_allow_html=True)

    # Freshness: the view updates on refresh, not live. Stamp when it loaded and
    # how current the underlying data is, then offer an explicit refresh.
    loaded = datetime.now(ZoneInfo("America/New_York"))
    through = prices.index[-1].strftime("%b %-d, %Y")
    fcol, bcol = st.columns([5, 1])
    with fcol:
        st.markdown(
            f"<div style='color:{MUTED};font-size:.8rem;margin:-.4rem 0 1rem'>"
            f"Updated <b style='color:{TEXT}'>{loaded.strftime('%b %-d, %-I:%M %p ET')}</b> · "
            f"portfolio value live from Alpaca · market data through {through} (completed daily bars)"
            f"</div>", unsafe_allow_html=True)
    with bcol:
        if st.button("↻ Refresh", use_container_width=True):
            st.cache_data.clear()
            st.cache_resource.clear()
            st.rerun()

    # --- market summary tiles ---
    acct = live["account"]
    if acct:
        pl = live["open_pl"]
        base = acct.equity - pl
        plpct = (pl / base * 100) if base else 0.0
        cls = "up" if pl >= 0 else "down"
        port_v = money(acct.equity)
        port_d = (f"<div class='d {cls}'>{'▲' if pl>=0 else '▼'} {money(abs(pl))} "
                  f"({plpct:+.2f}%)</div><div class='d sub'>Open P/L · {live['positions']} positions</div>")
    else:
        port_v, port_d = "—", "<div class='d sub'>Connect Alpaca in .env</div>"

    try:
        b_regime, gross, rows, n_universe = load_basket()
    except Exception as exc:
        b_regime, gross, rows, n_universe = regime, 0.0, [], 0

    reg_tile = tile("Detected regime",
                    f"<span style='color:{rc}'>{regime.upper()}</span>",
                    f"<div class='d sub'>{conf:.0%} confidence</div>")
    exp_tile = tile("Basket exposure", f"{gross:.0%}",
                    f"<div class='d sub'>{max(0,1-gross):.0%} cash · {len(rows)} names</div>")
    spy_cls = "up" if spy_chg >= 0 else "down"
    spy_tile = tile(f"{anchor} (S&P 500)", money(spy_px),
                    f"<div class='d {spy_cls}'>{'▲' if spy_chg>=0 else '▼'} {abs(spy_chg):.2f}%</div>")
    st.markdown("<div class='yf-tiles'>" + tile("Portfolio value", port_v, port_d)
                + reg_tile + exp_tile + spy_tile + "</div>", unsafe_allow_html=True)

    # --- challenge scoreboard: you vs S&P 500 ---
    if acct:
        ch = load_challenge(anchor)
        st.markdown("<div class='yf-h'>You vs S&amp;P 500 "
                    "<span class='sub'>30-day challenge · paper equity rebased to 100</span></div>",
                    unsafe_allow_html=True)
        if ch is None:
            st.markdown("<div class='yf-list'><div class='yf-empty'>Scoreboard builds after your "
                        "first full trading day. Orders are in — check back once the market opens "
                        "and they fill.</div></div>", unsafe_allow_html=True)
        else:
            bot_n, spy_n = ch
            lead = float(bot_n.iloc[-1] - spy_n.iloc[-1])
            lcls, lword = ("up", "ahead of") if lead >= 0 else ("down", "behind")
            st.markdown(f"<div class='yf-tile' style='margin-bottom:.6rem'>"
                        f"<div class='k'>Your edge vs S&amp;P 500</div>"
                        f"<div class='v {lcls}'>{lead:+.2f}%</div>"
                        f"<div class='d sub'>you {bot_n.iloc[-1]-100:+.2f}% · "
                        f"SPY {spy_n.iloc[-1]-100:+.2f}% · {lword} the index</div></div>",
                        unsafe_allow_html=True)
            cfig = go.Figure()
            cfig.add_trace(go.Scatter(x=bot_n.index, y=bot_n, name="You", mode="lines",
                                      line=dict(color=TEAL, width=2.4),
                                      hovertemplate="You: %{y:.2f}<extra></extra>"))
            cfig.add_trace(go.Scatter(x=spy_n.index, y=spy_n, name="S&P 500", mode="lines",
                                      line=dict(color=MUTED, width=1.8, dash="dot"),
                                      hovertemplate="SPY: %{y:.2f}<extra></extra>"))
            cfig.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=8, r=8, t=6, b=8), hovermode="x unified",
                               font=dict(family=FONT, color=MUTED, size=12),
                               legend=dict(orientation="h", y=1.12, x=0),
                               hoverlabel=dict(bgcolor=SURFACE2, bordercolor=BORDER,
                                               font=dict(family=FONT, color=TEXT)))
            cfig.update_xaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.4, color=MUTED)
            cfig.update_yaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.4, color=MUTED)
            st.plotly_chart(cfig, use_container_width=True, config={"displayModeBar": False})

    # --- price chart ---
    st.markdown(f"<div class='yf-h'>{anchor} price <span class='sub'>2-year history · "
                f"shaded by detected regime</span></div>", unsafe_allow_html=True)
    st.plotly_chart(price_chart(prices, series), use_container_width=True,
                    config={"displayModeBar": False})
    st.plotly_chart(regime_ribbon(prices, series), use_container_width=True,
                    config={"displayModeBar": False})
    present = [r for r in REGIME_COLORS if (series["canonical"] == r).any()]
    st.markdown("<div class='yf-legend'>" + "".join(
        f"<span><i style='background:{REGIME_COLORS[r]}'></i>{r}</span>" for r in present)
        + "</div>", unsafe_allow_html=True)

    # --- target basket watchlist ---
    st.markdown(f"<div class='yf-h'>Target basket <span class='sub'>momentum + regime + vol target · "
                f"regime {b_regime} · {gross:.0%} invested</span></div>", unsafe_allow_html=True)
    if not rows:
        st.markdown("<div class='yf-list'><div class='yf-empty'>Basket unavailable.</div></div>",
                    unsafe_allow_html=True)
    else:
        # Why these stocks, why this size — the selection + sizing rationale.
        each = (gross / len(rows)) if rows else 0
        st.markdown(
            f"<div style='color:{MUTED};font-size:.84rem;line-height:1.6;margin:-.2rem 0 .8rem'>"
            f"<b style='color:{TEXT}'>Why these names:</b> the {len(rows)} strongest of {n_universe} "
            f"S&amp;P stocks by <b>12-month momentum</b> (trailing return, skipping the last month to "
            f"avoid short-term reversal). <b style='color:{TEXT}'>Why this size:</b> the <b>{b_regime}</b> "
            f"regime sets total exposure to <b>{gross:.0%}</b>, then volatility targeting equal-weights "
            f"them (~{each:.1%} each) so the basket runs near its risk target.</div>",
            unsafe_allow_html=True)
        maxw = max((r["weight"] for r in rows), default=1) or 1
        head = ("<div class='yf-head'><div>Symbol · why selected</div><div>Trend</div>"
                "<div class='r'>Price · 1d</div><div class='r wtcol'>Target weight</div></div>")
        body = []
        for r in rows:
            col = UP if r["chg"] >= 0 else DOWN
            badge_bg = "rgba(21,199,132,0.14)" if r["chg"] >= 0 else "rgba(240,97,109,0.14)"
            wbar = (r["weight"] / maxw * 100) if r["weight"] else 0
            why = f"#{r['rank']} by momentum · {r['momentum']*100:+.0f}% 12-mo trend" if r.get("rank") else ""
            body.append(
                f"""<div class="yf-row">
                  <div><div class="sym">{r['ticker']}</div><div class="co">{r['name']}</div>
                    <div style="color:{TEAL};font-size:.72rem;font-weight:600;margin-top:.12rem">{why}</div></div>
                  <div>{sparkline(r['spark'], col)}</div>
                  <div class="px">{money(r['price'])}<br>
                    <span class="yf-badge" style="color:{col};background:{badge_bg}">{pct(r['chg']*100)}</span></div>
                  <div class="yf-wt wtcol"><div class="track"><div class="fill" style="width:{wbar:.0f}%"></div></div>
                    <span class="pct">{r['weight']:.1%}</span></div>
                </div>""")
        st.markdown(f"<div class='yf-list'>{head}{''.join(body)}</div>", unsafe_allow_html=True)
        if gross <= 0:
            st.caption("Regime is risk-off → the bot holds cash. Names above are the current "
                       "momentum leaders it would buy when the regime turns risk-on.")

    # --- risk controls ---
    st.markdown("<div class='yf-h'>Risk controls</div>", unsafe_allow_html=True)
    rcfg = s.get("risk", {})
    blocked = (PROJECT_ROOT / rcfg.get("block_file", "TRADING_BLOCKED")).exists()
    kill = f"<span class='down'>Blocked</span>" if blocked else f"<span class='up'>Armed</span>"
    tiles = [("Kill switch", kill, ""), ("Daily flatten", f"-{rcfg.get('daily_loss_flatten',0.03):.0%}", ""),
             ("Max drawdown stop", f"-{rcfg.get('max_drawdown_stop',0.10):.0%}", ""),
             ("Max leverage", f"{rcfg.get('max_leverage',1.5):.2f}x", "")]
    st.markdown("<div class='yf-tile-row'>" + "".join(tile(k, v, d) for k, v, d in tiles)
                + "</div>", unsafe_allow_html=True)

    # --- order activity (live from Alpaca) ---
    st.markdown("<div class='yf-h'>Order activity <span class='sub'>live from your Alpaca account</span></div>",
                unsafe_allow_html=True)
    orders = load_orders()
    if not orders:
        msg = ("No orders yet — they appear here once the bot trades."
               if acct else "Connect Alpaca to see your orders.")
        st.markdown(f"<div class='yf-list'><div class='yf-empty'>{msg}</div></div>",
                    unsafe_allow_html=True)
    else:
        body = []
        for o in orders:
            side = str(o.get("side", "")).upper()
            col = UP if side == "BUY" else DOWN
            qty = o.get("filled_qty") or o.get("qty") or ""
            title = f"{side} {qty} {o.get('symbol', '')}"
            status = o.get("status", "")
            price = o.get("filled_avg_price")
            sub = status + (f" @ {money(float(price))}" if price else "")
            ts = str(o.get("filled_at") or o.get("submitted_at") or "")[:16].replace("T", " ")
            body.append(
                f"""<div class="yf-row" style="grid-template-columns:auto 1fr auto">
                  <div style="width:6px;height:30px;border-radius:3px;background:{col}"></div>
                  <div><div class="sym" style="font-size:.9rem">{title}</div>
                    <div class="co">{sub}</div></div>
                  <div class="co">{ts}</div></div>""")
        st.markdown(f"<div class='yf-list'>{''.join(body)}</div>", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
