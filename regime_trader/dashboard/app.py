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

# ------------------------------------------------- palette (layered for depth)
PAGE = "#080b10"        # deepest: page background
SHELL = "#0f141b"       # outer bezel "tray" around cards
SURFACE = "#171d26"     # inner card "plate"
SURFACE2 = "#212a35"    # tracks / hover lift
BORDER = "#29323d"      # hairline
HAIR = "rgba(255,255,255,0.06)"   # top-edge highlight (machined look)
TEXT = "#eef1f5"
MUTED = "#929caa"       # muted slate
UP = "#15c784"          # gains
DOWN = "#f0616d"        # losses
TEAL = "#16c4a8"        # brand accent (teal-green)
FONT = "'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif"

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
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');

        /* deep page with a faint top glow for spatial depth */
        .stApp {{ background:
            radial-gradient(1100px 480px at 78% -8%, rgba(22,196,168,0.07), transparent 60%),
            radial-gradient(900px 420px at 10% -10%, rgba(60,90,160,0.06), transparent 55%),
            {PAGE}; }}
        html, body, [data-testid="stAppViewContainer"], [class*="css"] {{
            font-family:{FONT}; color:{TEXT}; }}
        [data-testid="stHeader"] {{ background:transparent; }}
        #MainMenu, footer, [data-testid="stToolbar"] {{ visibility:hidden; }}
        .block-container {{ max-width:1140px; padding-top:1.8rem; padding-bottom:5rem; }}
        ::selection {{ background:rgba(22,196,168,0.28); }}

        .yf-bar {{ display:flex; align-items:center; gap:.6rem; margin-bottom:.5rem; }}
        .yf-bar .mark {{ width:30px; height:30px; border-radius:9px;
            background:linear-gradient(150deg,{TEAL},#0c9c84); display:flex; align-items:center;
            justify-content:center; color:#04201b; font-weight:800; font-size:16px;
            box-shadow:inset 0 1px 0 rgba(255,255,255,0.4), 0 4px 14px -4px rgba(22,196,168,0.5); }}
        .yf-bar .name {{ font-weight:800; font-size:1.06rem; letter-spacing:-.01em; }}
        .yf-bar .status {{ margin-left:auto; font-size:.76rem; color:{MUTED}; font-weight:600;
            display:flex; align-items:center; gap:.45rem; padding:.32rem .7rem; border-radius:999px;
            background:{SHELL}; border:1px solid {BORDER}; }}
        .yf-bar .dot {{ width:7px; height:7px; border-radius:50%; box-shadow:0 0 7px currentColor; }}

        /* ---- Double-bezel: outer tray (shell) hugging an inner plate (core) ---- */
        .yf-tile {{ background:{SHELL}; border:1px solid {BORDER}; border-radius:18px; padding:5px;
            display:flex;
            box-shadow:0 1px 1px rgba(0,0,0,0.5), 0 16px 34px -16px rgba(0,0,0,0.7);
            transition:transform .5s cubic-bezier(.32,.72,0,1), box-shadow .5s cubic-bezier(.32,.72,0,1); }}
        .yf-tile:hover {{ transform:translateY(-2px);
            box-shadow:0 1px 1px rgba(0,0,0,0.5), 0 22px 44px -18px rgba(0,0,0,0.8); }}
        .yf-core {{ flex:1; background:linear-gradient(180deg, rgba(255,255,255,0.025), rgba(255,255,255,0));
            background-color:{SURFACE}; border-radius:14px; padding:.95rem 1.05rem;
            box-shadow:inset 0 1px 0 {HAIR}; }}
        .yf-tiles {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; margin-bottom:.4rem; }}
        .yf-tile-row {{ display:grid; grid-template-columns:repeat(4,1fr); gap:.7rem; }}
        .yf-tile .k {{ color:{MUTED}; font-size:.7rem; font-weight:700; letter-spacing:.12em;
            text-transform:uppercase; }}
        .yf-tile .v {{ font-size:1.55rem; font-weight:800; margin-top:.35rem; letter-spacing:-.025em;
            font-variant-numeric:tabular-nums; line-height:1.08; }}
        .yf-tile .d {{ font-size:.82rem; font-weight:700; margin-top:.25rem; font-variant-numeric:tabular-nums; }}
        .yf-tile .d.sub {{ color:{MUTED}; font-weight:600; }}
        .up {{ color:{UP}; }} .down {{ color:{DOWN}; }}

        /* section headers with a small accent tick */
        .yf-h {{ font-size:1.06rem; font-weight:700; margin:2.4rem 0 .8rem; letter-spacing:-.01em;
            display:flex; align-items:baseline; gap:.55rem; }}
        .yf-h::before {{ content:""; width:3px; height:15px; border-radius:2px; background:{TEAL};
            align-self:center; box-shadow:0 0 8px rgba(22,196,168,0.6); }}
        .yf-h .sub {{ color:{MUTED}; font-weight:500; font-size:.82rem; }}

        /* lists: shell tray wrapping an inner plate */
        .yf-list {{ background:{SHELL}; border:1px solid {BORDER}; border-radius:18px; padding:5px;
            box-shadow:0 1px 1px rgba(0,0,0,0.5), 0 16px 34px -16px rgba(0,0,0,0.7); }}
        .yf-list-core {{ background:{SURFACE}; border-radius:14px; overflow:hidden;
            box-shadow:inset 0 1px 0 {HAIR}; }}
        .yf-row {{ display:grid; grid-template-columns:1.7fr 84px 1.1fr 1.4fr; align-items:center;
            gap:.6rem; padding:.78rem 1.05rem; border-top:1px solid {BORDER};
            transition:background .35s cubic-bezier(.32,.72,0,1); }}
        .yf-row:first-child {{ border-top:none; }}
        .yf-row:hover {{ background:rgba(255,255,255,0.022); }}
        .yf-row .sym {{ font-weight:700; font-size:.95rem; letter-spacing:-.01em; }}
        .yf-row .co {{ color:{MUTED}; font-size:.76rem; white-space:nowrap; overflow:hidden;
            text-overflow:ellipsis; }}
        .yf-row .px {{ text-align:right; font-weight:700; font-variant-numeric:tabular-nums; font-size:.92rem; }}
        .yf-badge {{ display:inline-block; min-width:62px; text-align:center; padding:.2rem .45rem;
            border-radius:7px; font-size:.78rem; font-weight:700; font-variant-numeric:tabular-nums; }}
        .yf-wt {{ display:flex; align-items:center; gap:.55rem; justify-content:flex-end; }}
        .yf-wt .track {{ flex:1; max-width:120px; height:7px; border-radius:5px; background:{SURFACE2};
            overflow:hidden; box-shadow:inset 0 1px 2px rgba(0,0,0,0.4); }}
        .yf-wt .fill {{ height:100%; border-radius:5px;
            background:linear-gradient(90deg,{TEAL},#1fe0bd); }}
        .yf-wt .pct {{ font-size:.82rem; font-weight:700; color:{TEXT}; font-variant-numeric:tabular-nums;
            min-width:42px; text-align:right; }}
        .yf-head {{ display:grid; grid-template-columns:1.7fr 84px 1.1fr 1.4fr; gap:.6rem;
            padding:.62rem 1.05rem .5rem; color:{MUTED}; font-size:.68rem; font-weight:700;
            letter-spacing:.1em; text-transform:uppercase; }}
        .yf-head .r {{ text-align:right; }}
        .yf-empty {{ color:{MUTED}; padding:1.5rem 1.2rem; font-size:.9rem; }}

        .yf-legend {{ display:flex; flex-wrap:wrap; gap:1.1rem; margin-top:.7rem; padding-left:.2rem; }}
        .yf-legend span {{ color:{MUTED}; font-size:.78rem; font-weight:600;
            display:inline-flex; align-items:center; gap:.4rem; }}
        .yf-legend i {{ width:10px; height:10px; border-radius:3px; }}

        /* refresh button -> pill that matches the bezels */
        .stButton > button {{ background:{SHELL}; color:{TEXT}; border:1px solid {BORDER};
            border-radius:999px; font-weight:600; font-size:.82rem; padding:.4rem .9rem;
            box-shadow:inset 0 1px 0 {HAIR}; transition:transform .4s cubic-bezier(.32,.72,0,1),
            background .4s ease; }}
        .stButton > button:hover {{ background:{SURFACE2}; border-color:{TEAL}; transform:translateY(-1px); }}
        .stButton > button:active {{ transform:scale(.97); }}

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

    # Rank reflects the actual selection score (risk-adjusted if A2 is on); the
    # displayed momentum is the plain trailing 12-1 return, which reads cleanly.
    scores = pt.ranker.scores(closes[stocks]).dropna().sort_values(ascending=False)
    rank_of = {sym: i + 1 for i, sym in enumerate(scores.index)}
    lb, sk = pt.ranker.lookback, pt.ranker.skip
    raw_ret = (closes[stocks].iloc[-1 - sk] / closes[stocks].iloc[-1 - lb] - 1) \
        if len(closes) > lb else None

    rows = []
    for t in b.selected:
        s = closes[t].dropna()
        chg = float(s.iloc[-1] / s.iloc[-2] - 1) if len(s) > 1 else 0.0
        mom = float(raw_ret.get(t, 0.0)) if raw_ret is not None else 0.0
        rows.append({"ticker": t, "name": NAMES.get(t, t), "weight": float(b.weights.get(t, 0.0)),
                     "price": float(s.iloc[-1]), "chg": chg, "spark": s.tail(30).tolist(),
                     "momentum": mom, "rank": rank_of.get(t, 0)})
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
    return (f"<div class='yf-tile'><div class='yf-core'>"
            f"<div class='k'>{label}</div><div class='v'>{value}</div>{delta_html}"
            f"</div></div>")


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
    # regime overlay bands (smoothed for a clean, organized look)
    for x0, x1, reg in _smooth_runs(series["canonical"]):
        fig.add_vrect(x0=x0, x1=x1, fillcolor=REGIME_COLORS.get(reg, MUTED),
                      opacity=0.08, line_width=0, layer="below")
    fig.update_layout(
        height=330, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=52, t=8, b=8), font=dict(family=FONT, color=MUTED, size=12),
        hovermode="x", hoverdistance=-1, spikedistance=-1,
        hoverlabel=dict(bgcolor=SURFACE2, bordercolor=BORDER,
                        font=dict(family=FONT, color=TEXT)), showlegend=False)
    # spikesnap="cursor" -> the crosshair follows the mouse fluidly instead of
    # jumping between data points; hoverdistance=-1 keeps the tooltip always on.
    fig.update_xaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.4, color=MUTED,
                     showspikes=True, spikecolor=TEXT, spikethickness=1, spikedash="dot",
                     spikemode="across", spikesnap="cursor")
    fig.update_yaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.4, color=MUTED,
                     tickprefix="$", range=[lo - pad, hi + pad], side="right")
    return fig


def regime_ribbon(prices, series) -> str:
    """Clean proportional regime strip as HTML: flush segments, identical height,
    rounded ends, no gaps. (Plotly vrects leave slivers and subpixel height jitter.)
    Side margins match the price chart's plot area so it lines up underneath."""
    runs = _smooth_runs(series["canonical"])
    segs = []
    for x0, x1, reg in runs:
        weight = max((x1 - x0).days, 1)   # width proportional to duration
        col = REGIME_COLORS.get(reg, MUTED)
        segs.append(f"<div style='flex:{weight} 1 0;background:{col}'></div>")
    return ("<div style='display:flex;height:20px;border-radius:7px;overflow:hidden;"
            "margin:8px 52px 2px 10px;box-shadow:inset 0 1px 0 rgba(255,255,255,0.12),"
            "0 2px 6px rgba(0,0,0,0.45)'>" + "".join(segs) + "</div>")


def _runs(series):
    vals, idx, runs, start = series.tolist(), series.index, [], 0
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            runs.append((idx[start], idx[i - 1], vals[start]))
            start = i
    runs.append((idx[start], idx[-1], vals[start]))
    return runs


def _smooth_runs(series, min_len: int = 11):
    """Contiguous regime blocks for display, with short flickers (< min_len bars)
    absorbed into the preceding block so the ribbon reads as clean, coherent
    bands instead of a cluttered barcode. Display-only — never affects trading."""
    vals, idx = series.tolist(), series.index
    if not vals:
        return []
    runs, start = [], 0
    for i in range(1, len(vals)):
        if vals[i] != vals[i - 1]:
            runs.append([start, i - 1, vals[start]])
            start = i
    runs.append([start, len(vals) - 1, vals[start]])
    merged = []
    for r in runs:
        if merged and (r[1] - r[0] + 1) < min_len:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    return [(idx[a], idx[b], lab) for a, b, lab in merged]


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
            st.markdown("<div class='yf-list'><div class='yf-list-core'><div class='yf-empty'>"
                        "Scoreboard builds after your first full trading day. Orders are in — check "
                        "back once the market opens and they fill.</div></div></div>",
                        unsafe_allow_html=True)
        else:
            bot_n, spy_n = ch
            lead = float(bot_n.iloc[-1] - spy_n.iloc[-1])
            lcls, lword = ("up", "ahead of") if lead >= 0 else ("down", "behind")
            st.markdown(f"<div class='yf-tile' style='margin-bottom:.6rem'><div class='yf-core'>"
                        f"<div class='k'>Your edge vs S&amp;P 500</div>"
                        f"<div class='v {lcls}'>{lead:+.2f}%</div>"
                        f"<div class='d sub'>you {bot_n.iloc[-1]-100:+.2f}% · "
                        f"SPY {spy_n.iloc[-1]-100:+.2f}% · {lword} the index</div></div></div>",
                        unsafe_allow_html=True)
            cfig = go.Figure()
            cfig.add_trace(go.Scatter(x=bot_n.index, y=bot_n, name="You", mode="lines",
                                      line=dict(color=TEAL, width=2.4),
                                      hovertemplate="You: %{y:.2f}<extra></extra>"))
            cfig.add_trace(go.Scatter(x=spy_n.index, y=spy_n, name="S&P 500", mode="lines",
                                      line=dict(color=MUTED, width=1.8, dash="dot"),
                                      hovertemplate="SPY: %{y:.2f}<extra></extra>"))
            cfig.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                               margin=dict(l=10, r=10, t=6, b=8), hovermode="x unified",
                               hoverdistance=-1, spikedistance=-1,
                               font=dict(family=FONT, color=MUTED, size=12),
                               legend=dict(orientation="h", y=1.12, x=0),
                               hoverlabel=dict(bgcolor=SURFACE2, bordercolor=BORDER,
                                               font=dict(family=FONT, color=TEXT)))
            cfig.update_xaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.4, color=MUTED,
                              showspikes=True, spikecolor=TEXT, spikethickness=1, spikedash="dot",
                              spikemode="across", spikesnap="cursor")
            cfig.update_yaxes(showgrid=True, gridcolor=BORDER, gridwidth=0.4, color=MUTED)
            st.plotly_chart(cfig, use_container_width=True, config={"displayModeBar": False})

    # --- price chart ---
    st.markdown(f"<div class='yf-h'>{anchor} price <span class='sub'>regime-detection window · "
                f"shaded by detected regime</span></div>", unsafe_allow_html=True)
    st.plotly_chart(price_chart(prices, series), use_container_width=True,
                    config={"displayModeBar": False})
    st.markdown(regime_ribbon(prices, series), unsafe_allow_html=True)
    present = [r for r in REGIME_COLORS if (series["canonical"] == r).any()]
    st.markdown("<div class='yf-legend'>" + "".join(
        f"<span><i style='background:{REGIME_COLORS[r]}'></i>{r}</span>" for r in present)
        + "</div>", unsafe_allow_html=True)

    # --- target basket watchlist ---
    st.markdown(f"<div class='yf-h'>Target basket <span class='sub'>momentum + regime + vol target · "
                f"regime {b_regime} · {gross:.0%} invested</span></div>", unsafe_allow_html=True)
    if not rows:
        st.markdown("<div class='yf-list'><div class='yf-list-core'><div class='yf-empty'>"
                    "Basket unavailable.</div></div></div>", unsafe_allow_html=True)
    else:
        # Why these stocks, why this size — the selection + sizing rationale.
        each = (gross / len(rows)) if rows else 0
        st.markdown(
            f"<div style='color:{MUTED};font-size:.84rem;line-height:1.6;margin:-.2rem 0 .8rem'>"
            f"<b style='color:{TEXT}'>Why these names:</b> the {len(rows)} strongest of our "
            f"{n_universe}-stock S&amp;P large-cap watchlist by <b>risk-adjusted 12-month momentum</b> "
            f"(trailing return per unit of volatility, skipping the last month). "
            f"<b style='color:{TEXT}'>Why this size:</b> <b>volatility targeting</b> sets total "
            f"exposure to <b>{gross:.0%}</b> and equal-weights the basket (~{each:.1%} each) so it runs "
            f"near its risk target; the hard -10% drawdown breaker is the safety net. "
            f"(Regime now shown for information only, not gating.)</div>",
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
        st.markdown(f"<div class='yf-list'><div class='yf-list-core'>{head}{''.join(body)}</div></div>",
                    unsafe_allow_html=True)
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
        st.markdown(f"<div class='yf-list'><div class='yf-list-core'><div class='yf-empty'>{msg}"
                    f"</div></div></div>", unsafe_allow_html=True)
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
        st.markdown(f"<div class='yf-list'><div class='yf-list-core'>{''.join(body)}</div></div>",
                    unsafe_allow_html=True)


if __name__ == "__main__":
    main()
