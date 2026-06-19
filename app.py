"""Implied Probability Density Terminal.

Streamlit UI over the 4-step Breeden-Litzenberger + CRRA pipeline.
Run with:  streamlit run app.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from datetime import datetime, timezone

from src import GAMMA
from src import data, density, rates
from src.pipeline import compute_densities, compute_surface

PLOTLY_CDN = "https://cdn.plot.ly/plotly-2.35.2.min.js"


def render_tv_chart(fig: go.Figure, bg: str = "#FFFFFF", pad: int = 46) -> None:
    """Render a Plotly figure with TradingView-style wheel interaction.

    Native Streamlit/Plotly scroll zooms both axes at once. Here we intercept
    the wheel so that, like TradingView:
      • over the price (y) axis  -> scale price only
      • over the time (x) axis   -> stretch / compress time only
      • over the chart body      -> zoom time, auto-fit price to visible candles
    Drag still pans (Plotly's dragmode='pan'); double-click resets.
    """
    height = int(fig.layout.height or 600) + pad
    fig_json = fig.to_json()
    html = """
<head><meta charset="utf-8"><script src="__CDN__"></script>
<style>html,body{margin:0;background:__BG__;overflow:hidden}#c{width:100%;height:__H__px}</style>
</head><body><div id="c"></div><script>
const fig = __FIG__;
delete fig.layout.height; delete fig.layout.width; fig.layout.autosize = true;
const gd = document.getElementById('c');
const cfg = {displaylogo:false, scrollZoom:false, responsive:true,
  modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d']};
Plotly.newPlot(gd, fig.data, fig.layout, cfg).then(() => {
  gd.addEventListener('wheel', onWheel, {passive:false, capture:true});
});
function zoomAxis(ax, name, px, f){
  const r0 = ax.r2l(ax.range[0]), r1 = ax.r2l(ax.range[1]);
  const c = ax.p2l(px);
  const o = {}; o[name+'.range'] = [ax.l2r(c-(c-r0)*f), ax.l2r(c+(r1-c)*f)];
  return o;
}
function autofitY(){
  const xa = gd._fullLayout.xaxis;
  const x0 = xa.r2l(xa.range[0]), x1 = xa.r2l(xa.range[1]);
  let lo = Infinity, hi = -Infinity;
  for (const tr of fig.data){
    if (tr.type !== 'candlestick') continue;
    for (let i=0;i<tr.x.length;i++){
      const xv = new Date(tr.x[i]).getTime();
      if (xv>=x0 && xv<=x1){ if(tr.low[i]<lo)lo=tr.low[i]; if(tr.high[i]>hi)hi=tr.high[i]; }
    }
  }
  if (lo<hi){ const p=(hi-lo)*0.08; return {'yaxis.range':[lo-p, hi+p]}; }
  return null;
}
function onWheel(e){
  e.preventDefault(); e.stopImmediatePropagation();
  const fl = gd._fullLayout, xa = fl.xaxis, ya = fl.yaxis;
  const rect = gd.getBoundingClientRect();
  const mx = e.clientX-rect.left, my = e.clientY-rect.top;
  const pL = xa._offset, pR = xa._offset+xa._length;
  const pB = ya._offset+ya._length;
  const f = e.deltaY>0 ? 1.1 : 1/1.1;
  if (my > pB){                                   // time axis -> stretch time
    Plotly.relayout(gd, zoomAxis(xa,'xaxis', mx-xa._offset, f));
  } else if (mx > pR || mx < pL){                 // price axis -> scale price
    Plotly.relayout(gd, zoomAxis(ya,'yaxis', my-ya._offset, f));
  } else {                                        // body -> zoom time, fit price
    Plotly.relayout(gd, zoomAxis(xa,'xaxis', mx-xa._offset, f)).then(()=>{
      const yf = autofitY(); if (yf) Plotly.relayout(gd, yf);
    });
  }
}
</script></body>
"""
    html = (html.replace("__CDN__", PLOTLY_CDN)
                .replace("__BG__", bg)
                .replace("__H__", str(height - 6))
                .replace("__FIG__", fig_json))
    components.html(html, height=height, scrolling=False)

# --- Theme palettes -----------------------------------------------------
# Restrained, multi-colour professional palettes (no neon, no single-hue
# dominance): green/red candles, blue risk-neutral, violet real-world,
# violet cone, cool->warm volatility surface.
FONT = "Inter, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"

THEMES = {
    "Light": dict(
        bg="#FFFFFF", panel="#F8FAFC", text="#0F172A", text_dim="#64748B",
        border="#E2E8F0", grid="#EDF1F6", accent="#2563EB",
        up="#16A34A", down="#DC2626", q="#2563EB",
        p="#7C3AED", p_fill="rgba(124,58,237,0.12)", spot="#475569",
        cone=[[0.0, "rgba(124,58,237,0.0)"], [0.3, "rgba(124,58,237,0.28)"],
              [0.7, "rgba(109,40,217,0.6)"], [1.0, "rgba(124,58,237,0.9)"]],
        surface=[[0.0, "#1D4ED8"], [0.35, "#0891B2"], [0.7, "#16A34A"],
                 [1.0, "#CA8A04"]],
    ),
    "Dark": dict(
        bg="#0F1115", panel="#171A21", text="#E6E8EB", text_dim="#9AA4B2",
        border="#2A2F3A", grid="#222733", accent="#3B82F6",
        up="#22C55E", down="#EF4444", q="#60A5FA",
        p="#A78BFA", p_fill="rgba(167,139,250,0.15)", spot="#94A3B8",
        cone=[[0.0, "rgba(167,139,250,0.0)"], [0.3, "rgba(139,92,246,0.35)"],
              [0.7, "rgba(124,58,237,0.7)"], [1.0, "rgba(167,139,250,0.95)"]],
        surface=[[0.0, "#1E3A8A"], [0.35, "#0E7490"], [0.7, "#15803D"],
                 [1.0, "#EAB308"]],
    ),
}

st.set_page_config(page_title="Implied Probability Density", layout="wide")

_title_col, _theme_col = st.columns([5, 1])
with _theme_col:
    mode = st.radio("Theme", ["Light", "Dark"], index=0, horizontal=True,
                    label_visibility="collapsed")
T = THEMES[mode]

# --- Theme CSS ----------------------------------------------------------
st.markdown(
    f"""
    <style>
    .stApp {{ background:{T['bg']}; color:{T['text']}; }}
    html, body, [class*="css"], button, input, select, textarea {{ font-family:{FONT}; }}
    h1,h2,h3,h4,h5,h6 {{ color:{T['text']}; font-weight:600; letter-spacing:0; }}
    p, span, label, .stMarkdown, [data-testid="stWidgetLabel"] {{ color:{T['text']}; }}
    .app-subtitle {{ color:{T['text_dim']}; font-size:14px; }}
    .metric-card {{ background:{T['panel']}; border:1px solid {T['border']};
        border-radius:8px; padding:10px 14px; }}
    .metric-label {{ color:{T['text_dim']}; font-size:11px; text-transform:uppercase;
        letter-spacing:.4px; margin-bottom:2px; }}
    .metric-value {{ color:{T['text']}; font-size:20px; font-weight:600; }}
    .diag-table {{ width:100%; border-collapse:collapse; }}
    .diag-table td {{ padding:6px 10px; border-bottom:1px solid {T['border']}; }}
    .diag-table td.k {{ color:{T['text_dim']}; font-size:12px; }}
    .diag-table td.v {{ color:{T['text']}; text-align:right; font-weight:600; }}
    .stTextInput input, .stNumberInput input {{ background:{T['panel']};
        color:{T['text']}; border:1px solid {T['border']}; }}
    div[data-baseweb="select"] > div {{ background:{T['panel']};
        border-color:{T['border']}; color:{T['text']}; }}
    div[data-baseweb="popover"], div[data-baseweb="menu"], ul[role="listbox"] {{
        background:{T['panel']}; color:{T['text']}; }}
    .stButton button {{ background:{T['panel']}; color:{T['text']};
        border:1px solid {T['border']}; border-radius:6px; }}
    .stButton button:hover {{ border-color:{T['accent']}; color:{T['accent']}; }}
    [data-testid="stExpander"] {{ background:{T['panel']};
        border:1px solid {T['border']}; border-radius:8px; }}
    </style>
    """,
    unsafe_allow_html=True,
)

with _title_col:
    st.markdown("# Implied Probability Density")
    st.markdown(
        f"<div class='app-subtitle'>Breeden–Litzenberger risk-neutral density "
        f"→ CRRA real-world density · γ = {GAMMA}</div>",
        unsafe_allow_html=True,
    )


# --- Formatting helpers -------------------------------------------------
def money(x: float) -> str:
    return f"${x:,.2f}"


def pct(x: float) -> str:
    return f"{x * 100:.2f}%"


# --- Cached data access -------------------------------------------------
@st.cache_data(show_spinner=False, ttl=900)
def _expirations(ticker: str):
    return data.get_expirations(ticker)


@st.cache_data(show_spinner=False, ttl=900)
def _curve():
    return rates.fetch_curve()


@st.cache_data(show_spinner=True, ttl=300)
def _compute(ticker: str, expiry: str, spline_s):
    return compute_densities(ticker, expiry, spline_s=spline_s, rate_curve=_curve())


@st.cache_data(show_spinner=False, ttl=300)
def _history(ticker: str, timeframe: str):
    return data.get_history(ticker, timeframe)


# Persist the cone for the whole app session (no ttl): it is recomputed only
# when the ticker or smoothing changes, not when the timeframe changes.
@st.cache_data(show_spinner=False)
def _surface(ticker: str, expiries: tuple[str, ...], spline_s):
    return compute_surface(ticker, list(expiries), spline_s=spline_s,
                           rate_curve=_curve())


def _days_to(expiry: str) -> float:
    d = datetime.strptime(expiry, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    return max((d - datetime.now(timezone.utc)).total_seconds() / 86400.0, 0.04)


def _cone_expiries(expiries: list[str], max_days: float = 450.0,
                   max_n: int = 12) -> list[str]:
    """Pick the expiries that build the cone (ticker-level, timeframe-agnostic):
    cap the horizon to keep the price axis sane, then evenly sample up to max_n."""
    elig = [e for e in expiries if _days_to(e) <= max_days] or expiries[:3]
    if len(elig) <= max_n:
        return elig
    idx = np.linspace(0, len(elig) - 1, max_n).round().astype(int)
    return [elig[i] for i in sorted(set(idx))]


# Initial view window per timeframe: (history days shown, future days shown).
VIEW_WINDOW = {"1H": (None, 12), "1D": (150, 150), "1W": (730, 450)}


# --- Chart builders -----------------------------------------------------
def build_price_chart(hist: pd.DataFrame, surface, timeframe: str, T: dict,
                      selected_expiry: str | None = None) -> go.Figure:
    """Candlestick history + a forward probability cone built from a term
    structure of expiries.

    Each listed expiry contributes its own density at its real calendar
    date; we interpolate between them over time, so the cloud fans out as
    uncertainty accumulates. Colour is normalised per time-slice (so the
    cone stays visible), while the hover reports the true probability.
    """
    fig = go.Figure()
    last = hist.index[-1]
    tz = hist.index.tz

    # Historical candles ------------------------------------------------
    fig.add_trace(go.Candlestick(
        x=hist.index, open=hist["Open"], high=hist["High"],
        low=hist["Low"], close=hist["Close"], name=f"{surface.ticker} {timeframe}",
        increasing=dict(line=dict(color=T["up"]), fillcolor=T["up"]),
        decreasing=dict(line=dict(color=T["down"]), fillcolor=T["down"]),
        showlegend=True,
    ))

    # Assemble the term structure, prepended with a 'now' near-delta so the
    # cone fans out from today's known price.
    price = surface.price
    sigma0 = max(surface.spot * 0.004, price[1] - price[0])
    spike = np.exp(-0.5 * ((price - surface.spot) / sigma0) ** 2)
    spike /= np.trapezoid(spike, price)

    cols = np.column_stack([spike, surface.prob])          # (M, K+1) densities

    # Knot times measured from the last candle so the cone starts at "now".
    def _tau_from_last(exp):
        return max((pd.Timestamp(exp).tz_localize(tz) - last).total_seconds()
                   / (365.0 * 86400.0), 1e-6)
    knot_tau = np.array([0.0] + [_tau_from_last(e) for e in surface.expiries])

    # Future x as business days (so weekend range-breaks leave no gaps).
    horizon = float(knot_tau.max() * 365.0)
    fut = pd.bdate_range(start=last.normalize() + pd.Timedelta(days=1),
                         end=last + pd.Timedelta(days=horizon + 1))
    if tz is not None and fut.tz is None:
        fut = fut.tz_localize(tz)
    fut_tau = np.clip((fut - last).total_seconds() / (365.0 * 86400.0), 0.0, None)

    dense = np.empty((len(price), len(fut)))
    for i in range(len(price)):
        dense[i] = np.interp(fut_tau, knot_tau, cols[i])

    prob_pct = dense * 100.0                                # true %/$1 (hover)
    col_max = dense.max(axis=0, keepdims=True)
    col_max[col_max == 0] = 1.0
    z = dense / col_max                                     # per-slice colour

    fig.add_trace(go.Heatmap(
        # Plain nested list (not a numpy array): Plotly's JSON encoder would
        # binary-encode a 2D numpy customdata, which the CDN build can't decode
        # for heatmaps, leaving the hover template unsubstituted.
        x=fut, y=price, z=z, customdata=np.round(prob_pct, 2).tolist(),
        zmin=0.0, zmax=1.0, colorscale=T["cone"],
        showscale=False, name="cone",
        hovertemplate="%{x|%Y-%m-%d}<br>Price: $%{y:.2f}"
                      "<br>Prob.: %{customdata:.2f}% per $1<extra></extra>",
    ))

    # Legend proxy (heatmaps don't appear in the legend).
    fig.add_trace(go.Scatter(
        x=[None], y=[None], mode="markers", name="Forward probability cone",
        marker=dict(color=T["p"], size=10, symbol="square"), showlegend=True,
    ))

    # Spot + NOW + per-expiry markers -----------------------------------
    fig.add_hline(y=surface.spot, line=dict(color=T["spot"], width=1, dash="dot"),
                  annotation_text=f"Spot {money(surface.spot)}",
                  annotation_position="left",
                  annotation_font=dict(color=T["spot"], size=11))
    fig.add_vline(x=last, line=dict(color=T["text_dim"], width=1, dash="dot"))

    for exp in surface.expiries:
        ed = pd.Timestamp(exp).tz_localize(tz)
        dash = "solid" if exp == selected_expiry else "dot"
        width = 1.5 if exp == selected_expiry else 0.7
        color = T["accent"] if exp == selected_expiry else T["border"]
        fig.add_vline(x=ed, line=dict(color=color, width=width, dash=dash))
    if selected_expiry in surface.expiries:
        ed = pd.Timestamp(selected_expiry).tz_localize(tz)
        fig.add_annotation(x=ed, y=price[-1], text=f"Selected · {selected_expiry}",
                           showarrow=False, font=dict(color=T["accent"], size=10),
                           xanchor="right", yanchor="top",
                           bgcolor=T["panel"])

    # Initial TradingView-style view window (user can pan/scroll beyond it).
    look_back, look_fwd = VIEW_WINDOW.get(timeframe, (150, 150))
    x0 = hist.index[0] if look_back is None else last - pd.Timedelta(days=look_back)
    x0 = max(x0, hist.index[0])
    x1 = last + pd.Timedelta(days=look_fwd)

    spike_kw = dict(showspikes=True, spikemode="across", spikesnap="cursor",
                    spikethickness=1, spikedash="dot", spikecolor=T["text_dim"])

    fig.update_layout(
        paper_bgcolor=T["bg"], plot_bgcolor=T["bg"], dragmode="pan",
        font=dict(color=T["text_dim"], family=FONT),
        title=dict(text=f"{surface.ticker} · Price & Forward Probability Cone",
                   font=dict(color=T["text"], size=16)),
        xaxis=dict(title="Date", gridcolor=T["grid"], color=T["text_dim"],
                   rangeslider=dict(visible=False), range=[x0, x1],
                   rangebreaks=[dict(bounds=["sat", "mon"])], **spike_kw),
        yaxis=dict(title="Price", gridcolor=T["grid"], color=T["text_dim"], side="right",
                   **spike_kw),
        legend=dict(bgcolor=T["panel"], bordercolor=T["border"], borderwidth=1,
                    font=dict(color=T["text"]), x=0.01, y=0.99),
        hovermode="x", height=600, margin=dict(t=60, b=40, l=60, r=70),
    )
    return fig


def build_density_chart(res, T: dict, shade: tuple | None = None) -> go.Figure:
    """Head-on view of risk-neutral q(S_T) and real-world p(S_T).

    ``shade`` = (lo, hi) optionally highlights the probability-calculator
    region (None bound = open-ended on that side).
    """
    fig = go.Figure()
    # customdata = probability (% the price lands within $1 of that level).
    fig.add_trace(go.Scatter(
        x=res.grid, y=res.q, name="q(S_T) · risk-neutral",
        line=dict(color=T["q"], width=2), customdata=res.q * 100.0,
        hovertemplate="Price: $%{x:.2f}<br>Prob. (risk-neutral): %{customdata:.2f}% per $1<extra></extra>"))
    fig.add_trace(go.Scatter(
        x=res.grid, y=res.p, name=f"p(S_T) · real-world (γ={res.gamma})",
        line=dict(color=T["p"], width=2.5),
        fill="tozeroy", fillcolor=T["p_fill"], customdata=res.p * 100.0,
        hovertemplate="Price: $%{x:.2f}<br>Prob. (real-world): %{customdata:.2f}% per $1<extra></extra>"))

    if shade is not None:
        lo, hi = shade
        x0 = res.grid[0] if lo is None else lo
        x1 = res.grid[-1] if hi is None else hi
        fig.add_vrect(x0=x0, x1=x1, fillcolor=T["p_fill"],
                      line_width=0, layer="below")

    ymax = float(max(res.q.max(), res.p.max())) * 1.1
    fig.add_shape(type="line", x0=res.spot, x1=res.spot, y0=0, y1=ymax,
                  line=dict(color=T["spot"], width=1, dash="dot"))
    fig.add_annotation(x=res.spot, y=ymax, text=f"Spot {money(res.spot)}",
                       showarrow=False, font=dict(color=T["spot"], size=11),
                       yanchor="bottom")

    spike_kw = dict(showspikes=True, spikemode="across", spikesnap="cursor",
                    spikethickness=1, spikedash="dot", spikecolor=T["text_dim"])
    fig.update_layout(
        paper_bgcolor=T["bg"], plot_bgcolor=T["bg"], dragmode="pan",
        font=dict(color=T["text_dim"], family=FONT),
        title=dict(text=f"{res.ticker} · {res.expiry} · Terminal Price Distribution",
                   font=dict(color=T["text"], size=16)),
        xaxis=dict(title="Terminal price  S_T", gridcolor=T["grid"], color=T["text_dim"],
                   **spike_kw),
        yaxis=dict(title="Probability density", gridcolor=T["grid"], color=T["text_dim"]),
        legend=dict(bgcolor=T["panel"], bordercolor=T["border"], borderwidth=1,
                    font=dict(color=T["text"]), x=0.01, y=0.99),
        hovermode="x unified", height=460, margin=dict(t=60, b=50, l=60, r=30),
    )
    return fig


def build_vol_surface(surface, T: dict) -> go.Figure:
    """3D implied-volatility surface: IV vs moneyness (K/S) vs days-to-expiry.

    Reuses the fitted smile of each cone expiry (curve.grid / curve.iv),
    resampled onto a shared moneyness grid. Points outside an expiry's traded
    strike range are left as gaps (no misleading extrapolation).
    """
    spot = surface.spot
    money_grid = np.linspace(0.80, 1.20, 60)
    days, Z = [], []
    for r in surface.results:
        moneyness = r.curve.grid / spot
        iv = np.interp(money_grid, moneyness, r.curve.iv * 100.0,
                       left=np.nan, right=np.nan)
        days.append(r.tau * 365.0)
        Z.append(iv)
    Z = np.array(Z)                      # (n_expiry, n_moneyness)

    cmin = float(np.nanpercentile(Z, 2))
    cmax = float(np.nanpercentile(Z, 98))
    fig = go.Figure(go.Surface(
        x=money_grid, y=np.array(days), z=Z, connectgaps=False,
        cmin=cmin, cmax=cmax, colorscale=T["surface"],
        colorbar=dict(title=dict(text="IV %", font=dict(color=T["text_dim"])),
                      tickfont=dict(color=T["text_dim"]), outlinecolor=T["border"],
                      len=0.6),
        lighting=dict(ambient=0.7, diffuse=0.6, specular=0.1),
        hovertemplate="Moneyness K/S: %{x:.2f}<br>Days: %{y:.0f}"
                      "<br>IV: %{z:.1f}%<extra></extra>",
    ))

    axis_kw = dict(backgroundcolor=T["panel"], gridcolor=T["grid"],
                   zerolinecolor=T["grid"], color=T["text_dim"], showbackground=True)
    fig.update_layout(
        paper_bgcolor=T["bg"], font=dict(color=T["text_dim"], family=FONT),
        title=dict(text=f"{surface.ticker} · Implied Volatility Surface",
                   font=dict(color=T["text"], size=16)),
        scene=dict(
            xaxis=dict(title="Moneyness K/S", **axis_kw),
            yaxis=dict(title="Days to expiry", **axis_kw),
            zaxis=dict(title="Implied vol %", **axis_kw),
            bgcolor=T["bg"],
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.8)),
            aspectratio=dict(x=1, y=1.3, z=0.7),
        ),
        height=640, margin=dict(t=60, b=10, l=10, r=10),
    )
    return fig


# --- Controls -----------------------------------------------------------
c1, c2, c3, c4, c5 = st.columns([1.2, 1.5, 1.1, 1.7, 0.8])
with c1:
    ticker = st.text_input("Ticker", value="SPY",
                           help="Any symbol with listed options (SPY, AAPL, QQQ…)").strip().upper()

expiry = None
if ticker:
    try:
        exps = _expirations(ticker)
        with c2:
            expiry = st.selectbox("Expiration", exps, index=0)
    except Exception as e:
        st.error(f"Could not load expirations for '{ticker}': {e}")

with c3:
    timeframe = st.selectbox("Timeframe", list(data.TIMEFRAMES.keys()), index=1)
with c4:
    smooth_pct = st.slider("Smile smoothing", 0.0, 100.0, 50.0, 5.0,
                           help="Higher = smoother spline (less noise, more bias).")
with c5:
    st.write("")
    st.write("")
    if st.button("Refresh", help="Re-pull live data and rebuild the cone."):
        st.cache_data.clear()
        st.rerun()

st.divider()

# --- Compute & render ---------------------------------------------------
if ticker and expiry:
    try:
        spline_s = None if smooth_pct == 50.0 else float(smooth_pct) / 50.0
        res = _compute(ticker, expiry, spline_s)
    except Exception as e:
        st.error(f"Pipeline failed: {e}")
        st.stop()

    s = res.summary

    # Stat strip --------------------------------------------------------
    cards = [
        ("Spot", money(s["spot"])),
        ("Forward", money(s["forward"])),
        ("Risk-free r(τ)", pct(s["r"])),
        ("Days to expiry", f"{s['tau_days']:.1f}"),
        ("Risk-neutral mean", money(s["q_mean"])),
        ("Real-world mean", money(s["p_mean"])),
        ("Risk aversion γ", f"{s['gamma']:.1f}"),
    ]
    cols = st.columns(len(cards))
    for col, (label, val) in zip(cols, cards):
        col.markdown(
            f"<div class='metric-card'><div class='metric-label'>{label}</div>"
            f"<div class='metric-value'>{val}</div></div>",
            unsafe_allow_html=True,
        )

    st.write("")

    # Forward probability cone (computed once per ticker; timeframe only
    # changes the candles and the view window, never recomputes the cone).
    surface = None
    try:
        cone_exps = _cone_expiries(exps)
        surface = _surface(ticker, tuple(cone_exps), spline_s)
        hist = _history(ticker, timeframe)
        render_tv_chart(
            build_price_chart(hist, surface, timeframe, T, selected_expiry=expiry),
            bg=T["bg"])
        st.caption(
            f"Cone built from {len(surface.expiries)} expiries "
            f"({surface.expiries[0]} → {surface.expiries[-1]}). Scroll over the "
            f"body to zoom time (price auto-fits); scroll over the price/time axis "
            f"to scale just that axis; drag to pan; double-click to reset."
        )
    except Exception as e:
        st.warning(f"Forward cone unavailable: {e}")

    # Probability calculator -------------------------------------------
    shade = None
    if surface is not None:
        st.markdown("#### Probability Calculator")
        qa, qb, qc = st.columns([2.2, 1.3, 2.5])
        with qa:
            default_date = min(surface.expiries,
                               key=lambda e: abs(_days_to(e) - _days_to(expiry)))
            sel_dates = st.multiselect("Date(s)", surface.expiries,
                                       default=[default_date],
                                       format_func=lambda e: f"{e}  ({_days_to(e):.0f}d)")
        with qb:
            use_p = st.radio("Measure", ["Real-world (p)", "Risk-neutral (q)"],
                             index=0) == "Real-world (p)"
            qtype = st.radio("Query", ["Between", "Below", "Above"], index=0)
        with qc:
            spot = surface.spot
            if qtype == "Between":
                lo = st.number_input("Lower price", value=round(spot * 0.95, 2), step=1.0)
                hi = st.number_input("Upper price", value=round(spot * 1.05, 2), step=1.0)
                bounds, label = (lo, hi), f"between ${lo:,.2f} and ${hi:,.2f}"
            elif qtype == "Below":
                x = st.number_input("Price", value=round(spot * 0.95, 2), step=1.0)
                bounds, label = (None, x), f"below ${x:,.2f}"
            else:
                y = st.number_input("Price", value=round(spot * 1.05, 2), step=1.0)
                bounds, label = (y, None), f"above ${y:,.2f}"

        shade = bounds
        by_expiry = {r.expiry: r for r in surface.results}
        rows = []
        for d in sel_dates:
            r = by_expiry.get(d)
            if r is None:
                continue
            dens = r.p if use_p else r.q
            prob = density.interval_probability(r.grid, dens, bounds[0], bounds[1])
            rows.append((d, f"{_days_to(d):.0f}", f"{prob * 100:.1f}%"))

        if rows:
            measure_lbl = "real-world" if use_p else "risk-neutral"
            head = (f"<tr><td class='k'>Expiry</td><td class='k'>Days</td>"
                    f"<td class='v'>P({measure_lbl}) {label}</td></tr>")
            body = "".join(f"<tr><td class='k'>{d}</td><td class='k'>{n}</td>"
                           f"<td class='v'>{p}</td></tr>" for d, n, p in rows)
            st.markdown(f"<table class='diag-table'>{head}{body}</table>",
                        unsafe_allow_html=True)
        else:
            st.caption("Pick at least one date to compute a probability.")

    # Density chart (shaded by the calculator selection) ---------------
    render_tv_chart(build_density_chart(res, T, shade=shade), bg=T["bg"])

    # 3D implied-volatility surface ------------------------------------
    if surface is not None and len(surface.results) >= 2:
        st.markdown("#### Implied Volatility Surface")
        st.plotly_chart(build_vol_surface(surface, T), width="stretch",
                        config={"scrollZoom": True, "displaylogo": False})
        st.caption(
            "Fitted IV across moneyness (K/S) and maturity, from the same "
            "expiries that build the cone. Drag to rotate, scroll to zoom; "
            "gaps are strikes that don't trade at that maturity."
        )

    # Diagnostics -------------------------------------------------------
    with st.expander("Diagnostics"):
        rows = {
            "Strikes used": f"{s['n_strikes']}",
            "Risk-neutral std dev": money(s["q_std"]),
            "Real-world std dev": money(s["p_std"]),
            "q(S) integrates to": f"{s['q_area']:.4f}",
            "p(S) integrates to": f"{s['p_area']:.4f}",
            "Risk premium (p − q mean)": money(s["p_mean"] - s["q_mean"]),
        }
        table = "".join(
            f"<tr><td class='k'>{k}</td><td class='v'>{v}</td></tr>"
            for k, v in rows.items()
        )
        st.markdown(f"<table class='diag-table'>{table}</table>",
                    unsafe_allow_html=True)
        st.caption(
            "Sanity checks: q-mean should sit near the forward. The CRRA reweight "
            "p ∝ S^γ·q lifts the right tail, so the real-world mean sits above the "
            "forward — this gap is the equity risk premium."
        )
else:
    st.info("Enter a ticker with listed options (e.g. SPY, AAPL, QQQ) to begin.")
