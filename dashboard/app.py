"""
dashboard/app.py — NYC Taxi Demand Forecasting
Run: streamlit run dashboard/app.py
"""

import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).parent.parent))
from src.predict import DemandPredictor

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NYC Taxi · Demand Intelligence",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Inject minimal CSS (kept very short to avoid rendering bug) ────────────────
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap" rel="stylesheet">
    <style>
    html, body, [class*="css"] { font-family: 'Inter', sans-serif !important; }
    h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; }
    .block-container { padding-top: 1.5rem !important; padding-bottom: 3rem !important; }
    #MainMenu, footer { visibility: hidden; }
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1rem 1.2rem;
        box-shadow: 0 1px 3px rgba(0,0,0,.06);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #1a56db;
        box-shadow: 0 4px 12px rgba(26,86,219,.1);
        transition: all .2s;
    }
    div[data-testid="stMetricLabel"] { font-size: .75rem !important; font-weight: 600 !important; text-transform: uppercase; letter-spacing: .05em; color: #64748b !important; }
    div[data-testid="stMetricValue"] { font-family: 'Space Grotesk', sans-serif !important; font-size: 1.9rem !important; color: #1a56db !important; }
    div[data-testid="stMetricDelta"] { font-size: .78rem !important; }
    section[data-testid="stSidebar"] { background: #0f172a !important; }
    section[data-testid="stSidebar"] * { color: #cbd5e1 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] .stSlider label,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stDateInput label { color: #94a3b8 !important; font-size: .72rem !important; font-weight: 600 !important; letter-spacing: .06em !important; text-transform: uppercase !important; }
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] > div,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] span,
    section[data-testid="stSidebar"] .stSelectbox div[data-baseweb="select"] input,
    section[data-testid="stSidebar"] .stDateInput input,
    section[data-testid="stSidebar"] .stNumberInput input { color: #f1f5f9 !important; background: #1e293b !important; border-color: #334155 !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] { background: #1e293b !important; }
    section[data-testid="stSidebar"] [data-baseweb="select"] * { color: #f1f5f9 !important; }
    section[data-testid="stSidebar"] input { color: #f1f5f9 !important; }
    section[data-testid="stSidebar"] .stButton > button { background: #1a56db !important; color: white !important; border: none !important; border-radius: 10px !important; width: 100% !important; font-weight: 600 !important; font-size: .88rem !important; padding: .65rem !important; box-shadow: 0 4px 14px rgba(26,86,219,.35) !important; }
    section[data-testid="stSidebar"] .stButton > button:hover { background: #1e40af !important; transform: translateY(-1px) !important; }
    .stTabs [data-baseweb="tab-list"] { gap: 4px; background: #f1f5f9; padding: 4px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { border-radius: 8px; font-weight: 600; font-size: .83rem; color: #64748b; padding: 6px 18px; }
    .stTabs [aria-selected="true"] { background: #1a56db !important; color: white !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Plotly helpers ─────────────────────────────────────────────────────────────
BLUE   = "#1a56db"
INDIGO = "#4f46e5"
TEAL   = "#0d9488"
AMBER  = "#d97706"
RED    = "#dc2626"
GREEN  = "#16a34a"
PAL    = [BLUE, TEAL, AMBER, INDIGO, RED, GREEN]

def chart_layout(title="", xtitle="", ytitle="", height=360, **kw):
    base = dict(
        title=dict(text=title, font=dict(family="Space Grotesk,sans-serif",
                   size=14, color="#0f172a"), x=0, xanchor="left"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#fafafa",
        font=dict(family="Inter,sans-serif", color="#64748b", size=11),
        height=height,
        margin=dict(l=8, r=8, t=44, b=8),
        xaxis=dict(title=xtitle, gridcolor="#f1f5f9", linecolor="#e2e8f0",
                   tickfont=dict(size=11), zeroline=False),
        yaxis=dict(title=ytitle, gridcolor="#f1f5f9", linecolor="#e2e8f0",
                   tickfont=dict(size=11), zeroline=False),
        legend=dict(bgcolor="rgba(255,255,255,.9)", bordercolor="#e2e8f0",
                    borderwidth=1, font=dict(size=11)),
        hoverlabel=dict(bgcolor="white", font_size=12,
                        font_family="Inter,sans-serif",
                        bordercolor="#e2e8f0"),
    )
    base.update(kw)
    return base

# ── Loaders ────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_predictor():
    try:
        return DemandPredictor(), None
    except FileNotFoundError as e:
        return None, str(e)

@st.cache_data
def load_eda():
    p = Path(__file__).parent.parent / "data" / "cache" / "eda_stats.npz"
    if not p.exists():
        return None
    raw = np.load(p, allow_pickle=True)
    return {k: raw[k] for k in raw.files}

predictor, load_err = load_predictor()
eda = load_eda()

# ══════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🚕 NYC Taxi")
    st.markdown("**Demand Intelligence**")
    st.caption("ML Dashboard · XGBoost · v1.0")
    st.divider()

    st.markdown("**Forecast Settings**")
    zones = sorted(predictor.valid_zones) if predictor else list(range(1, 266))
    zone_id     = st.selectbox("Pickup Zone", zones,
                               index=zones.index(161) if 161 in zones else 0)
    target_date = st.date_input("Date", value=datetime.now().date())
    target_hour = st.slider("Start Hour", 0, 23, 8, format="%d:00")
    hours_ahead = st.slider("Hours to Forecast", 1, 48, 24)

    st.markdown("**Trip Parameters**")
    avg_fare = st.number_input("Avg Fare ($)", 5.0, 100.0, 15.0, 0.5)
    avg_dist = st.number_input("Avg Distance (mi)", 0.5, 20.0, 2.5, 0.1)

    st.divider()
    predict_btn = st.button("🔮 Generate Forecast", type="primary")

    if predictor:
        info = predictor.model_info
        st.markdown("**Model Stats**")
        st.markdown(f"""
        <div style="font-size:.78rem;line-height:2;color:#94a3b8;font-family:monospace">
        TYPE &nbsp; XGBoost<br>
        R² &nbsp;&nbsp;&nbsp;&nbsp; {info.get('test_r2',0):.4f}<br>
        MAE &nbsp;&nbsp; {info.get('test_mae',0):.2f} trips/hr<br>
        RMSE &nbsp; {info.get('test_rmse',0):.2f}<br>
        ZONES &nbsp;{info.get('zones_covered',0)}<br>
        DATA &nbsp; 2024 → 2026
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    st.markdown("""
    <div style="font-size:.78rem">
    <a href="https://github.com/IshanGupta09" target="_blank"
       style="color:#60a5fa;text-decoration:none;display:block;padding:3px 0">
       🐙 IshanGupta09
    </a>
    <a href="https://www.linkedin.com/in/ishan-gupta091/" target="_blank"
       style="color:#60a5fa;text-decoration:none;display:block;padding:3px 0">
       💼 ishan-gupta091
    </a>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown("# 🚕 NYC Taxi Demand Forecasting")
    st.markdown(
        "XGBoost model trained on **76M+ trips** across 26 months of NYC TLC data. "
        "Predicts hourly pickup demand per zone."
    )
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.info("📅 Data: Jan 2024 – Feb 2026")

st.divider()

# ── KPI metrics ────────────────────────────────────────────────────────────────
if predictor:
    info = predictor.model_info
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test R²",       f"{info.get('test_r2',0):.4f}",    "Model accuracy")
    c2.metric("Test MAE",      f"{info.get('test_mae',0):.2f}",   "trips / hour")
    c3.metric("Test RMSE",     f"{info.get('test_rmse',0):.2f}",  "root mean sq error")
    c4.metric("Zones Covered", f"{info.get('zones_covered',0)}",  "NYC TLC zones")
    c5.metric("Training Data", "76M+",                             "trips · 26 months")
    st.markdown("<br>", unsafe_allow_html=True)
else:
    # Demo mode KPIs — show expected values from training
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Test R²",       "0.9602", "Model accuracy")
    c2.metric("Test MAE",      "7.34",   "trips / hour")
    c3.metric("Test RMSE",     "13.94",  "root mean sq error")
    c4.metric("Zones Covered", "89",     "NYC TLC zones")
    c5.metric("Training Data", "76M+",   "trips · 26 months")
    st.info("📌 **Option A — Demo Mode:** Charts load from EDA cache. Commit model files to enable live predictions.")
    st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮  Forecast",
    "📊  Demand Patterns",
    "📈  Year-over-Year",
    "🗺️  Zone Rankings",
])

# ────────────────────────────────────────────────────────────────
# TAB 1 · FORECAST
# ────────────────────────────────────────────────────────────────
with tab1:
    if not predictor:
        # ── OPTION A: Demo mode — model not committed to repo ──────────────
        st.info("🎯 **Demo Mode** — Model not loaded. Deploy with Option B to enable live predictions.")
        st.markdown("""
        ### How to enable live predictions on Streamlit Cloud

        **Option B** (recommended) — commit the model files:
        ```bash
        # Remove models/ from .gitignore, then:
        git add models/xgb_demand_model.pkl models/model_meta.json
        git add data/cache/eda_stats.npz
        git commit -m "feat: add trained model for Streamlit Cloud deployment"
        git push
        ```
        The dashboard will then show live predictions. Explore the other tabs
        to see **Demand Patterns**, **Year-over-Year**, and **Zone Rankings** charts.
        """)

        # Show sample forecast chart with dummy data so the tab isn't empty
        st.markdown("#### 📊 Sample Forecast (illustrative)")
        hours = list(range(24))
        sample = [18,12,8,6,5,6,10,22,38,42,40,38,35,33,35,38,44,48,42,35,28,24,20,18]
        fig_demo = go.Figure()
        fig_demo.add_trace(go.Scatter(
            x=hours, y=sample, mode="lines+markers", name="Sample Demand",
            line=dict(color=BLUE, width=2.5, shape="spline", smoothing=0.7),
            marker=dict(size=5, color=BLUE, line=dict(color="white", width=1.5)),
            hovertemplate="Hour %{x}:00 — %{y} trips (sample)<extra></extra>",
        ))
        fig_demo.add_trace(go.Scatter(
            x=hours, y=[sum(sample)/24]*24, mode="lines",
            name="Avg", line=dict(color=TEAL, width=1.5, dash="dot"),
        ))
        fig_demo.update_layout(chart_layout(
            title="Sample Hourly Demand Profile (illustrative — not real predictions)",
            xtitle="Hour of Day", ytitle="Trips", height=350,
        ))
        st.plotly_chart(fig_demo, width="stretch")
        st.caption("⚠️ This is illustrative data. Deploy with Option B to see real XGBoost predictions.")

    else:
        start_dt = datetime(target_date.year, target_date.month,
                            target_date.day, target_hour)
        try:
            with st.spinner("Generating forecast..."):
                preds = predictor.predict_next_hours(
                    zone_id=zone_id, start_dt=start_dt,
                    hours=hours_ahead, avg_fare=avg_fare, avg_distance=avg_dist,
                )
            df = pd.DataFrame(preds)
            df["dt"]    = pd.to_datetime(df["target_datetime"])
            df["label"] = df["dt"].dt.strftime("%b %d %H:00")
            demand      = df["predicted_demand"]

            # KPI row
            total   = int(demand.sum())
            peak_v  = int(demand.max())
            avg_v   = round(float(demand.mean()), 1)
            peak_l  = df.loc[demand.idxmax(), "label"]
            min_v   = int(demand.min())

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Predicted Trips", f"{total:,}")
            c2.metric("Peak Hour Demand",       str(peak_v), "trips")
            c3.metric("Avg Trips / Hour",       str(avg_v))
            c4.metric("Min Hour Demand",        str(min_v))
            c5.metric("Peak Time",              peak_l)

            st.markdown("<br>", unsafe_allow_html=True)

            # Main forecast chart
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df["dt"], y=demand,
                fill="tozeroy", fillcolor="rgba(26,86,219,.06)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=df["dt"], y=demand, mode="lines+markers",
                name="Predicted Demand",
                line=dict(color=BLUE, width=2.5, shape="spline", smoothing=0.7),
                marker=dict(size=5, color=BLUE,
                            line=dict(color="white", width=1.5)),
                hovertemplate="<b>%{x|%b %d %H:00}</b><br>%{y} trips<extra></extra>",
            ))
            # Rolling avg
            roll = demand.rolling(3, center=True, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                x=df["dt"], y=roll, mode="lines",
                name="3-hr Rolling Avg",
                line=dict(color=TEAL, width=1.5, dash="dot"),
                hoverinfo="skip",
            ))
            # Peak marker
            pk = df.loc[demand.idxmax()]
            fig.add_trace(go.Scatter(
                x=[pk["dt"]], y=[pk["predicted_demand"]],
                mode="markers+text", name="Peak",
                marker=dict(size=13, color=RED, symbol="star",
                            line=dict(color="white", width=2)),
                text=["Peak"], textposition="top center",
                textfont=dict(color=RED, size=11),
                hoverinfo="skip",
            ))
            fig.update_layout(chart_layout(
                title=f"Demand Forecast — Zone {zone_id}  ·  Next {hours_ahead} Hours",
                xtitle="Time", ytitle="Predicted Pickups", height=380,
                showlegend=True,
            ))
            st.plotly_chart(fig, width="stretch")

            # Hourly bars + table
            col_a, col_b = st.columns([3, 1])
            with col_a:
                colors_bar = [RED if v == demand.max() else
                              TEAL if v == demand.min() else BLUE
                              for v in demand]
                fig2 = go.Figure(go.Bar(
                    x=df["label"], y=demand,
                    marker_color=colors_bar, marker_opacity=0.8,
                    hovertemplate="<b>%{x}</b><br>%{y} trips<extra></extra>",
                ))
                fig2.update_layout(chart_layout(
                    title="Hourly Breakdown  (🔴 peak · 🟢 trough)",
                    xtitle="Hour", ytitle="Trips", height=280, showlegend=False,
                    xaxis=dict(tickangle=-45, gridcolor="#f1f5f9",
                               linecolor="#e2e8f0", tickfont=dict(size=9),
                               zeroline=False),
                ))
                st.plotly_chart(fig2, width="stretch")

            with col_b:
                st.markdown("**📋 Hourly Table**")
                st.dataframe(
                    df[["label", "predicted_demand"]].rename(
                        columns={"label": "Hour", "predicted_demand": "Trips"}
                    ),
                    width="stretch", hide_index=True, height=280,
                )

        except ValueError as e:
            st.error(f"Prediction error: {e}")


# ────────────────────────────────────────────────────────────────
# TAB 2 · DEMAND PATTERNS
# ────────────────────────────────────────────────────────────────
with tab2:
    if not eda:
        st.warning("Build EDA cache first: `python scripts/build_cache.py`")
    else:
        ha = eda["hour_avg"]
        wa = eda["wday_avg"]
        mk = eda["month_keys"].tolist()
        mv = eda["month_vals"].tolist()

        col1, col2 = st.columns(2)

        with col1:
            colors_h = [RED if (7<=h<=9 or 16<=h<=19) else BLUE for h in range(24)]
            fig = go.Figure(go.Bar(
                x=list(range(24)), y=ha,
                marker_color=colors_h, marker_opacity=0.82,
                hovertemplate="Hour %{x}:00 — %{y:.1f} avg trips<extra></extra>",
            ))
            fig.add_vrect(x0=6.5,  x1=9.5,  fillcolor=RED,   opacity=0.05,
                          annotation_text="AM Rush", annotation_font_color=RED,
                          annotation_font_size=10)
            fig.add_vrect(x0=15.5, x1=19.5, fillcolor=AMBER, opacity=0.05,
                          annotation_text="PM Rush", annotation_font_color=AMBER,
                          annotation_font_size=10)
            fig.update_layout(chart_layout(
                title="Avg Demand by Hour of Day",
                xtitle="Hour", ytitle="Avg Trips / Zone-Hour",
                height=320, showlegend=False,
            ))
            st.plotly_chart(fig, width="stretch")

        with col2:
            fig = go.Figure()
            for i, (lbl, clr) in enumerate([("Weekday", BLUE), ("Weekend", TEAL)]):
                fig.add_trace(go.Scatter(
                    x=list(range(24)), y=wa[i], mode="lines+markers",
                    name=lbl, line=dict(color=clr, width=2.5,
                    shape="spline", smoothing=0.6),
                    marker=dict(size=5),
                    hovertemplate=f"<b>{lbl}</b> Hour %{{x}}:00 — %{{y:.1f}}<extra></extra>",
                ))
            fig.update_layout(chart_layout(
                title="Weekday vs Weekend Demand",
                xtitle="Hour", ytitle="Avg Trips / Zone-Hour", height=320,
            ))
            st.plotly_chart(fig, width="stretch")

        # Monthly volume
        m_colors = [BLUE if m.startswith("2026") else
                    TEAL if m.startswith("2025") else
                    INDIGO for m in mk]
        fig = go.Figure(go.Bar(
            x=mk, y=mv, marker_color=m_colors, marker_opacity=0.82,
            hovertemplate="<b>%{x}</b><br>%{y:,.0f} trips<extra></extra>",
        ))
        for yr_m in ["2025-01", "2026-01"]:
            if yr_m in mk:
                idx = mk.index(yr_m)
                fig.add_vline(x=idx-0.5, line_color="#cbd5e1", line_dash="dash")
                fig.add_annotation(x=idx, y=max(mv)*0.95,
                                   text=yr_m[:4], showarrow=False,
                                   font=dict(color="#94a3b8", size=11))
        fig.update_layout(chart_layout(
            title="Monthly Trip Volume (2024–2026)",
            xtitle="Month", ytitle="Total Trips",
            height=300, showlegend=False,
        ))
        st.plotly_chart(fig, width="stretch")

        # Stat cards
        pi = int(np.argmax(mv))
        si = int(np.argmin(mv))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peak Month",    mk[pi], f"{mv[pi]/1e6:.2f}M trips")
        c2.metric("Slowest Month", mk[si], f"{mv[si]/1e6:.2f}M trips")
        c3.metric("Peak Hour",     f"{int(np.argmax(ha))}:00", "highest avg demand")
        c4.metric("Off-Peak Hour", f"{int(np.argmin(ha))}:00", "lowest avg demand")


# ────────────────────────────────────────────────────────────────
# TAB 3 · YEAR-OVER-YEAR
# ────────────────────────────────────────────────────────────────
with tab3:
    if not eda:
        st.warning("Build EDA cache first: `python scripts/build_cache.py`")
    else:
        yoy   = eda["yoy_avg"]
        years = eda["years"].tolist()

        fig = go.Figure()
        for i, (yr, row) in enumerate(zip(years, yoy)):
            fig.add_trace(go.Scatter(
                x=list(range(24)), y=row, mode="lines+markers",
                name=str(int(yr)),
                line=dict(color=PAL[i % len(PAL)], width=2.8,
                           shape="spline", smoothing=0.6),
                marker=dict(size=6, line=dict(color="white", width=1.5)),
                hovertemplate=f"<b>{int(yr)}</b> Hour %{{x}}:00 — %{{y:.1f}}<extra></extra>",
            ))
        fig.update_layout(chart_layout(
            title="Year-over-Year Hourly Demand Comparison",
            xtitle="Hour of Day", ytitle="Avg Trips / Zone-Hour",
            height=420,
            xaxis=dict(tickmode="linear", tick0=0, dtick=1,
                       gridcolor="#f1f5f9", linecolor="#e2e8f0",
                       tickfont=dict(size=10), zeroline=False),
        ))
        st.plotly_chart(fig, width="stretch")

        if len(years) >= 2:
            st.markdown("### 📊 Growth Summary")
            rows = []
            for i in range(1, len(years)):
                prev, curr = yoy[i-1].sum(), yoy[i].sum()
                pct = ((curr-prev)/prev*100) if prev > 0 else 0
                rows.append({
                    "Period":        f"{int(years[i-1])} → {int(years[i])}",
                    "Demand Change": f"{'↑' if pct>=0 else '↓'} {abs(pct):.1f}%",
                    "Peak Hour":     f"{int(np.argmax(yoy[i]))}:00",
                    "Off-Peak Hour": f"{int(np.argmin(yoy[i]))}:00",
                    "Avg Demand":    f"{curr/24:.1f} trips/zone-hr",
                })
            st.dataframe(pd.DataFrame(rows), width="stretch", hide_index=True)


# ────────────────────────────────────────────────────────────────
# TAB 4 · ZONE RANKINGS
# ────────────────────────────────────────────────────────────────
with tab4:
    if not eda:
        st.warning("Build EDA cache first: `python scripts/build_cache.py`")
    else:
        zl = eda["top15_zones"].tolist()
        vl = eda["top15_vals"].tolist()

        df_z = pd.DataFrame({
            "Zone ID":     [str(int(z)) for z in zl],
            "Total Trips": [int(v)       for v in vl],
        }).sort_values("Total Trips")

        col1, col2 = st.columns([3, 2])

        with col1:
            n = len(df_z)
            bar_clrs = [AMBER if i == n-1 else
                        TEAL  if i >= n-3 else BLUE
                        for i in range(n)]
            fig = go.Figure(go.Bar(
                x=df_z["Total Trips"], y=df_z["Zone ID"],
                orientation="h", marker_color=bar_clrs, marker_opacity=0.85,
                hovertemplate="Zone <b>%{y}</b><br>%{x:,.0f} trips<extra></extra>",
            ))
            fig.update_layout(chart_layout(
                title="Top 15 Busiest Pickup Zones (2024–2026)",
                xtitle="Total Pickups", ytitle="Zone ID", height=460,
                xaxis=dict(tickformat=".2s", gridcolor="#f1f5f9",
                           linecolor="#e2e8f0", tickfont=dict(size=11),
                           zeroline=False),
            ))
            st.plotly_chart(fig, width="stretch")

        with col2:
            st.markdown("### 🏆 Leaderboard")
            df_r = df_z.sort_values("Total Trips", ascending=False).reset_index(drop=True)
            df_r.index += 1
            medals = {1: "🥇", 2: "🥈", 3: "🥉"}
            df_r.index  = [medals.get(i, f"#{i}") for i in df_r.index]
            df_r["Total Trips"] = df_r["Total Trips"].apply(lambda x: f"{x/1e6:.2f}M")
            st.dataframe(df_r, width="stretch", height=460)


# ── FOOTER ─────────────────────────────────────────────────────────────────────
st.divider()
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    st.caption("Built by **Ishan Gupta** · NYC Taxi Demand Forecasting · XGBoost + FastAPI + MLflow + Streamlit")
with c2:
    st.markdown("[🐙 GitHub — IshanGupta09](https://github.com/IshanGupta09)")
with c3:
    st.markdown("[💼 LinkedIn — ishan-gupta091](https://www.linkedin.com/in/ishan-gupta091/)")