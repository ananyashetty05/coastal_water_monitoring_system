import streamlit as st
import pandas as pd
import altair as alt

from core.state import get_df
from core.processor import (
    get_stats, get_timeseries, get_wqi_distribution,
    filter_df, get_countries, get_waterbody_types,
    METRIC_LABELS, WQI_ORDER,
)
from components.metric_row import render as render_metrics
from components.quality_badge import render as render_badge
from components.summary_table import render as render_table

st.set_page_config(page_title="Analytics · CoastalWatch", layout="wide")


def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


def _daily_metric_frame(df: pd.DataFrame, location: str, metric: str, label: str) -> pd.DataFrame:
    frame = (
        df[df["location"] == location][["date", metric]]
        .dropna()
        .groupby("date", as_index=False)
        .mean(numeric_only=True)
        .sort_values("date")
        .rename(columns={metric: label})
    )
    return frame


def _inference_note(text: str):
    st.markdown(
        f"""
        <div style="background:#f8fafc;border:1px solid #cbd5e1;border-radius:10px;
            padding:.7rem .9rem;margin-top:.55rem;color:#0f172a;font-size:.88rem;">
            <b>Inference:</b> {text}
        </div>
        """,
        unsafe_allow_html=True,
    )


load_css()

st.title("📊 Water Quality Analytics")
st.caption("Parameter trends, CCME scores, and quality breakdown by monitoring station.")

df = get_df()
if df is None:
    st.warning("⚠️ No data loaded. Please go to **Upload Data** first.")
    st.stop()

# ── Global filters (sidebar) ──────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    countries = get_countries(df)
    sel_country = st.selectbox("Country", ["All"] + countries)

    wbt_types = get_waterbody_types(df)
    sel_wbt   = st.selectbox("Waterbody Type", ["All"] + wbt_types)

filtered_df = filter_df(
    df,
    country=None if sel_country == "All" else sel_country,
    waterbody_type=None if sel_wbt == "All" else sel_wbt,
)

# ── Location selector (dropdown only) ────────────────────────────────────────
locations = sorted(filtered_df["location"].unique())
if not locations:
    st.warning("No locations match the current filters.")
    st.stop()

if "selected_location" not in st.session_state or st.session_state["selected_location"] not in locations:
    st.session_state["selected_location"] = locations[0]

st.markdown("**Select monitoring station**")
default_station = st.session_state.get("selected_location", locations[0])
default_idx = locations.index(default_station) if default_station in locations else 0
location = st.selectbox("Station", locations, index=default_idx, key="analytics_station_select")
st.session_state["selected_location"] = location

stats = get_stats(filtered_df, location)
if not stats:
    st.error("No data available for the selected location.")
    st.stop()

meta = stats.get("_meta", {})

# ── Station info banner ───────────────────────────────────────────────────────
st.markdown(
    f"""<div style="background:#f0f7ff;border:1px solid #d0e8ff;border-radius:10px;
        padding:.8rem 1.2rem;margin-bottom:.8rem;font-size:.9rem;color:#1a3a5c;">
        📍 <b>{meta.get('location','')}</b> &nbsp;·&nbsp;
        🌍 {meta.get('country','')} &nbsp;·&nbsp;
        💧 {meta.get('waterbody_type','')} &nbsp;·&nbsp;
        📋 {meta.get('n', 0):,} records
        ({meta.get('date_from','')} → {meta.get('date_to','')})
    </div>""",
    unsafe_allow_html=True,
)

# ── CCME quality badge ────────────────────────────────────────────────────────
render_badge({
    "status":   meta.get("ccme_wqi", "Unknown"),
    "score":    meta.get("ccme_values"),
    "messages": [],
})

st.markdown("#### Latest Parameter Readings")
render_metrics(stats)

st.markdown("---")

# ── Trend + Distribution charts ───────────────────────────────────────────────
left, right = st.columns(2, gap="large")

PLOTTABLE = {k: v for k, v in METRIC_LABELS.items() if k in stats}

with left:
    st.subheader("📈 Trend Over Time")
    trend_metric = st.selectbox(
        "Parameter", list(PLOTTABLE.keys()),
        format_func=lambda k: PLOTTABLE[k],
        key="trend_metric",
    )
    trend_label = PLOTTABLE[trend_metric]
    trend_df = _daily_metric_frame(filtered_df, location, trend_metric, trend_label)
    trend_chart = (
        alt.Chart(trend_df)
        .mark_line(point=True, color="#38bdf8", strokeWidth=3)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{trend_label}:Q", title=trend_label),
            tooltip=[
                alt.Tooltip("date:T", title="Date"),
                alt.Tooltip(f"{trend_label}:Q", title=trend_label, format=".3f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(trend_chart, width="stretch")

    slope = stats[trend_metric]["trend"]
    direction = "↑ Rising" if slope > 1e-6 else ("↓ Falling" if slope < -1e-6 else "→ Stable")
    st.caption(f"90-day trend: **{direction}** ({slope:+.5f} units/day)")
    if not trend_df.empty:
        start_val = float(trend_df[trend_label].iloc[0])
        end_val = float(trend_df[trend_label].iloc[-1])
        delta = end_val - start_val
        if abs(delta) < 1e-6:
            trend_text = f"{trend_label} has remained broadly stable across the observed period."
        elif delta > 0:
            trend_text = f"{trend_label} ends about {delta:.2f} units higher than it starts, indicating an overall upward movement."
        else:
            trend_text = f"{trend_label} ends about {abs(delta):.2f} units lower than it starts, indicating an overall decline."
        _inference_note(trend_text)

with right:
    st.subheader("📊 Monthly Averages")
    bar_metric = st.selectbox(
        "Parameter", list(PLOTTABLE.keys()),
        format_func=lambda k: PLOTTABLE[k],
        key="bar_metric",
    )
    bar_label = PLOTTABLE[bar_metric]
    monthly = (
        _daily_metric_frame(filtered_df, location, bar_metric, bar_label)
        .set_index("date")[bar_label]
        .resample("ME")
        .mean()
        .dropna()
        .reset_index()
    )
    monthly_chart = (
        alt.Chart(monthly)
        .mark_bar(color="#60a5fa", cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X("yearmonth(date):T", title="Month"),
            y=alt.Y(f"{bar_label}:Q", title=bar_label),
            tooltip=[
                alt.Tooltip("yearmonth(date):T", title="Month"),
                alt.Tooltip(f"{bar_label}:Q", title=bar_label, format=".3f"),
            ],
        )
        .properties(height=280)
    )
    st.altair_chart(monthly_chart, width="stretch")
    st.caption("Each bar = monthly mean")
    if not monthly.empty:
        peak_row = monthly.loc[monthly[bar_label].idxmax()]
        low_row = monthly.loc[monthly[bar_label].idxmin()]
        _inference_note(
            f"The highest monthly average for {bar_label} occurs in {peak_row['date']:%b %Y} "
            f"({peak_row[bar_label]:.2f}), while the lowest occurs in {low_row['date']:%b %Y} "
            f"({low_row[bar_label]:.2f})."
        )

# ── CCME WQI breakdown for this location ─────────────────────────────────────
st.markdown("---")
wqi_col, params_col = st.columns(2, gap="large")

with wqi_col:
    st.subheader("🏷️ CCME WQI History")
    loc_df = filtered_df[filtered_df["location"] == location].copy()
    wqi_dist = get_wqi_distribution(filtered_df, location)
    wqi_colors_hex = {
        "Excellent": "#1a9850",
        "Good":      "#91cf60",
        "Marginal":  "#fee08b",
        "Fair":      "#fc8d59",
        "Poor":      "#d73027",
    }
    for label, count in wqi_dist.items():
        total = wqi_dist.sum()
        pct   = count / total * 100 if total else 0
        bar_w = max(int(pct * 2), 4)  # scale to ~200px max
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem;'>"
            f"<span style='width:80px;font-size:.85rem;'>{label}</span>"
            f"<div style='background:{wqi_colors_hex.get(label,'#aaa')};"
            f"width:{bar_w}px;height:18px;border-radius:4px;'></div>"
            f"<span style='font-size:.82rem;color:#555;'>{count} ({pct:.1f}%)</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # CCME score over time
    ccme_df = _daily_metric_frame(filtered_df, location, "ccme_values", "CCME Score")
    if not ccme_df.empty:
        ccme_chart = (
            alt.Chart(ccme_df)
            .mark_line(point=True, color="#f59e0b", strokeWidth=3)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("CCME Score:Q", title="CCME Score", scale=alt.Scale(domain=[0, 100])),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("CCME Score:Q", title="CCME Score", format=".2f"),
                ],
            )
            .properties(height=220)
        )
        st.altair_chart(ccme_chart, width="stretch")
    if not wqi_dist.empty:
        dominant_label = wqi_dist.idxmax()
        dominant_pct = (wqi_dist.max() / wqi_dist.sum() * 100) if wqi_dist.sum() else 0
        if not ccme_df.empty:
            latest_ccme = float(ccme_df["CCME Score"].iloc[-1])
            ccme_text = (
                f"{dominant_label} is the most frequent WQI class at this station "
                f"({dominant_pct:.1f}% of records), and the latest CCME score is {latest_ccme:.2f}."
            )
        else:
            ccme_text = f"{dominant_label} is the most frequent WQI class at this station ({dominant_pct:.1f}% of records)."
        _inference_note(ccme_text)

with params_col:
    st.subheader("🔎 All Parameters (normalised)")
    all_frames = []
    for metric, label in METRIC_LABELS.items():
        if metric == "ccme_values" or metric not in stats:
            continue
        metric_df = _daily_metric_frame(filtered_df, location, metric, label)
        if metric_df.empty:
            continue
        values = metric_df[label]
        denom = (values.max() - values.min()) + 1e-9
        metric_df["Normalised Value"] = (values - values.min()) / denom
        metric_df["Parameter"] = label
        all_frames.append(metric_df[["date", "Parameter", "Normalised Value"]])

    if all_frames:
        norm_df = pd.concat(all_frames, ignore_index=True)
        norm_chart = (
            alt.Chart(norm_df)
            .mark_line(strokeWidth=2.5)
            .encode(
                x=alt.X("date:T", title="Date"),
                y=alt.Y("Normalised Value:Q", title="Normalised Value", scale=alt.Scale(domain=[0, 1])),
                color=alt.Color("Parameter:N", title="Parameter"),
                tooltip=[
                    alt.Tooltip("date:T", title="Date"),
                    alt.Tooltip("Parameter:N", title="Parameter"),
                    alt.Tooltip("Normalised Value:Q", title="Normalised", format=".3f"),
                ],
            )
            .properties(height=300)
        )
        st.altair_chart(norm_chart, width="stretch")
        variability = (
            norm_df.groupby("Parameter", as_index=False)["Normalised Value"]
            .agg(lambda s: float(s.max() - s.min()))
            .sort_values("Normalised Value", ascending=False)
        )
        top_param = variability.iloc[0]
        _inference_note(
            f"{top_param['Parameter']} shows the widest relative variation over time, "
            f"with a normalised spread of {top_param['Normalised Value']:.2f}."
        )
    else:
        st.info("No parameter history available for normalised comparison.")
    st.caption("All parameters normalised 0–1 for visual comparison.")

# ── Year-wise summary ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Year-wise Summary")

from core.processor import METRICS, METRIC_LABELS as ML

loc_df = filtered_df[filtered_df["location"] == location].copy()
loc_df["year"] = loc_df["date"].dt.year

year_rows = []
for year, grp in loc_df.groupby("year"):
    row = {"Year": int(year), "Records": len(grp)}
    for m in METRICS:
        col = grp[m].dropna()
        if not col.empty:
            row[ML.get(m, m)] = round(float(col.mean()), 2)
        else:
            row[ML.get(m, m)] = None
    # CCME WQI most common label
    if "ccme_wqi" in grp.columns:
        row["CCME WQI"] = grp["ccme_wqi"].mode().iloc[0] if not grp["ccme_wqi"].dropna().empty else "—"
    year_rows.append(row)

if year_rows:
    year_df = pd.DataFrame(year_rows).set_index("Year")
    st.dataframe(year_df, width="stretch")
else:
    st.info("No year-wise data available.")

st.markdown("---")
st.subheader("📊 Parameter Summary")
render_table(stats)
