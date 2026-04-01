import streamlit as st
import pandas as pd

from core.state import get_df, get_ml_bundle
from core.predictor import predict
from core.processor import (
    filter_df, get_countries, get_waterbody_types, METRIC_LABELS
)
from components.quality_badge import render as render_badge

st.set_page_config(page_title="Predictions · CoastalWatch", layout="wide")


def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css()

st.title("🔮 7-Day Water Quality Forecast")
st.caption(
    "Linear-regression forecasts for CCME score and individual parameters. "
    "Row colours reflect projected quality category."
)

df = get_df()
if df is None:
    st.warning("⚠️ No data loaded. Please go to **Upload Data** first.")
    st.stop()

# ── Filters ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    sel_country = st.selectbox("Country", ["All"] + get_countries(df))
    sel_wbt     = st.selectbox("Waterbody Type", ["All"] + get_waterbody_types(df))

filtered_df = filter_df(
    df,
    country=None if sel_country == "All" else sel_country,
    waterbody_type=None if sel_wbt == "All" else sel_wbt,
)

locations = sorted(filtered_df["location"].unique())
if not locations:
    st.warning("No locations match filters.")
    st.stop()

location = st.selectbox("Select monitoring station", locations)

# Reuse cached classifier benchmark if available.
# If it isn't ready, fall back to rule-based quality labels so the page
# still renders immediately instead of blocking on model training.
ml_bundle = get_ml_bundle()
if ml_bundle is None:
    st.caption("Using rule-based quality outlook while no cached ML benchmark is available.")

# ── Run forecast ──────────────────────────────────────────────────────────────
with st.spinner("Running forecast model…"):
    try:
        result = predict(filtered_df, location, classifier_bundle=ml_bundle)
    except Exception as e:
        st.error(f"Forecast failed: {e}")
        st.stop()

predictions    = result["predictions"]     # dict: metric → list[float] (7 values)
forecast_series = result["forecast_series"] # dict: metric → pd.DataFrame (Historical + Forecast cols)
if result.get("best_model_name"):
    st.caption(f"Classification backend: `{result['best_model_name']}`")

# ── CCME projected badge ──────────────────────────────────────────────────────
day7_ccme = predictions.get("ccme_values", [None] * 7)[-1]

def _ccme_to_label(score) -> str:
    if score is None:
        return "Unknown"
    if score >= 95:  return "Excellent"
    if score >= 80:  return "Good"
    if score >= 65:  return "Marginal"
    if score >= 45:  return "Fair"
    return "Poor"


def _quality_theme(label: str) -> tuple[str, str]:
    if label in {"Excellent", "Good"}:
        return "#d9f99d", "#166534"
    if label in {"Marginal", "Fair"}:
        return "#fde68a", "#b45309"
    return "#fecaca", "#b91c1c"

proj_label = _ccme_to_label(day7_ccme)
st.markdown("#### Projected Quality — Day 7")
render_badge({
    "status":   proj_label,
    "score":    round(day7_ccme, 2) if day7_ccme is not None else None,
    "messages": [f"Forecast CCME score: {day7_ccme:.1f}" if day7_ccme is not None else "No CCME data"],
})

card_bg, card_border = _quality_theme(proj_label)
st.markdown(
    f"""
    <style>
      .cw-pred-grid {{
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 14px;
        margin-top: 14px;
      }}
      .cw-pred-card {{
        background: {card_bg};
        border: 1px solid {card_border};
        border-radius: 12px;
        padding: 16px 14px;
      }}
      .cw-pred-label {{
        color: #000000;
        font-size: 0.82rem;
        margin: 0 0 8px 0;
      }}
      .cw-pred-value {{
        color: #000000;
        font-size: 1.8rem;
        font-weight: 800;
        margin: 0;
        line-height: 1.1;
        word-break: break-word;
      }}
      @media (max-width: 900px) {{
        .cw-pred-grid {{
          grid-template-columns: 1fr;
        }}
      }}
    </style>
    <div class="cw-pred-grid">
      <div class="cw-pred-card">
        <p class="cw-pred-label">Selected Station</p>
        <p class="cw-pred-value">{location}</p>
      </div>
      <div class="cw-pred-card">
        <p class="cw-pred-label">Forecast Horizon</p>
        <p class="cw-pred-value">7 days</p>
      </div>
      <div class="cw-pred-card">
        <p class="cw-pred-label">Projected Day 7 CCME</p>
        <p class="cw-pred-value">{f"{day7_ccme:.2f}" if day7_ccme is not None else "N/A"}</p>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Day-by-day table ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📅 Day-by-Day Parameter Forecast")

# Build table: rows = Day+1…+7, cols = metrics
table_data = {
    METRIC_LABELS.get(m, m): [round(v, 3) for v in vals]
    for m, vals in predictions.items()
}
forecast_df = pd.DataFrame(table_data, index=[f"Day +{i+1}" for i in range(7)])
forecast_df.insert(
    0,
    "Projected Quality",
    [
        _ccme_to_label(score)
        for score in predictions.get("ccme_values", [None] * len(forecast_df))
    ],
)
st.dataframe(forecast_df, width="stretch")

# ── Day-by-day quality summary ────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧭 Daily Quality Outlook")

quality_rows = []
for item in result.get("quality_forecast", []):
    quality_rows.append(
        {
            "Day": f"Day +{item.get('day')}",
            "Date": pd.to_datetime(item.get("date")).date() if item.get("date") is not None else None,
            "Projected Quality": item.get("wqi_label") or item.get("status") or "Unknown",
            "Confidence": (
                round(float(item["confidence"]) * 100, 1)
                if item.get("confidence") is not None
                else None
            ),
            "Method": item.get("method", "N/A"),
            "Recommendations": " | ".join(item.get("recommendations", [])) or "No action suggested",
        }
    )

if quality_rows:
    quality_df = pd.DataFrame(quality_rows)
    st.dataframe(
        quality_df,
        width="stretch",
        column_config={
            "Confidence": st.column_config.NumberColumn("Confidence (%)", format="%.1f"),
        },
        hide_index=True,
    )
else:
    st.info("No quality outlook could be generated for the selected station.")

if result.get("recommendations"):
    st.markdown("#### Recommended Actions")
    for rec in result["recommendations"]:
        st.markdown(f"- {rec}")

# ── Forecast chart ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📈 Historical + Forecast Trend")

plot_metrics = {k: v for k, v in METRIC_LABELS.items() if k in forecast_series}
metric_choice = st.selectbox(
    "Select parameter to plot",
    list(plot_metrics.keys()),
    format_func=lambda k: plot_metrics[k],
)

if metric_choice in forecast_series:
    series_data = forecast_series[metric_choice]  # DataFrame with Historical / Forecast columns
    st.line_chart(series_data, width="stretch", height=340)
    st.caption(
        "**Historical** (solid) = actual observations · "
        "**Forecast** (lighter) = 7-day OLS regression projection."
    )
else:
    st.info("No forecast series available for this parameter.")

# ── Model explanation ─────────────────────────────────────────────────────────
with st.expander("ℹ️ About the forecast model"):
    st.markdown(
        """
        **Method**: Ordinary Least Squares (OLS) linear regression fitted on all
        available observations for the selected station.

        **Parameters forecast**: Dissolved Oxygen, pH, Ammonia, BOD, Orthophosphate,
        Temperature, Nitrogen, Nitrate, CCME Score.

        **Horizon**: 7 daily point estimates per parameter.

        **Limitations**:
        - Linear models cannot capture seasonal cycles or pollution events.
        - Accuracy degrades significantly beyond 3–4 days.
        - Stations with very few readings may produce unreliable forecasts.
        - Predictions are clipped to physically plausible ranges.

        For regulatory decisions, always use on-site measurements.
        """
    )
