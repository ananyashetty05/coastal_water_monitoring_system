import streamlit as st
import pandas as pd
import pydeck as pdk

from core.state import get_df
from core.processor import (
    get_location_summaries, filter_df, get_countries, get_waterbody_types, WQI_ORDER
)

st.set_page_config(page_title="Map View · CoastalWatch", layout="wide")


def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css()

st.title("🗺️ Location Map")
st.caption("Monitoring stations colour-coded by most-recent CCME Water Quality Index rating.")

df = get_df()
if df is None:
    st.warning("⚠️ No data loaded. Please go to **Upload Data** first.")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.header("Filters")
    country_opts = ["All"] + get_countries(df)
    sel_country  = st.selectbox("Country", country_opts)

    wbt_opts = ["All"] + get_waterbody_types(df)
    sel_wbt  = st.selectbox("Waterbody Type", wbt_opts)

filtered = filter_df(
    df,
    country=None if sel_country == "All" else sel_country,
    waterbody_type=None if sel_wbt == "All" else sel_wbt,
)

summaries = get_location_summaries(filtered).reset_index(drop=True)

if summaries.empty:
    st.warning("No stations match the current filters.")
    st.stop()

# ── WQI colour mapping ────────────────────────────────────────────────────────
WQI_COLORS = {
    "Excellent": [26,  152, 80,  220],
    "Good":      [145, 207, 96,  220],
    "Marginal":  [254, 224, 139, 220],
    "Fair":      [252, 141, 89,  220],
    "Poor":      [215, 48,  39,  220],
}
WQI_HEX = {
    "Excellent": "#1a9850",
    "Good":      "#91cf60",
    "Marginal":  "#fee08b",
    "Fair":      "#fc8d59",
    "Poor":      "#d73027",
}

# Add color as separate r/g/b/a columns — pydeck reads these reliably
def _wqi_rgba(wqi_label):
    return WQI_COLORS.get(wqi_label, [100, 130, 200, 200])

summaries["r"] = summaries["ccme_wqi"].apply(lambda w: _wqi_rgba(w)[0])
summaries["g"] = summaries["ccme_wqi"].apply(lambda w: _wqi_rgba(w)[1])
summaries["b"] = summaries["ccme_wqi"].apply(lambda w: _wqi_rgba(w)[2])
summaries["a"] = summaries["ccme_wqi"].apply(lambda w: _wqi_rgba(w)[3])
summaries["radius"] = 12000  # metres — visible at zoom 5

# Fill None values for tooltip rendering
for col in ["do", "ph", "ammonia", "bod", "temp", "nitrogen", "nitrate", "ccme_values"]:
    if col in summaries.columns:
        summaries[col] = summaries[col].fillna("N/A")

# ── Layout (map + legend/station side panel) ─────────────────────────────────
map_col, info_col = st.columns([3, 1], gap="large")

with map_col:
    layer = pdk.Layer(
        "ScatterplotLayer",
        data=summaries,
        get_position=["lon", "lat"],
        get_fill_color=["r", "g", "b", "a"],   # explicit RGBA columns — no lambda needed
        get_radius="radius",
        radius_min_pixels=6,
        radius_max_pixels=25,
        pickable=True,
        auto_highlight=True,
        highlight_color=[255, 255, 255, 80],
    )

    centre_lat = float(summaries["lat"].mean())
    centre_lon = float(summaries["lon"].mean())

    # Auto-zoom: wider spread = lower zoom
    lat_spread = float(summaries["lat"].max() - summaries["lat"].min())
    lon_spread = float(summaries["lon"].max() - summaries["lon"].min())
    spread     = max(lat_spread, lon_spread)
    zoom       = 1 if spread > 100 else (3 if spread > 30 else (5 if spread > 5 else 7))

    view = pdk.ViewState(
        latitude=centre_lat,
        longitude=centre_lon,
        zoom=zoom,
        pitch=0,
        bearing=0,
    )

    tooltip = {
        "html": (
            "<b>{location}</b><br>"
            "🏷️ {country} · {waterbody_type}<br>"
            "📊 CCME: <b>{ccme_wqi}</b> (score: {ccme_values})<br>"
            "💧 DO: {do} mg/L &nbsp;|&nbsp; pH: {ph}<br>"
            "🌡️ Temp: {temp} °C &nbsp;|&nbsp; Ammonia: {ammonia} mg/L<br>"
            "🌿 Nitrogen: {nitrogen} mg/L &nbsp;|&nbsp; Nitrate: {nitrate} mg/L<br>"
            "📅 Latest: {date} &nbsp;|&nbsp; 📋 {n} records"
        ),
        "style": {
            "background": "#0a2342",
            "color": "white",
            "fontFamily": "sans-serif",
            "fontSize": "12px",
            "padding": "10px",
            "borderRadius": "8px",
            "maxWidth": "320px",
        },
    }

    # Use "road" (OpenStreetMap-based) — no Mapbox token required
    st.pydeck_chart(
        pdk.Deck(
            layers=[layer],
            initial_view_state=view,
            tooltip=tooltip,
            map_style="road",          # built-in OSM style, always works without a token
        ),
        width="stretch",
        height=520,
    )

with info_col:
    st.subheader("Levels")
    for label, hex_col in WQI_HEX.items():
        count = int((summaries["ccme_wqi"] == label).sum())
        st.markdown(
            f"<div style='display:flex;align-items:center;gap:.6rem;margin-bottom:.4rem;'>"
            f"<div style='width:14px;height:14px;border-radius:50%;flex-shrink:0;"
            f"background:{hex_col};'></div>"
            f"<span style='font-size:.88rem;'><b>{label}</b> ({count})</span></div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader(f"Stations ({len(summaries)})")
    station_options = summaries["location"].astype(str).sort_values().unique().tolist()
    selected_station = st.selectbox("Select station", station_options, key="map_station_select")
    selected_row = summaries[summaries["location"] == selected_station].iloc[0]
    st.markdown(
        f"<div style='background:#f8f9fa;border:1px solid #e5e7eb;border-radius:8px;"
        f"padding:.55rem .65rem;margin-top:.35rem;color:#000;font-size:.82rem;'>"
        f"<b style='color:#000;'>{selected_row['location']}</b><br>"
        f"Country: <b style='color:#000;'>{selected_row.get('country', 'N/A')}</b><br>"
        f"WQI: <b style='color:#000;'>{selected_row.get('ccme_wqi', 'N/A')}</b> · "
        f"Score: <b style='color:#000;'>{selected_row.get('ccme_values', 'N/A')}</b>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ── Country-wise station quality summary ─────────────────────────────────────
st.markdown("---")
st.subheader("🌍 Country-wise Station Quality")

country_wqi_counts = (
    summaries.pivot_table(
        index="country",
        columns="ccme_wqi",
        values="location",
        aggfunc="count",
        fill_value=0,
    )
    .reindex(columns=WQI_ORDER, fill_value=0)
    .astype(int)
)

country_wqi_pct = country_wqi_counts.div(country_wqi_counts.sum(axis=1), axis=0).mul(100).round(1)
country_avg_ccme = summaries.groupby("country", as_index=True)["ccme_values"].mean().round(2)

country_summary = pd.concat(
    [country_wqi_counts.add_suffix(" (#)"), country_wqi_pct.add_suffix(" (%)")],
    axis=1,
)
country_summary["Avg CCME Score"] = country_avg_ccme
country_summary["Stations"] = country_wqi_counts.sum(axis=1).astype(int)
country_summary = country_summary.sort_values("Stations", ascending=False)

st.dataframe(country_summary, width="stretch")

# ── Summary table ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📋 Latest Readings by Station")

keep = ["location", "country", "waterbody_type", "date",
        "ccme_wqi", "ccme_values", "do", "ph", "ammonia", "bod", "temp", "nitrogen", "n"]
disp = summaries[[c for c in keep if c in summaries.columns]].copy()
disp.columns = {
    "location":       "Location",
    "country":        "Country",
    "waterbody_type": "Type",
    "date":           "Latest Date",
    "ccme_wqi":       "CCME WQI",
    "ccme_values":    "CCME Score",
    "do":             "DO (mg/L)",
    "ph":             "pH",
    "ammonia":        "Ammonia",
    "bod":            "BOD",
    "temp":           "Temp (°C)",
    "nitrogen":       "Nitrogen",
    "n":              "Records",
}.values()

st.dataframe(
    disp.sort_values("CCME Score", ascending=False),
    width="stretch",
    hide_index=True,
)
