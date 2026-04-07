"""
pages/03_map.py  —  Country-level choropleth map
Aggregates station readings per country → shaded regions.
Robust country-name alias matching so USA, UK, China etc. all resolve.
"""
import json
import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import requests

from core.processor import param_status, get_stats

st.set_page_config(page_title="Station Map · CoastalWatch", page_icon="🗺️", layout="wide")

# ── guard ──────────────────────────────────────────────────────────────────────
if "df" not in st.session_state or "schema" not in st.session_state:
    st.warning("⚠️ Upload a dataset on the Home page first.")
    st.stop()

df      = st.session_state["df"]
schema  = st.session_state["schema"]
metrics = schema.get("metrics", {})
metric_keys = [k for k in metrics if k in df.columns and df[k].notna().sum() > 0]

st.title("🗺️ Water Quality Map")
st.caption("Countries shaded by overall water quality — select a country card for station detail.")

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")
    colour_options  = ["Overall Quality"] + [metrics[k].get("label", k) for k in metric_keys]
    colour_by_label = st.selectbox("Colour by", colour_options)
    colour_by = (
        None if colour_by_label == "Overall Quality"
        else metric_keys[colour_options.index(colour_by_label) - 1]
    )
    if "waterbody_type" in df.columns and df["waterbody_type"].nunique() > 1:
        depth_opts = ["All"] + sorted(df["waterbody_type"].dropna().unique().tolist())
        sel_depth  = st.selectbox("Depth / Layer", depth_opts)
    else:
        sel_depth = "All"
    map_style = st.selectbox(
        "Map style", ["dark", "light", "satellite", "road"], index=0,
        format_func=lambda s: {"dark":"🌑 Dark","satellite":"🛰️ Satellite",
                               "light":"☀️ Light","road":"🗺️ Road"}[s],
    )
    map_style_url = {
        "dark":      "mapbox://styles/mapbox/dark-v10",
        "satellite": "mapbox://styles/mapbox/satellite-streets-v11",
        "light":     "mapbox://styles/mapbox/light-v10",
        "road":      "mapbox://styles/mapbox/streets-v11",
    }[map_style]

# ── constants ──────────────────────────────────────────────────────────────────
STATUS_ORDER = {"Poor": 0, "Moderate": 1, "Safe": 2, "Unknown": 3}
STATUS_COLOR_RGBA = {
    "Poor":     [239, 68,  68,  200],
    "Moderate": [245, 158, 11,  200],
    "Safe":     [34,  197, 94,  200],
    "Unknown":  [148, 163, 184, 80],
}
STATUS_HEX = {
    "Poor": "#ef4444", "Moderate": "#f59e0b",
    "Safe": "#22c55e", "Unknown":  "#94a3b8",
}

# ── country name alias table ───────────────────────────────────────────────────
# Maps common shorthand → GeoJSON "ADMIN" field name (Natural Earth)
COUNTRY_ALIASES: dict[str, str] = {
    # English shorthand → full ADMIN name
    "USA":                        "United States of America",
    "US":                         "United States of America",
    "UNITED STATES":              "United States of America",
    "AMERICA":                    "United States of America",
    "UK":                         "United Kingdom",
    "GREAT BRITAIN":              "United Kingdom",
    "BRITAIN":                    "United Kingdom",
    "ENGLAND":                    "United Kingdom",
    "SCOTLAND":                   "United Kingdom",
    "WALES":                      "United Kingdom",
    "NORTHERN IRELAND":           "United Kingdom",
    "UAE":                        "United Arab Emirates",
    "RUSSIA":                     "Russia",
    "RUSSIAN FEDERATION":         "Russia",
    "SOUTH KOREA":                "South Korea",
    "KOREA":                      "South Korea",
    "REPUBLIC OF KOREA":          "South Korea",
    "NORTH KOREA":                "North Korea",
    "DPRK":                       "North Korea",
    "CHINA":                      "China",
    "PRC":                        "China",
    "PEOPLES REPUBLIC OF CHINA":  "China",
    "HONG KONG":                  "China",
    "MACAU":                      "China",
    "TAIWAN":                     "Taiwan",
    "VIETNAM":                    "Vietnam",
    "VIET NAM":                   "Vietnam",
    "CZECHIA":                    "Czech Republic",
    "CZECH REPUBLIC":             "Czech Republic",
    "NORTH MACEDONIA":            "Macedonia",
    "ESWATINI":                   "Swaziland",
    "CABO VERDE":                 "Cape Verde",
    "TIMOR-LESTE":                "East Timor",
    "MYANMAR":                    "Myanmar",
    "BURMA":                      "Myanmar",
    "IRAN":                       "Iran",
    "ISLAMIC REPUBLIC OF IRAN":   "Iran",
    "SYRIA":                      "Syria",
    "SYRIAN ARAB REPUBLIC":       "Syria",
    "BOLIVIA":                    "Bolivia",
    "TANZANIA":                   "Tanzania",
    "REPUBLIC OF IRELAND":        "Ireland",
    "IRELAND":                    "Ireland",
    "CANADA":                     "Canada",
    "AUSTRALIA":                  "Australia",
    "INDIA":                      "India",
    "JAPAN":                      "Japan",
    "GERMANY":                    "Germany",
    "FRANCE":                     "France",
    "BRAZIL":                     "Brazil",
    "ITALY":                      "Italy",
    "SPAIN":                      "Spain",
    "MEXICO":                     "Mexico",
    "INDONESIA":                  "Indonesia",
    "NETHERLANDS":                "Netherlands",
    "HOLLAND":                    "Netherlands",
    "TURKEY":                     "Turkey",
    "TURKIYE":                    "Turkey",
    "SAUDI ARABIA":               "Saudi Arabia",
    "SOUTH AFRICA":               "South Africa",
    "ARGENTINA":                  "Argentina",
    "EGYPT":                      "Egypt",
    "NIGERIA":                    "Nigeria",
    "PAKISTAN":                   "Pakistan",
    "BANGLADESH":                 "Bangladesh",
    "PHILIPPINES":                "Philippines",
    "MALAYSIA":                   "Malaysia",
    "SINGAPORE":                  "Singapore",
    "THAILAND":                   "Thailand",
    "NEW ZEALAND":                "New Zealand",
    "GREECE":                     "Greece",
    "PORTUGAL":                   "Portugal",
    "SWEDEN":                     "Sweden",
    "NORWAY":                     "Norway",
    "DENMARK":                    "Denmark",
    "FINLAND":                    "Finland",
    "POLAND":                     "Poland",
    "BELGIUM":                    "Belgium",
    "SWITZERLAND":                "Switzerland",
    "AUSTRIA":                    "Austria",
    "ISRAEL":                     "Israel",
    "KENYA":                      "Kenya",
    "GHANA":                      "Ghana",
    "ETHIOPIA":                   "Ethiopia",
    "MOROCCO":                    "Morocco",
    "ALGERIA":                    "Algeria",
    "TUNISIA":                    "Tunisia",
    "LIBYA":                      "Libya",
    "JORDAN":                     "Jordan",
    "IRAQ":                       "Iraq",
    "KUWAIT":                     "Kuwait",
    "QATAR":                      "Qatar",
    "BAHRAIN":                    "Bahrain",
    "OMAN":                       "Oman",
    "YEMEN":                      "Yemen",
    "SRI LANKA":                  "Sri Lanka",
    "MYANMAR":                    "Myanmar",
    "CAMBODIA":                   "Cambodia",
    "LAOS":                       "Laos",
    "NEPAL":                      "Nepal",
    "BHUTAN":                     "Bhutan",
    "MONGOLIA":                   "Mongolia",
    "KAZAKHSTAN":                 "Kazakhstan",
    "UZBEKISTAN":                 "Uzbekistan",
    "UKRAINE":                    "Ukraine",
    "ROMANIA":                    "Romania",
    "HUNGARY":                    "Hungary",
    "CROATIA":                    "Croatia",
    "SERBIA":                     "Serbia",
    "BULGARIA":                   "Bulgaria",
    "SLOVAKIA":                   "Slovakia",
    "COLOMBIA":                   "Colombia",
    "PERU":                       "Peru",
    "CHILE":                      "Chile",
    "VENEZUELA":                  "Venezuela",
    "CUBA":                       "Cuba",
    "JAMAICA":                    "Jamaica",
    "PANAMA":                     "Panama",
    "COSTA RICA":                 "Costa Rica",
    "ECUADOR":                    "Ecuador",
    "PARAGUAY":                   "Paraguay",
    "URUGUAY":                    "Uruguay",
}


def _resolve_country(name: str) -> str:
    """Normalize a country name from data → GeoJSON ADMIN name."""
    upper = name.strip().upper()
    return COUNTRY_ALIASES.get(upper, name.strip())  # fallback: use as-is


# ── filter by depth ────────────────────────────────────────────────────────────
work = df.copy()
if sel_depth != "All" and "waterbody_type" in work.columns:
    work = work[work["waterbody_type"] == sel_depth]
if work.empty:
    st.error("No data for the selected depth/layer.")
    st.stop()

# ── aggregate per country ──────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def build_country_summary(data_hash: str, _work: pd.DataFrame,
                          colour_by: str | None) -> pd.DataFrame:
    rows = []
    for country_raw, grp in _work.groupby("country"):
        country = _resolve_country(str(country_raw))
        worst   = "Unknown"
        c_val   = None
        param_lines = []

        for key in metric_keys:
            if key not in grp.columns:
                continue
            vals = grp[key].dropna()
            if vals.empty:
                continue
            mean_val = float(vals.mean())
            status, _ = param_status(key, mean_val, schema)
            label = metrics[key].get("label", key)
            unit  = metrics[key].get("unit", "")
            shex  = STATUS_HEX.get(status, "#94a3b8")
            param_lines.append(
                f"{metrics[key].get('icon','•')} {label}: {mean_val:.3f} {unit} [{status}]"
            )
            if STATUS_ORDER.get(status, 3) < STATUS_ORDER.get(worst, 3):
                worst = status
            if key == colour_by:
                c_val = mean_val

        rows.append({
            "country_raw": str(country_raw),
            "country":     country,           # resolved GeoJSON name
            "status":      worst,
            "color":       STATUS_COLOR_RGBA.get(worst, STATUS_COLOR_RGBA["Unknown"]),
            "n_stations":  grp["location"].nunique(),
            "n_records":   len(grp),
            "date_from":   grp["date"].min().strftime("%Y-%m-%d") if "date" in grp.columns else "?",
            "date_to":     grp["date"].max().strftime("%Y-%m-%d") if "date" in grp.columns else "?",
            "colour_val":  c_val,
        })

    cdf = pd.DataFrame(rows)

    if colour_by and not cdf.empty and cdf["colour_val"].notna().any():
        vals = cdf["colour_val"].dropna()
        lo, hi = vals.min(), vals.max()
        hib = metrics.get(colour_by, {}).get("higher_is_better", True)
        def _gc(v):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                return STATUS_COLOR_RGBA["Unknown"]
            norm = (v - lo) / (hi - lo) if hi != lo else 0.5
            if not hib: norm = 1 - norm
            return ([34,197,94,200] if norm > 0.6 else
                    [245,158,11,200] if norm > 0.3 else [239,68,68,200])
        cdf["color"] = cdf["colour_val"].apply(_gc)

    return cdf.reset_index(drop=True)


_hash = str(len(work)) + str(sorted(work.columns.tolist())) + str(sel_depth) + str(colour_by)
with st.spinner("Building country summary…"):
    country_df = build_country_summary(_hash, work, colour_by)

if country_df.empty:
    st.error("No country data found. Make sure your dataset has a 'country' column.")
    st.stop()

# ── KPI strip ──────────────────────────────────────────────────────────────────
n_safe = (country_df["status"] == "Safe").sum()
n_mod  = (country_df["status"] == "Moderate").sum()
n_poor = (country_df["status"] == "Poor").sum()

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Countries", len(country_df))
k1.metric("Stations",  int(country_df["n_stations"].sum()))
for col, label, val, color in [
    (k2, "🟢 Safe",     n_safe, "#22c55e"),
    (k3, "🟡 Moderate", n_mod,  "#f59e0b"),
    (k4, "🔴 Poor",     n_poor, "#ef4444"),
]:
    col.markdown(
        f"<div style='text-align:center;padding:10px'>"
        f"<div style='font-size:.75rem;color:#64748b'>{label}</div>"
        f"<div style='font-size:1.8rem;font-weight:800;color:{color}'>{val}</div>"
        f"</div>", unsafe_allow_html=True,
    )

st.markdown("---")

# ── country cards ──────────────────────────────────────────────────────────────
st.subheader("🌍 Countries")

n_cols    = min(len(country_df), 5)
card_cols = st.columns(n_cols)
selected_country = st.session_state.get("selected_country", country_df.iloc[0]["country_raw"])

for i, (_, row) in enumerate(country_df.iterrows()):
    col   = card_cols[i % n_cols]
    color = STATUS_HEX.get(row["status"], "#94a3b8")
    is_sel = row["country_raw"] == selected_country
    border = f"3px solid {color}" if is_sel else f"1px solid {color}44"
    bg     = f"{color}22"         if is_sel else f"{color}0d"
    with col:
        st.markdown(
            f"<div style='background:{bg};border:{border};border-radius:12px;"
            f"padding:12px;text-align:center;margin-bottom:8px'>"
            f"<div style='font-size:.78rem;font-weight:700;color:#e2e8f0'>{row['country_raw']}</div>"
            f"<div style='font-size:1.1rem;font-weight:800;color:{color};margin:4px 0'>{row['status']}</div>"
            f"<div style='font-size:.68rem;color:#64748b'>"
            f"{row['n_stations']} station{'s' if row['n_stations']!=1 else ''} · {row['n_records']:,} records</div>"
            f"</div>", unsafe_allow_html=True,
        )
        if st.button("Select", key=f"sel_{row['country_raw']}", use_container_width=True):
            st.session_state["selected_country"] = row["country_raw"]
            st.rerun()

st.markdown("---")

# ── GeoJSON choropleth ─────────────────────────────────────────────────────────
st.subheader("🗺️ Map")

@st.cache_data(show_spinner=False)
def load_world_geojson() -> dict:
    url = "https://raw.githubusercontent.com/datasets/geo-countries/master/data/countries.geojson"
    try:
        r = requests.get(url, timeout=10)
        return r.json()
    except Exception:
        return {"type": "FeatureCollection", "features": []}

with st.spinner("Loading world map…"):
    world_geo = load_world_geojson()

# build lookup by resolved country name (UPPER)
status_lookup = {row["country"].upper(): row["status"] for _, row in country_df.iterrows()}
color_lookup  = {row["country"].upper(): row["color"]  for _, row in country_df.iterrows()}

# annotate GeoJSON — try ADMIN, then NAME, then ISO_A3
features = []
for feat in world_geo.get("features", []):
    props = feat.get("properties", {})
    # try multiple property keys that Natural Earth uses
    candidates = [
        props.get("ADMIN", ""),
        props.get("name", ""),
        props.get("NAME", ""),
        props.get("NAME_LONG", ""),
        props.get("FORMAL_EN", ""),
    ]
    matched_status = "Unknown"
    matched_color  = STATUS_COLOR_RGBA["Unknown"]
    for cand in candidates:
        key = cand.upper()
        if key in status_lookup:
            matched_status = status_lookup[key]
            matched_color  = color_lookup[key]
            break

    feat = dict(feat)
    feat["properties"] = {
        **props,
        "color":  matched_color,
        "status": matched_status,
        "label":  props.get("ADMIN") or props.get("name") or "",
    }
    features.append(feat)

annotated_geo = {"type": "FeatureCollection", "features": features}

geojson_layer = pdk.Layer(
    "GeoJsonLayer",
    data=annotated_geo,
    pickable=True,
    stroked=True,
    filled=True,
    get_fill_color="properties.color",
    get_line_color=[255, 255, 255, 40],
    line_width_min_pixels=1,
    auto_highlight=True,
    highlight_color=[255, 255, 255, 50],
)

lat_center = float(work["lat"].mean()) if "lat" in work.columns and work["lat"].notna().any() else 20.0
lon_center = float(work["lon"].mean()) if "lon" in work.columns and work["lon"].notna().any() else 0.0

st.pydeck_chart(
    pdk.Deck(
        layers=[geojson_layer],
        initial_view_state=pdk.ViewState(latitude=lat_center, longitude=lon_center, zoom=2, pitch=0),
        map_style=map_style_url,
        tooltip={
            "html": "<b>{properties.label}</b><br>Status: <b>{properties.status}</b>",
            "style": {"background":"#0f172a","color":"#e2e8f0",
                      "borderRadius":"8px","padding":"10px","fontSize":"13px",
                      "border":"1px solid #1e293b"},
        },
    ),
    use_container_width=True,
)

if colour_by:
    lab = metrics[colour_by].get("label", colour_by)
    hib = metrics[colour_by].get("higher_is_better", True)
    st.caption(f"🎨 **{lab}** · {'🟢 High=good → 🔴 Low=poor' if hib else '🟢 Low=good → 🔴 High=poor'}")
else:
    st.caption("🎨 🟢 Safe  🟡 Moderate  🔴 Poor  ⚪ No data")

st.markdown("---")

# ── country summary table ──────────────────────────────────────────────────────
st.subheader("📋 Country Summary")
tbl = country_df[["country_raw","status","n_stations","n_records","date_from","date_to"]].rename(columns={
    "country_raw":"Country","status":"Quality","n_stations":"Stations",
    "n_records":"Records","date_from":"From","date_to":"To",
})
st.dataframe(tbl, hide_index=True, use_container_width=True)

st.markdown("---")

# ── station drill-down ─────────────────────────────────────────────────────────
st.subheader(f"🔍 Station Detail — {selected_country}")

country_stations = sorted(
    work[work["country"] == selected_country]["location"].unique().tolist()
) if "country" in work.columns else []

if not country_stations:
    # fallback: match on resolved name
    resolved = _resolve_country(selected_country)
    country_stations = sorted(
        work[work["country"].apply(_resolve_country) == resolved]["location"].unique().tolist()
    )

if not country_stations:
    st.info(f"No stations found for {selected_country}.")
else:
    sel_station = st.selectbox("Select station", country_stations)
    with st.spinner(f"Loading {sel_station}…"):
        stats = get_stats(work, sel_station, schema)

    meta = stats.get("_meta", {})
    sc1, sc2, sc3, sc4 = st.columns(4)
    sc1.metric("Records",    meta.get("n", "—"))
    sc2.metric("From",       meta.get("date_from", "—"))
    sc3.metric("To",         meta.get("date_to", "—"))
    sc4.metric("Water Type", meta.get("waterbody", "—"))

    avail_s = [(k, v) for k, v in stats.items() if k != "_meta" and isinstance(v, dict)]
    if avail_s:
        dcols = st.columns(min(len(avail_s), 5))
        for col, (key, val) in zip(dcols, avail_s):
            meta_m  = metrics.get(key, {})
            status, sc_col = param_status(key, val["latest"], schema)
            t_icon  = "↑" if val["trend"] > 1e-5 else ("↓" if val["trend"] < -1e-5 else "→")
            col.markdown(
                f"<div style='background:{sc_col}18;border:1px solid {sc_col}55;"
                f"border-radius:10px;padding:12px;text-align:center'>"
                f"<div style='font-size:.68rem;color:#64748b;text-transform:uppercase'>"
                f"{meta_m.get('icon','')} {meta_m.get('label',key)}</div>"
                f"<div style='font-size:1.4rem;font-weight:800;color:{sc_col}'>{val['latest']:.3f}</div>"
                f"<div style='font-size:.68rem;color:{sc_col}'>{status} {t_icon}</div>"
                f"<div style='font-size:.62rem;color:#64748b'>"
                f"min {val['min']:.3f} · max {val['max']:.3f}</div>"
                f"<div style='font-size:.62rem;color:#475569'>{meta_m.get('unit','')}</div>"
                f"</div>", unsafe_allow_html=True,
            )