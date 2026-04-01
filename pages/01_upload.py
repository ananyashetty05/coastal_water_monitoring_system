import streamlit as st
import pandas as pd

from core.processor import parse_csv, get_countries, get_wqi_distribution, WQI_ORDER
from core.classifier import benchmark_models
from core.state import set_df, get_df, set_ml_bundle

st.set_page_config(page_title="Upload Data · CoastalWatch", layout="wide")


def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css()

st.title("📤 Upload Data")
st.caption("Load `data.csv` — the coastal water quality dataset.")

# ── Schema reference ──────────────────────────────────────────────────────────
with st.expander("📋 Expected CSV columns", expanded=False):
    st.markdown(
        """
        | Column | Description |
        |---|---|
        | `Country` | Ireland or England |
        | `Area` | Monitoring station / waterbody name |
        | `Waterbody Type` | Coastal / Transitional / Estuarine / Sea Water |
        | `Date` | DD-MM-YYYY |
        | `Ammonia (mg/l)` | Ammonia concentration |
        | `Biochemical Oxygen Demand (mg/l)` | BOD |
        | `Dissolved Oxygen (mg/l)` | DO |
        | `Orthophosphate (mg/l)` | Orthophosphate |
        | `pH (ph units)` | pH |
        | `Temperature (cel)` | Water temperature |
        | `Nitrogen (mg/l)` | Total nitrogen |
        | `Nitrate (mg/l)` | Nitrate |
        | `CCME_Values` | Numeric CCME score (0–100) |
        | `CCME_WQI` | Excellent / Good / Marginal / Fair / Poor |
        """
    )

# ── Upload ────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload data.csv",
    type=["csv"],
    label_visibility="collapsed",
)

if uploaded:
    with st.spinner("Parsing and cleaning data…"):
        try:
            df = parse_csv(uploaded)
            set_df(df)
            ml_bundle = benchmark_models(df)
            set_ml_bundle(ml_bundle)
            st.success(
                f"✅ Loaded **{len(df):,} rows** · "
                f"**{df['location'].nunique()} locations** · "
                f"**{', '.join(get_countries(df))}** · "
                f"{df['date'].min().date()} → {df['date'].max().date()}"
            )
            if isinstance(ml_bundle, dict) and ml_bundle.get("available"):
                st.caption(
                    f"ML benchmark ready · best classifier: "
                    f"`{ml_bundle.get('best_model_name', 'N/A')}`"
                )
        except ValueError as e:
            st.error(f"❌ {e}")
        except Exception as e:
            st.error(f"❌ Failed to initialize ML benchmark: {e}")

# ── Preview ───────────────────────────────────────────────────────────────────
df = get_df()
if df is not None:
    st.markdown("---")

    # Top-level KPIs
    k1, k2, k3 = st.columns(3)
    k1.metric("Total Records",    f"{len(df):,}")
    k2.metric("Unique Locations", df["location"].nunique())
    k3.metric("Countries",        df["country"].nunique())

    d1, d2 = st.columns(2)
    d1.metric("Date From", str(df["date"].min().date()))
    d2.metric("Date To",   str(df["date"].max().date()))

    # WQI distribution bar
    st.markdown("#### CCME WQI Distribution (all records)")
    wqi_dist = get_wqi_distribution(df)
    wqi_colors = {
        "Excellent": "#1a9850",
        "Good":      "#91cf60",
        "Marginal":  "#fee08b",
        "Fair":      "#fc8d59",
        "Poor":      "#d73027",
    }
    # Build a mini horizontal bar chart with st.progress workaround
    total = wqi_dist.sum()
    cols_wqi = st.columns(len(wqi_dist))
    for col, (label, count) in zip(cols_wqi, wqi_dist.items()):
        pct = count / total * 100 if total else 0
        col.markdown(
            f"<div style='text-align:center;'>"
            f"<div style='background:{wqi_colors.get(label,'#999')};"
            f"border-radius:6px;height:14px;width:{max(pct,2):.0f}%;margin:auto;'></div>"
            f"<p style='font-size:.8rem;margin:.3rem 0 0;'><b>{label}</b><br>{count:,} ({pct:.1f}%)</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.subheader("📄 Data Preview")

    # Filters
    fc1, fc2, fc3 = st.columns(3)
    countries   = ["All"] + sorted(df["country"].unique().tolist())
    wbo_types   = ["All"] + sorted(df["waterbody_type"].unique().tolist())
    wqi_labels  = ["All"] + [l for l in WQI_ORDER if l in df["ccme_wqi"].unique()]

    sel_country = fc1.selectbox("Country",       countries)
    sel_wbt     = fc2.selectbox("Waterbody Type", wbo_types)
    sel_wqi     = fc3.selectbox("CCME WQI",       wqi_labels)

    preview = df.copy()
    if sel_country != "All":
        preview = preview[preview["country"] == sel_country]
    if sel_wbt != "All":
        preview = preview[preview["waterbody_type"] == sel_wbt]
    if sel_wqi != "All":
        preview = preview[preview["ccme_wqi"] == sel_wqi]

    display_cols = [
        "country", "location", "waterbody_type", "date",
        "do", "ph", "ammonia", "bod", "temp",
        "nitrogen", "nitrate", "ccme_values", "ccme_wqi",
    ]
    st.dataframe(
        preview[display_cols].sort_values("date", ascending=False).head(300),
        width="stretch",
        height=380,
    )
    st.caption(f"Showing up to 300 rows of {len(preview):,} filtered records.")

else:
    st.info("👆 Upload `data.csv` to get started.")
