"""
app.py  —  FINAL VERSION
Home page: upload CSV → Groq analyses → KPIs displayed.
- Station selector now includes "All Stations" option
- Shows aggregated stats across all stations when All is selected
"""
import streamlit as st
import pandas as pd

from core.detector  import detect, _key, GROQ_OK
from core.processor import parse, get_stats, param_status

st.set_page_config(
    page_title="CoastalWatch", page_icon="🌊", layout="wide"
)

# ── hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:linear-gradient(135deg,#0a2342,#1a5276,#0e6655);
     border-radius:14px;padding:2.5rem;margin-bottom:1.5rem;color:white">
  <h1 style="margin:0;font-size:2.4rem">🌊 CoastalWatch</h1>
  <p style="opacity:.85;margin-top:.5rem;max-width:680px">
    Upload any marine / ocean / coastal water quality CSV.
    AI detects all parameters automatically and generates insights.
  </p>
</div>
""", unsafe_allow_html=True)

# ── API key status ─────────────────────────────────────────────────────────────
api_key = _key()
if not GROQ_OK:
    st.error("❌ `groq` not installed. Run: `pip install groq`")
elif not api_key:
    st.warning(
        "⚠️ **GROQ_API_KEY not set.**  \n"
        "The app will still work but uses heuristic column detection (less accurate).  \n"
        "To enable AI detection: create `.streamlit/secrets.toml` with:  \n"
        "```\nGROQ_API_KEY = \"gsk_your_key_here\"\n```  \n"
        "Get a free key at https://console.groq.com"
    )
else:
    st.success(f"✅ Groq API ready ({api_key[:8]}…)")

# ── upload ─────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Upload your water quality CSV",
    type=["csv"],
    help="Any format — column names are detected automatically."
)

if uploaded is None:
    st.info("👆 Upload a CSV to get started.")
    with st.expander("What formats are supported?"):
        st.markdown("""
        - Any water quality / ocean / coastal monitoring CSV
        - Column names are detected automatically — no renaming needed
        - Supports: DO, pH, BOD, ammonia, turbidity, salinity, nutrients, bacteria, etc.
        - Supports depth layers (Surface Water, Middle Water, Bottom Water)
        - Date formats: DD-MM-YYYY, YYYY-MM-DD, MM/DD/YYYY, etc.
        """)
    st.stop()

# ── detect + parse ─────────────────────────────────────────────────────────────
with st.spinner("🤖 AI is analysing your dataset…"):
    try:
        raw_df = pd.read_csv(uploaded)
        raw_df.columns = raw_df.columns.str.strip()

        if raw_df.empty:
            st.error("❌ CSV file is empty.")
            st.stop()

        schema = detect(raw_df)
        uploaded.seek(0)
        df = parse(uploaded, schema)

    except ValueError as e:
        st.error(f"❌ {e}")
        st.stop()
    except Exception as e:
        st.error(f"❌ Unexpected error: {e}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

# store for other pages
st.session_state["df"]     = df
st.session_state["schema"] = schema

# ── status badge ───────────────────────────────────────────────────────────────
conf   = schema.get("confidence", 0)
is_llm = schema.get("_llm", False)
color  = "#22c55e" if conf > 0.8 else ("#f59e0b" if conf > 0.5 else "#ef4444")
method = "🤖 Groq LLM" if is_llm else "🔧 Heuristic"

c1, c2, c3 = st.columns(3)
c1.markdown(
    f"<span style='background:{color}22;color:{color};padding:4px 14px;"
    f"border-radius:99px;font-size:.85rem;font-weight:700'>"
    f"{method} · Confidence {conf*100:.0f}%</span>",
    unsafe_allow_html=True,
)
if not df.empty:
    c2.success(f"✅ {len(df):,} rows loaded")
c3.info(f"📍 {schema.get('detected_location','Unknown')}")

# ── dataset summary ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🧠 AI Dataset Analysis")

col1, col2 = st.columns([2, 1])
with col1:
    summary = schema.get("dataset_summary", "")
    if summary:
        st.info(summary)

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Records",  f"{len(df):,}")
    k2.metric("Stations",       df["location"].nunique())
    k3.metric("Parameters",     len(schema.get("metrics", {})))
    k4.metric("Date Range",
              f"{df['date'].min().date()} → {df['date'].max().date()}"
              if "date" in df.columns else "N/A")

with col2:
    concerns = schema.get("key_concerns", [])
    if concerns:
        st.markdown("**⚠️ Key Concerns**")
        for c in concerns:
            st.markdown(f"- {c}")

# ── parameter groups ────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🔬 Detected Parameters")

metrics = schema.get("metrics", {})
groups: dict = {}
for k, v in metrics.items():
    groups.setdefault(v.get("group", "other"), []).append((k, v))

if not groups:
    st.warning("No parameters detected. Check your CSV format.")
else:
    for group, items in sorted(groups.items()):
        with st.expander(f"**{group.title()}** — {len(items)} parameter(s)", expanded=True):
            gcols = st.columns(min(len(items), 5))
            for col, (key, meta) in zip(gcols, items):
                in_df = key in df.columns and df[key].notna().sum() > 0
                col.markdown(
                    f"<div style='background:{'#0f172a' if in_df else '#1a0a0a'};"
                    f"border:1px solid {'#1e293b' if in_df else '#3a1a1a'};"
                    f"border-radius:8px;padding:10px;text-align:center'>"
                    f"<div style='font-size:1.3rem'>{meta.get('icon','📊')}</div>"
                    f"<div style='font-size:.72rem;color:#94a3b8;margin-top:4px'>"
                    f"{meta.get('label',key)}</div>"
                    f"<div style='font-size:.65rem;color:#475569'>{meta.get('unit','')}</div>"
                    f"<div style='font-size:.6rem;color:{'#22c55e' if in_df else '#ef4444'}'>"
                    f"{'✅ data found' if in_df else '⚠️ no data'}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

# ── depth layers ───────────────────────────────────────────────────────────────
if "waterbody_type" in df.columns and df["waterbody_type"].nunique() > 1:
    st.markdown("---")
    st.subheader("🌊 Depth Layers Detected")
    dc = df["waterbody_type"].value_counts().reset_index()
    dc.columns = ["Layer", "Records"]
    st.dataframe(dc, hide_index=True, use_container_width=True)

# ── per-station KPIs ───────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📍 Station-level Latest Readings")

all_locs    = sorted(df["location"].unique().tolist())
ALL_LABEL   = "🌐 All Stations"
loc_options = [ALL_LABEL] + all_locs

location = st.selectbox("Select station", loc_options)

if location == ALL_LABEL:
    # ── Aggregate view across all stations ────────────────────────────────────
    st.caption(f"Showing **aggregated latest readings** across all {len(all_locs)} stations")

    metric_keys = [k for k in metrics if k in df.columns and df[k].notna().sum() > 0]
    if metric_keys:
        # For each metric: take the latest non-null value per station, then average
        agg_rows = []
        for key in metric_keys:
            latest_vals = (
                df[["location", "date", key]]
                .dropna(subset=[key])
                .sort_values("date")
                .groupby("location")[key]
                .last()
            )
            meta   = metrics.get(key, {})
            mean_v = latest_vals.mean()
            status, sc = param_status(key, mean_v, schema)
            agg_rows.append({
                "key": key, "meta": meta,
                "mean": mean_v, "status": status, "sc": sc,
                "n_stations": latest_vals.notna().sum(),
            })

        card_cols = st.columns(min(len(agg_rows), 5))
        for col, row in zip(card_cols, agg_rows[:10]):
            meta   = row["meta"]
            sc     = row["sc"]
            status = row["status"]
            col.markdown(
                f"<div style='background:{sc}18;border:1px solid {sc}55;"
                f"border-radius:10px;padding:12px;text-align:center'>"
                f"<div style='font-size:.68rem;color:#64748b;text-transform:uppercase'>"
                f"{meta.get('icon','')} {meta.get('label', row['key'])}</div>"
                f"<div style='font-size:1.5rem;font-weight:800;color:{sc}'>"
                f"{row['mean']:.3f}</div>"
                f"<div style='font-size:.68rem;color:{sc}'>{status}</div>"
                f"<div style='font-size:.62rem;color:#64748b'>"
                f"{meta.get('unit','')} · {row['n_stations']} stations</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
    else:
        st.info("No metric data available to aggregate.")

else:
    # ── Single-station view ───────────────────────────────────────────────────
    stats = get_stats(df, location, schema)

    if stats and "_meta" in stats:
        m = stats["_meta"]
        st.caption(
            f"Records: {m['n']} · {m['date_from']} → {m['date_to']} · "
            f"{m['country']} · {m['waterbody']}"
        )

    avail_stats = [(k, v) for k, v in stats.items() if k != "_meta" and isinstance(v, dict)]
    if avail_stats:
        card_cols = st.columns(min(len(avail_stats), 5))
        for col, (key, val) in zip(card_cols, avail_stats[:10]):
            meta   = metrics.get(key, {})
            status, sc = param_status(key, val["latest"], schema)
            t_icon = "↑" if val["trend"] > 1e-5 else ("↓" if val["trend"] < -1e-5 else "→")
            col.markdown(
                f"<div style='background:{sc}18;border:1px solid {sc}55;"
                f"border-radius:10px;padding:12px;text-align:center'>"
                f"<div style='font-size:.68rem;color:#64748b;text-transform:uppercase'>"
                f"{meta.get('icon','')} {meta.get('label',key)}</div>"
                f"<div style='font-size:1.5rem;font-weight:800;color:{sc}'>"
                f"{val['latest']:.3f}</div>"
                f"<div style='font-size:.68rem;color:{sc}'>{status} {t_icon}</div>"
                f"<div style='font-size:.62rem;color:#64748b'>{meta.get('unit','')}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )

# ── column mapping table ───────────────────────────────────────────────────────
with st.expander("🗺️ Full Column Mapping (click to expand)"):
    rows = []
    for k, v in metrics.items():
        rows.append({
            "CSV Column":    v.get("original_col", k),
            "Internal Key":  k,
            "Label":         v.get("label", k),
            "Unit":          v.get("unit", ""),
            "Group":         v.get("group", ""),
            "Higher=Better": v.get("higher_is_better"),
            "Safe Range":    str(v.get("safe_range", [])),
            "In Dataset":    "✅" if k in df.columns and df[k].notna().sum() > 0 else "❌",
        })
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ── nav hint ───────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("👈 Use the sidebar to navigate to **Analyser**, **Predictor**, or **Station Map**")