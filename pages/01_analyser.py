"""
pages/01_analyser.py
- Section 0: Full column inventory (what's in the dataset)
- Sidebar: Station + Depth/Layer + Date range filters
- Section 1: Line chart
- Section 2: Bar chart
- Section 3: Scatter plot
- Section 4: Correlation heatmap
- Section 5: Marine Life Analysis + auto-generated supporting charts
- Section 6: Depth layer comparison
- Section 7: Parameter summary table
"""
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from core.detector  import marine_analysis
from core.processor import get_stats, param_status

st.set_page_config(page_title="Analyser · CoastalWatch", layout="wide")
st.title("📊 Water Quality Analyser")

# ── guard ──────────────────────────────────────────────────────────────────────
if "df" not in st.session_state or "schema" not in st.session_state:
    st.warning("⚠️ Please upload a dataset on the Home page first.")
    st.stop()

df      = st.session_state["df"]
schema  = st.session_state["schema"]
metrics = schema.get("metrics", {})

avail  = [k for k in metrics if k in df.columns and df[k].notna().sum() > 2]
labels = {
    k: f"{metrics[k].get('icon','')} {metrics[k].get('label',k)} ({metrics[k].get('unit','')})"
    for k in avail
}

if not avail:
    st.error("No numeric parameters found. Check your dataset.")
    st.stop()

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 0 — COLUMN INVENTORY
# ═══════════════════════════════════════════════════════════════════════════════
st.subheader("🗂️ Dataset Column Inventory")
st.caption("Every column in your CSV and how it was interpreted.")

structural_rows = []
for col_type, col_name in [
    ("📅 Date",        schema.get("date_col")),
    ("📍 Location",    schema.get("location_col")),
    ("🌍 Country",     schema.get("country_col")),
    ("🌊 Depth/Layer", schema.get("waterbody_col")),
]:
    if col_name:
        structural_rows.append({
            "CSV Column": col_name, "Type": col_type,
            "Label": col_type, "Unit": "", "Group": "structural",
        })

metric_rows = []
for k, m in metrics.items():
    metric_rows.append({
        "CSV Column":    m.get("original_col", k),
        "Type":          "📊 Metric",
        "Label":         m.get("label", k),
        "Unit":          m.get("unit", ""),
        "Group":         m.get("group", "other"),
        "Icon":          m.get("icon", ""),
        "Higher=Better": m.get("higher_is_better"),
        "Safe Range":    str(m.get("safe_range", [])),
        "Poor Range":    str(m.get("poor_range", [])),
    })

all_schema_cols = set(filter(None, [
    schema.get("date_col"), schema.get("location_col"),
    schema.get("country_col"), schema.get("waterbody_col"),
] + [m.get("original_col","") for m in metrics.values()]))

unmapped = [
    c for c in df.columns
    if c not in all_schema_cols
    and c not in ["date","location","country","waterbody_type","lat","lon"]
]
skip_rows = [{"CSV Column": c, "Type": "⬜ Unmapped/Skipped",
              "Label": c, "Unit": "", "Group": ""} for c in unmapped]

tab_all, tab_metrics, tab_struct = st.tabs(["All Columns", "Metrics Only", "Structural"])
with tab_all:
    st.dataframe(pd.DataFrame(structural_rows + metric_rows + skip_rows),
                 hide_index=True, use_container_width=True)
with tab_metrics:
    if metric_rows:
        st.dataframe(pd.DataFrame(metric_rows), hide_index=True, use_container_width=True)
with tab_struct:
    if structural_rows:
        st.dataframe(pd.DataFrame(structural_rows), hide_index=True, use_container_width=True)
    if unmapped:
        st.caption(f"Unmapped columns (not used): {unmapped}")

# ═══════════════════════════════════════════════════════════════════════════════
# SIDEBAR FILTERS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.header("🔽 Filters")

    all_locations = sorted(df["location"].unique().tolist())
    sel_location  = st.selectbox("Station", ["All"] + all_locations)

    # Depth / layer (Surface Water, Middle Water, etc.)
    depth_src = schema.get("waterbody_col","")
    if "waterbody_type" in df.columns and df["waterbody_type"].nunique() > 1:
        all_depths = sorted(df["waterbody_type"].dropna().unique().tolist())
        sel_depth  = st.selectbox("Depth / Layer", ["All"] + all_depths)
        if depth_src:
            st.caption(f"Source column: `{depth_src}`")
    else:
        sel_depth = "All"

    if "date" in df.columns:
        min_d = df["date"].min().date()
        max_d = df["date"].max().date()
        date_range = st.date_input("Date range", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)
    else:
        date_range = None

# apply
fdf = df.copy()
if sel_location != "All":
    fdf = fdf[fdf["location"] == sel_location]
if sel_depth != "All":
    fdf = fdf[fdf["waterbody_type"] == sel_depth]
if date_range and len(date_range) == 2:
    fdf = fdf[(fdf["date"].dt.date >= date_range[0]) &
              (fdf["date"].dt.date <= date_range[1])]

st.markdown("---")
st.caption(
    f"**Filtered:** {len(fdf):,} records · "
    f"{fdf['location'].nunique()} station(s) · "
    f"Layer: **{sel_depth}**"
)

# Depth records mini-bar always visible
if "waterbody_type" in fdf.columns and fdf["waterbody_type"].nunique() > 1:
    dc = fdf["waterbody_type"].value_counts().reset_index()
    dc.columns = ["Layer","Records"]
    st.altair_chart(
        alt.Chart(dc)
        .mark_bar(cornerRadiusTopLeft=3, cornerRadiusTopRight=3, color="#38bdf8")
        .encode(x=alt.X("Layer:N", sort="-y"), y="Records:Q",
                tooltip=["Layer","Records"])
        .properties(height=120, title="Records by Depth Layer"),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LINE CHART
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📈 Trend Over Time")

lc1, lc2, lc3 = st.columns(3)
with lc1:
    y_line   = st.selectbox("Metric (Y)", avail, format_func=lambda k: labels[k], key="line_y")
with lc2:
    c_opts   = [c for c in ["waterbody_type","location","country"] if c in fdf.columns]
    color_by = st.selectbox("Color by", c_opts, key="line_color")
with lc3:
    agg_map  = {"Mean":"mean","Median":"median","Max":"max","Min":"min"}
    agg_lbl  = st.selectbox("Aggregation", list(agg_map.keys()), key="line_agg")

if "date" in fdf.columns and y_line in fdf.columns:
    ldf = (fdf[["date",color_by,y_line]].dropna()
           .groupby(["date",color_by], as_index=False)[y_line]
           .agg(agg_map[agg_lbl]))
    lab  = metrics[y_line].get("label",y_line)
    unit = metrics[y_line].get("unit","")
    st.altair_chart(
        alt.Chart(ldf)
        .mark_line(point=True, strokeWidth=2)
        .encode(
            x=alt.X("date:T", title="Date"),
            y=alt.Y(f"{y_line}:Q", title=f"{agg_lbl} {lab} ({unit})".strip()),
            color=alt.Color(f"{color_by}:N", title=color_by.replace("_"," ").title()),
            tooltip=["date:T", color_by, alt.Tooltip(f"{y_line}:Q", format=".3f", title=lab)],
        )
        .properties(height=320).interactive(),
        use_container_width=True,
    )
else:
    st.info("Need a date column and at least one metric.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — BAR CHART
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("📊 Bar Chart — Average by Group")

bc1, bc2 = st.columns(2)
with bc1:
    x_bar = st.selectbox("Group by (X)",
                          [c for c in ["waterbody_type","location","country"] if c in fdf.columns],
                          key="bar_x")
with bc2:
    y_bar = st.selectbox("Metric (Y)", avail, format_func=lambda k: labels[k], key="bar_y")

agg_bar = fdf.groupby(x_bar)[y_bar].mean().reset_index().dropna()
if not agg_bar.empty:
    lab  = metrics[y_bar].get("label",y_bar)
    unit = metrics[y_bar].get("unit","")
    st.altair_chart(
        alt.Chart(agg_bar)
        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
        .encode(
            x=alt.X(f"{x_bar}:N", sort="-y", title=x_bar.replace("_"," ").title()),
            y=alt.Y(f"{y_bar}:Q", title=f"Avg {lab} ({unit})".strip()),
            color=alt.Color(f"{x_bar}:N", legend=None),
            tooltip=[x_bar, alt.Tooltip(f"{y_bar}:Q", format=".3f", title=f"Avg {lab}")],
        )
        .properties(height=300),
        use_container_width=True,
    )

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — SCATTER PLOT
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🔵 Scatter Plot")

sc1, sc2, sc3 = st.columns(3)
with sc1:
    x_sc = st.selectbox("X axis", avail, index=0, format_func=lambda k: labels[k], key="sc_x")
with sc2:
    y_sc = st.selectbox("Y axis", avail, index=min(1,len(avail)-1), format_func=lambda k: labels[k], key="sc_y")
with sc3:
    c_sc = st.selectbox("Color by", [c for c in ["waterbody_type","location","country"] if c in fdf.columns], key="sc_c")

sc_df = fdf[[x_sc,y_sc,c_sc]].dropna()
if not sc_df.empty:
    pts = (
        alt.Chart(sc_df)
        .mark_circle(size=55, opacity=0.65)
        .encode(
            x=alt.X(f"{x_sc}:Q", title=metrics[x_sc].get("label",x_sc)),
            y=alt.Y(f"{y_sc}:Q", title=metrics[y_sc].get("label",y_sc)),
            color=alt.Color(f"{c_sc}:N"),
            tooltip=[c_sc, alt.Tooltip(f"{x_sc}:Q",format=".3f"),
                            alt.Tooltip(f"{y_sc}:Q",format=".3f")],
        ).properties(height=320).interactive()
    )
    reg = pts.transform_regression(x_sc, y_sc).mark_line(
        strokeDash=[5,3], color="white", opacity=0.4
    )
    st.altair_chart(pts + reg, use_container_width=True)
else:
    st.info("Not enough data for scatter plot with current filters.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — CORRELATION HEATMAP
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🌡️ Correlation Heatmap")

corr_avail = [k for k in avail if k in fdf.columns and fdf[k].notna().sum() > 5]
if len(corr_avail) >= 2:
    sel_corr = (
        st.multiselect("Select parameters", corr_avail, default=corr_avail[:10],
                       format_func=lambda k: metrics[k].get("label",k))
        if len(corr_avail) > 10 else corr_avail
    )
    if len(sel_corr) >= 2:
        corr      = fdf[sel_corr].corr().round(2)
        short     = {k: metrics[k].get("label",k)[:18] for k in sel_corr}
        corr.index   = [short[k] for k in corr.index]
        corr.columns = [short[k] for k in corr.columns]
        cm = corr.reset_index().melt(id_vars="index")
        cm.columns = ["x","y","r"]
        heat = (
            alt.Chart(cm).mark_rect()
            .encode(
                x=alt.X("x:N", axis=alt.Axis(labelAngle=-40, labelLimit=120)),
                y=alt.Y("y:N", axis=alt.Axis(labelLimit=120)),
                color=alt.Color("r:Q", scale=alt.Scale(scheme="redblue",domain=[-1,1]), title="r"),
                tooltip=["x","y",alt.Tooltip("r:Q",format=".2f")],
            ).properties(height=400)
        )
        txt = (
            alt.Chart(cm).mark_text(fontSize=9)
            .encode(x="x:N", y="y:N", text=alt.Text("r:Q",format=".2f"),
                    color=alt.condition((alt.datum.r>0.5)|(alt.datum.r<-0.5),
                                        alt.value("white"), alt.value("black")))
        )
        st.altair_chart(heat + txt, use_container_width=True)
else:
    st.info("Need at least 2 numeric parameters for the heatmap.")

# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MARINE LIFE ANALYSIS + SUPPORTING CHARTS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.subheader("🐠 Marine Life Impact Analysis")

loc_for_analysis = sel_location if sel_location != "All" else all_locations[0]
stats = get_stats(df, loc_for_analysis, schema)

if st.button("🤖 Generate Marine Life Analysis", type="primary"):
    with st.spinner("Analysing impact on marine ecosystems…"):
        analysis = marine_analysis(schema, stats)
    st.session_state["marine_analysis"] = analysis

if "marine_analysis" not in st.session_state:
    st.info("Click the button above to generate AI-powered marine life analysis.")
else:
    commentary = st.session_state["marine_analysis"]

    # Commentary box
    st.markdown(
        f"""<div style="background:#0f3460;border-left:4px solid #0ea5e9;
            border-radius:0 12px 12px 0;padding:18px 22px;
            font-size:.93rem;color:#e2e8f0;line-height:1.9;margin-bottom:1.2rem">
            🐠 {commentary}
        </div>""",
        unsafe_allow_html=True,
    )

    # ── Detect which metrics the commentary mentions ───────────────────────────
    commentary_lower = commentary.lower()
    mentioned: list[str] = []

    for k, m in metrics.items():
        if k not in fdf.columns or fdf[k].notna().sum() < 3:
            continue
        # match on label words OR original column name
        words = [w for w in m.get("label","").lower().split() if len(w) > 3]
        orig  = m.get("original_col","").lower()
        if any(w in commentary_lower for w in words) or orig in commentary_lower:
            if k not in mentioned:
                mentioned.append(k)

    # also add metrics with notable trends not already included
    for k, v in stats.items():
        if k == "_meta" or not isinstance(v, dict):
            continue
        if abs(v.get("trend",0)) > 1e-5 and k not in mentioned and k in fdf.columns:
            mentioned.append(k)

    mentioned = mentioned[:8]   # cap at 8

    if not mentioned:
        # fallback: just use top 4 by data availability
        mentioned = [k for k in avail if k in fdf.columns][:4]

    st.markdown("#### 📊 Supporting Evidence — Charts for the Analysis Above")
    st.caption(
        "Each chart below corresponds to a parameter mentioned in the commentary. "
        "Trends over time + depth layer breakdown are shown side by side."
    )

    for key in mentioned:
        m    = metrics.get(key, {})
        lab  = m.get("label", key)
        unit = m.get("unit","")
        icon = m.get("icon","📊")
        hib  = m.get("higher_is_better")
        stat = stats.get(key, {})

        st.markdown(f"---\n**{icon} {lab}**")

        # KPI row
        if stat:
            kpi1, kpi2, kpi3, kpi4 = st.columns(4)
            latest = stat.get("latest", float("nan"))
            trend  = stat.get("trend", 0)
            status, scolor = param_status(key, latest, schema)
            t_dir  = "↑ Rising" if trend > 1e-5 else ("↓ Falling" if trend < -1e-5 else "→ Stable")

            if hib is True and trend < -1e-5:
                concern = "⚠️ Concerning decline"
            elif hib is False and trend > 1e-5:
                concern = "⚠️ Concerning rise"
            else:
                concern = "Within expectations"

            kpi1.metric("Latest",  f"{latest:.3f} {unit}".strip())
            kpi2.metric("Mean",    f"{stat.get('mean',0):.3f} {unit}".strip())
            kpi3.metric("Trend",   t_dir)
            kpi4.metric("Status",  f"{status} — {concern}")

        # Two charts side by side: line trend + depth bar
        ch1, ch2 = st.columns(2)

        with ch1:
            if "date" in fdf.columns and key in fdf.columns:
                color_dim = "waterbody_type" if (
                    "waterbody_type" in fdf.columns and fdf["waterbody_type"].nunique() > 1
                ) else "location"
                td = (
                    fdf[["date", color_dim, key]].dropna()
                    .groupby(["date", color_dim], as_index=False)[key].mean()
                )
                if not td.empty:
                    st.altair_chart(
                        alt.Chart(td)
                        .mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=25))
                        .encode(
                            x=alt.X("date:T", title="Date"),
                            y=alt.Y(f"{key}:Q", title=f"{lab} ({unit})".strip()),
                            color=alt.Color(f"{color_dim}:N",
                                            legend=alt.Legend(orient="bottom",
                                                              labelFontSize=9,
                                                              titleFontSize=9)),
                            tooltip=["date:T", color_dim,
                                     alt.Tooltip(f"{key}:Q",format=".3f")],
                        )
                        .properties(height=220, title=f"Trend over time"),
                        use_container_width=True,
                    )

        with ch2:
            # depth bar OR distribution histogram
            if "waterbody_type" in fdf.columns and fdf["waterbody_type"].nunique() > 1 and key in fdf.columns:
                da = fdf.groupby("waterbody_type")[key].mean().reset_index().dropna()
                da.columns = ["Layer","value"]
                if not da.empty:
                    st.altair_chart(
                        alt.Chart(da)
                        .mark_bar(cornerRadiusTopLeft=4, cornerRadiusTopRight=4)
                        .encode(
                            x=alt.X("Layer:N", title="Depth Layer",
                                    axis=alt.Axis(labelAngle=-20)),
                            y=alt.Y("value:Q", title=f"Avg {lab} ({unit})".strip()),
                            color=alt.Color("Layer:N", legend=None),
                            tooltip=["Layer", alt.Tooltip("value:Q",format=".3f")],
                        )
                        .properties(height=220, title="Avg by Depth Layer"),
                        use_container_width=True,
                    )
            else:
                # histogram of distribution
                if key in fdf.columns and fdf[key].notna().sum() > 5:
                    hist_df = fdf[[key]].dropna()
                    st.altair_chart(
                        alt.Chart(hist_df)
                        .mark_bar(color="#38bdf8", opacity=0.8)
                        .encode(
                            x=alt.X(f"{key}:Q", bin=alt.Bin(maxbins=20),
                                    title=f"{lab} ({unit})".strip()),
                            y=alt.Y("count()", title="Count"),
                            tooltip=[alt.Tooltip(f"{key}:Q",format=".3f"), "count()"],
                        )
                        .properties(height=220, title="Distribution"),
                        use_container_width=True,
                    )
