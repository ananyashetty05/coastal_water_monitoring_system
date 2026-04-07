"""
pages/02_predictor.py
Monthly ensemble forecast with dual-panel charts.
Horizon: 3 months → 2 years.
Charts: compact overview + zoomed last-12mo + forecast zone.
"""
import hashlib
from datetime import datetime
from dateutil.relativedelta import relativedelta

import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

from core.predictor import forecast
from core.detector  import degradation_forecast_analysis
from core.processor import param_status

st.set_page_config(page_title="Predictor · CoastalWatch", layout="wide")
st.title("🔮 Water Quality Predictor")
st.caption("OLS trend + RF/ARIMA residual corrections — monthly granularity.")

# ── guard ──────────────────────────────────────────────────────────────────────
if "df" not in st.session_state or "schema" not in st.session_state:
    st.warning("⚠️ Upload a dataset on the Home page first.")
    st.stop()

df      = st.session_state["df"]
schema  = st.session_state["schema"]
metrics = schema.get("metrics", {})
avail   = [k for k in metrics if k in df.columns and df[k].notna().sum() >= 1]

if not avail:
    st.error("No numeric parameters found.")
    st.stop()

# ── sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    all_locations = sorted(df["location"].unique().tolist())
    location      = st.selectbox("Station", ["All Stations"] + all_locations)

    if "waterbody_type" in df.columns and df["waterbody_type"].nunique() > 1:
        depth_opts = ["All"] + sorted(df["waterbody_type"].dropna().unique().tolist())
        sel_depth  = st.selectbox("Depth / Layer", depth_opts)
    else:
        sel_depth = "All"

    st.markdown("---")
    st.subheader("📅 Forecast Horizon")
    horizon_months = st.select_slider(
        "Months ahead",
        options=[3, 6, 9, 12, 18, 24],
        value=12,
        format_func=lambda m: (
            f"{m} months" if m < 12 else
            "1 year"      if m == 12 else
            f"{m//12} years {m%12}m" if m % 12 else
            f"{m//12} years"
        ),
    )
    st.markdown("---")

    sel_metrics = st.multiselect(
        "Parameters to forecast",
        avail,
        default=avail[:5],
        format_func=lambda k: f"{metrics[k].get('icon','')} {metrics[k].get('label',k)}",
    )
    st.caption(f"Forecasting {len(sel_metrics)} parameter(s) only.")

    if st.button("🗑️ Clear cache", use_container_width=True):
        for k in list(st.session_state.keys()):
            if k.startswith("fc_"):
                del st.session_state[k]
        st.rerun()

if not sel_metrics:
    st.info("Select at least one parameter.")
    st.stop()

# ── prepare input df ───────────────────────────────────────────────────────────
POOLED = "All Stations"

if sel_depth != "All" and "waterbody_type" in df.columns:
    input_df = df[df["waterbody_type"] == sel_depth].copy()
else:
    input_df = df.copy()

if location == POOLED:
    metric_cols = [k for k in sel_metrics if k in input_df.columns]
    pooled = (
        input_df.groupby("date")[metric_cols]
        .mean(numeric_only=True)
        .reset_index()
    )
    pooled["location"]       = POOLED
    pooled["country"]        = input_df["country"].iloc[0] if "country" in input_df.columns else "Unknown"
    pooled["waterbody_type"] = sel_depth
    input_df          = pooled
    forecast_location = POOLED
    display_name      = f"All Stations ({df['location'].nunique()} pooled)"
else:
    forecast_location = location
    display_name      = location

# ── cache key ──────────────────────────────────────────────────────────────────
ck = "fc_" + hashlib.md5(
    f"{forecast_location}|{sel_depth}|{horizon_months}|{'_'.join(sorted(sel_metrics))}".encode()
).hexdigest()

# ── run forecast ───────────────────────────────────────────────────────────────
if ck not in st.session_state:
    prog_slot   = st.empty()
    status_slot = st.empty()
    bar         = prog_slot.progress(0, text="Starting forecast…")
    done_labels = []

    def on_progress(metric: str, done: int, total: int):
        lab = metrics.get(metric, {}).get("label", metric)
        done_labels.append(lab)
        bar.progress(done / total, text=f"✅ {lab} ({done}/{total})")
        status_slot.caption("  ·  ".join(done_labels[-4:]))

    results = forecast(
        input_df, forecast_location, schema,
        steps=horizon_months,
        metrics_filter=sel_metrics,
        progress_cb=on_progress,
    )

    bar.progress(1.0, text="✅ Done!")
    prog_slot.empty(); status_slot.empty()
    st.session_state[ck] = results
else:
    st.caption("📦 Loaded from cache.")

results     = st.session_state.get(ck, {})
has_results = [m for m in sel_metrics if m in results]

if not results:
    loc_rows = df if location == POOLED else df[df["location"] == location]
    st.error(
        f"No forecast for **{display_name}** (depth: {sel_depth}).  \n"
        f"Rows: **{len(loc_rows)}** · Tried: **{sel_metrics}**"
    )
    st.dataframe(loc_rows.head())
    st.stop()

end_date = datetime.today() + relativedelta(months=horizon_months)
st.caption(
    f"Station: **{display_name}** · Depth: **{sel_depth}** · "
    f"Forecast: **{horizon_months}mo** → **{end_date.strftime('%b %Y')}** · "
    f"**{len(has_results)}** parameter(s)"
)

# ── model weights ──────────────────────────────────────────────────────────────
with st.expander("🤖 Ensemble details per parameter"):
    for m in has_results:
        info   = results[m].get("ensemble_info", {})
        w      = info.get("weights", {})
        maes   = info.get("maes", {})
        slope  = info.get("ols_slope_per_month", 0)
        method = results[m].get("method", "?")
        lab    = metrics[m].get("label", m)
        st.markdown(
            f"**{metrics[m].get('icon','')} {lab}** — `{method}`  \n"
            f"OLS slope: `{slope:+.5f}` per month"
        )
        if w:
            wc = st.columns(len(w))
            for col, (model, wv) in zip(wc, w.items()):
                col.metric(model, f"{wv*100:.0f}%", f"MAE {maes.get(model,'?')}")
        st.divider()

# ── monthly quality outlook grid ───────────────────────────────────────────────
st.subheader("📅 Monthly Quality Outlook")
SC = {"Safe":"#22c55e","Moderate":"#f59e0b","Poor":"#ef4444","Unknown":"#94a3b8"}
PR = {"Poor":0,"Moderate":1,"Safe":2,"Unknown":3}

n_cols = min(horizon_months, 6)
chunks = [list(range(horizon_months))[i:i+n_cols] for i in range(0, horizon_months, n_cols)]

for chunk in chunks:
    cols = st.columns(len(chunk))
    for col, mo in zip(cols, chunk):
        month_label = (datetime.today() + relativedelta(months=mo+1)).strftime("%b %Y")
        statuses = [
            param_status(m, results[m]["forecast"][mo], schema)[0]
            for m in has_results if mo < len(results[m]["forecast"])
        ]
        overall = min(statuses, key=lambda s: PR.get(s,3)) if statuses else "Unknown"
        color   = SC[overall]
        col.markdown(
            f"<div style='background:{color}18;border:1.5px solid {color}44;"
            f"border-radius:10px;padding:10px 6px;text-align:center;margin-bottom:6px'>"
            f"<div style='font-size:.68rem;color:#64748b;font-weight:600'>{month_label}</div>"
            f"<div style='font-size:.88rem;font-weight:700;color:{color};margin-top:2px'>{overall}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── per-parameter charts ───────────────────────────────────────────────────────
st.subheader("📈 Forecast Charts")


def _ydomain(vals, pad=0.20):
    v = pd.Series(vals).dropna()
    if v.empty: return [0, 1]
    lo, hi = float(v.min()), float(v.max())
    margin = (hi - lo) * pad if hi != lo else max(abs(hi) * 0.15, 0.01)
    return [lo - margin, hi + margin]


for metric in has_results:
    res  = results[metric]
    lab  = metrics[metric].get("label", metric)
    unit = metrics[metric].get("unit", "")
    icon = metrics[metric].get("icon", "📊")
    hib  = metrics[metric].get("higher_is_better")

    # dataframes
    hist_s = res["historical"]
    hist   = pd.DataFrame({"date": pd.to_datetime(hist_s.index), "value": hist_s.values})
    hist["type"]  = "Historical"
    hist["lower"] = hist["value"]
    hist["upper"] = hist["value"]

    fcast = pd.DataFrame({
        "date":  pd.to_datetime(res["future_dates"]),
        "value": res["forecast"],
        "lower": res["lower"],
        "upper": res["upper"],
        "type":  "Forecast",
    })

    if fcast.empty:
        st.warning(f"No forecast for {lab}")
        continue

    last_hist_date = hist["date"].iloc[-1]
    last_hist_val  = float(hist["value"].iloc[-1])

    # bridge line: last historical → first forecast point
    bridge = pd.DataFrame({
        "date":  [last_hist_date, fcast["date"].iloc[0]],
        "value": [last_hist_val,  fcast["value"].iloc[0]],
    })

    split_rule = pd.DataFrame({"date": [last_hist_date]})
    y_title    = f"{lab} ({unit})" if unit else lab

    # change badge
    change = res["change"]
    good   = (change > 0 if hib is True else change < 0 if hib is False else None)
    bc     = "#22c55e" if good is True else "#ef4444" if good is False else "#94a3b8"
    mood   = "improving" if good is True else "worsening" if good is False else "changing"
    slope  = res["ensemble_info"].get("ols_slope_per_month", 0)
    slope_dir = "↗" if slope > 1e-6 else ("↘" if slope < -1e-6 else "→")
    method = res.get("method", "")

    st.markdown(
        f"### {icon} {lab} "
        f"<span style='background:{bc}22;color:{bc};padding:3px 12px;"
        f"border-radius:99px;font-size:.82rem;font-weight:600'>"
        f"{'↑' if change>0 else '↓'} {abs(change):.3f} {unit} · {mood}</span>"
        f"  <span style='color:#475569;font-size:.74rem'>"
        f"{slope_dir} {abs(slope):.4f}/mo · {method}</span>",
        unsafe_allow_html=True,
    )

    # ── PANEL 1: full history (compact) ──────────────────────────────────────
    all_vals    = list(hist["value"]) + list(fcast["value"]) + list(fcast["lower"]) + list(fcast["upper"])
    y_dom_full  = _ydomain(all_vals)
    shade_df    = pd.DataFrame({"x1": [last_hist_date], "x2": [fcast["date"].iloc[-1]]})

    p1 = (
        alt.layer(
            alt.Chart(shade_df).mark_rect(color="#38bdf8", opacity=0.07)
            .encode(x="x1:T", x2="x2:T"),

            alt.Chart(hist).mark_line(strokeWidth=1.5, color="#475569")
            .encode(
                x=alt.X("date:T", title=None,
                        axis=alt.Axis(format="%Y", grid=False, labelColor="#64748b")),
                y=alt.Y("value:Q", scale=alt.Scale(domain=y_dom_full, zero=False),
                        title=y_title,
                        axis=alt.Axis(grid=True, gridOpacity=0.1,
                                      labelColor="#64748b", titleColor="#64748b",
                                      tickCount=4)),
                tooltip=[alt.Tooltip("date:T", format="%b %Y"),
                         alt.Tooltip("value:Q", format=".3f")],
            ),

            alt.Chart(fcast).mark_area(color="#38bdf8", opacity=0.18)
            .encode(x="date:T",
                    y=alt.Y("lower:Q", scale=alt.Scale(domain=y_dom_full, zero=False)),
                    y2="upper:Q"),

            alt.Chart(fcast).mark_line(strokeWidth=2.5, color="#38bdf8")
            .encode(x="date:T",
                    y=alt.Y("value:Q", scale=alt.Scale(domain=y_dom_full, zero=False))),

            alt.Chart(split_rule).mark_rule(color="#38bdf8", strokeWidth=1.5,
                                             opacity=0.5, strokeDash=[4,3])
            .encode(x="date:T"),
        )
        .properties(height=120,
                    title=alt.TitleParams("Full history — shaded = forecast zone",
                                          color="#64748b", fontSize=11, anchor="start"))
        .configure_view(strokeWidth=0)
        .interactive()
    )

    # ── PANEL 2: zoomed — last 12mo history + forecast ────────────────────────
    zoom_start = last_hist_date - pd.DateOffset(months=12)
    hist_zoom  = hist[hist["date"] >= zoom_start].copy()

    zoom_vals  = (list(hist_zoom["value"]) + list(fcast["value"]) +
                  list(fcast["lower"]) + list(fcast["upper"]))
    y_dom_zoom = _ydomain(zoom_vals)
    annot_y    = y_dom_zoom[1]

    now_df  = pd.DataFrame({"date":[last_hist_date],"value":[annot_y],"text":["NOW"]})

    p2 = (
        alt.layer(
            # CI band
            alt.Chart(fcast).mark_area(color="#38bdf8", opacity=0.22)
            .encode(
                x="date:T",
                y=alt.Y("lower:Q", scale=alt.Scale(domain=y_dom_zoom, zero=False)),
                y2="upper:Q",
                tooltip=[
                    alt.Tooltip("date:T", format="%b %Y", title="Month"),
                    alt.Tooltip("value:Q", format=".3f", title=f"Forecast"),
                    alt.Tooltip("lower:Q", format=".3f", title="Lower 95%"),
                    alt.Tooltip("upper:Q", format=".3f", title="Upper 95%"),
                ],
            ),

            # historical line + dots
            alt.Chart(hist_zoom)
            .mark_line(strokeWidth=2.5, color="#94a3b8",
                       point=alt.OverlayMarkDef(color="#94a3b8", size=50, filled=True))
            .encode(
                x=alt.X("date:T", title="Month",
                        axis=alt.Axis(format="%b %Y", labelAngle=-35,
                                      grid=False, labelColor="#94a3b8")),
                y=alt.Y("value:Q", title=y_title,
                        scale=alt.Scale(domain=y_dom_zoom, zero=False),
                        axis=alt.Axis(grid=True, gridOpacity=0.12,
                                      labelColor="#94a3b8", titleColor="#94a3b8")),
                tooltip=[alt.Tooltip("date:T", format="%b %Y"),
                         alt.Tooltip("value:Q", format=".3f", title=f"Observed {lab}")],
            ),

            # bridge (dashed) from last hist to first forecast
            alt.Chart(bridge)
            .mark_line(strokeWidth=2, color="#38bdf8", strokeDash=[5,3])
            .encode(x="date:T",
                    y=alt.Y("value:Q", scale=alt.Scale(domain=y_dom_zoom, zero=False))),

            # forecast line + dots
            alt.Chart(fcast)
            .mark_line(strokeWidth=3, color="#38bdf8",
                       point=alt.OverlayMarkDef(color="#38bdf8", size=80,
                                                filled=True, stroke="#0ea5e9",
                                                strokeWidth=2))
            .encode(
                x="date:T",
                y=alt.Y("value:Q", scale=alt.Scale(domain=y_dom_zoom, zero=False)),
                tooltip=[
                    alt.Tooltip("date:T", format="%b %Y", title="Forecast Month"),
                    alt.Tooltip("value:Q", format=".3f", title=f"Forecast {lab}"),
                    alt.Tooltip("lower:Q", format=".3f", title="Lower 95%"),
                    alt.Tooltip("upper:Q", format=".3f", title="Upper 95%"),
                ],
            ),

            # NOW rule (amber dashed)
            alt.Chart(split_rule)
            .mark_rule(color="#f59e0b", strokeWidth=2, strokeDash=[6,3])
            .encode(x="date:T"),

            # NOW label
            alt.Chart(now_df)
            .mark_text(fontSize=10, fontWeight="bold", color="#f59e0b",
                       align="center", dy=-14)
            .encode(x="date:T",
                    y=alt.Y("value:Q", scale=alt.Scale(domain=y_dom_zoom, zero=False)),
                    text="text:N"),
        )
        .properties(height=300,
                    title=alt.TitleParams(
                        f"Last 12 months history + {horizon_months}-month forecast  "
                        f"(shaded = 95% CI)",
                        color="#94a3b8", fontSize=11, anchor="start"))
        .configure_view(strokeWidth=0)
        .configure_axis(labelColor="#94a3b8", titleColor="#94a3b8")
        .interactive()
    )

    st.altair_chart(p1,  use_container_width=True)
    st.altair_chart(p2,  use_container_width=True)
    st.markdown("---")

# ── forecast table ─────────────────────────────────────────────────────────────
st.subheader("📋 Monthly Forecast Table")

rows = []
for mo in range(horizon_months):
    month_label = (datetime.today() + relativedelta(months=mo+1)).strftime("%b %Y")
    row = {"Month": month_label}
    for m in has_results:
        lab  = metrics[m].get("label", m)
        unit = metrics[m].get("unit", "")
        fc, lo, hi = results[m]["forecast"], results[m]["lower"], results[m]["upper"]
        if mo < len(fc):
            row[lab] = f"{fc[mo]:.3f} {unit}  [{lo[mo]:.2f}–{hi[mo]:.2f}]".strip()
    rows.append(row)

st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

# ── change summary ─────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader(f"📉 Change: Current → {end_date.strftime('%b %Y')}")

chg_cols = st.columns(min(len(has_results), 5))
for col, m in zip(chg_cols, has_results):
    res_m = results[m]
    lab   = metrics[m].get("label", m)
    unit  = metrics[m].get("unit", "")
    hib   = metrics[m].get("higher_is_better")
    col.metric(
        label=f"{metrics[m].get('icon','')} {lab}",
        value=f"{res_m['end_value']:.3f} {unit}".strip(),
        delta=f"{res_m['change']:+.3f}",
        delta_color="inverse" if hib is False else "normal",
    )

# ── AI analysis ────────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🌊 AI Degradation Analysis")

if st.button("🤖 Generate AI Analysis", type="primary"):
    fc_summary = {
        m: {"current": results[m]["current"], "day7": results[m]["end_value"],
            "change":  results[m]["change"],  "forecast": results[m]["forecast"]}
        for m in has_results
    }
    with st.spinner("Generating…"):
        analysis = degradation_forecast_analysis(schema, fc_summary)
    st.session_state["forecast_analysis"] = analysis

if "forecast_analysis" in st.session_state:
    st.markdown(
        f"<div style='background:#0f172a;border-left:4px solid #f59e0b;"
        f"border-radius:0 10px 10px 0;padding:16px 20px;"
        f"font-size:.92rem;color:#e2e8f0;line-height:1.8'>"
        f"🌊 {st.session_state['forecast_analysis']}</div>",
        unsafe_allow_html=True,
    )