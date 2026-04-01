import streamlit as st
import pandas as pd

METRIC_META = {
    "do":             {"label": "Dissolved Oxygen", "unit": "mg/L", "icon": "💧"},
    "ph":             {"label": "pH Level",          "unit": "",     "icon": "⚗️"},
    "ammonia":        {"label": "Ammonia",            "unit": "mg/L", "icon": "🧪"},
    "temp":           {"label": "Temperature",        "unit": "°C",   "icon": "🌡️"},
    "bod":            {"label": "BOD",                "unit": "mg/L", "icon": "🌊"},
    "nitrogen":       {"label": "Nitrogen",           "unit": "mg/L", "icon": "🌿"},
    "nitrate":        {"label": "Nitrate",            "unit": "mg/L", "icon": "🔬"},
    "orthophosphate": {"label": "Orthophosphate",     "unit": "mg/L", "icon": "⚡"},
    "ccme_values":    {"label": "CCME Score",         "unit": "",     "icon": "📊"},
}

THRESHOLDS = {
    "do":             {"safe": (6,    None), "poor": (4,    None)},
    "ph":             {"safe": (6.5,  8.5),  "poor": (5,    9.5)},
    "ammonia":        {"safe": (None, 0.5),  "poor": (None, 1.0)},
    "temp":           {"safe": (None, 28),   "poor": (None, 35)},
    "bod":            {"safe": (None, 3),    "poor": (None, 6)},
    "nitrogen":       {"safe": (None, 1),    "poor": (None, 5)},
    "nitrate":        {"safe": (None, 10),   "poor": (None, 50)},
    "orthophosphate": {"safe": (None, 0.1),  "poor": (None, 0.5)},
    "ccme_values":    {"safe": (80,   None), "poor": (45,   None)},
}


def _status_label(key, value) -> str:
    if value is None or key not in THRESHOLDS:
        return "—"
    t = THRESHOLDS[key]
    lo_safe, hi_safe = t["safe"]
    lo_poor, hi_poor = t["poor"]
    is_poor = (lo_poor is not None and value <= lo_poor) or \
              (hi_poor is not None and value >= hi_poor)
    is_safe = (lo_safe is None or value >= lo_safe) and \
              (hi_safe is None or value <= hi_safe)
    if is_poor: return "🔴 Poor"
    if is_safe: return "🟢 Safe"
    return "🟡 Moderate"


def render(stats: dict):
    if not stats:
        st.info("No statistics available. Upload data or generate sample data first.")
        return

    st.markdown("#### 📋 Summary Statistics")

    rows = []
    for key, meta in METRIC_META.items():
        stat = stats.get(key)
        if not isinstance(stat, dict):
            min_val = avg_val = max_val = None
        else:
            min_val = stat.get("min")
            avg_val = stat.get("mean")
            max_val = stat.get("max")

        unit = meta["unit"]
        fmt  = lambda v: f"{v:.2f} {unit}".strip() if v is not None else "—"

        rows.append({
            "":       meta["icon"],
            "Metric": meta["label"],
            "Min":    fmt(min_val),
            "Avg":    fmt(avg_val),
            "Max":    fmt(max_val),
            "Status": _status_label(key, avg_val),
        })

    st.dataframe(
        pd.DataFrame(rows),
        hide_index=True,
        width="stretch",
    )

    st.caption("🟢 Safe · 🟡 Moderate · 🔴 Poor  — based on average value")
