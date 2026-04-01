import streamlit as st

METRIC_CONFIG = {
    "do":             {"label": "Dissolved O₂",  "unit": "mg/L", "icon": "💧",
                       "safe": (6,    None), "poor": (4,    None)},
    "ph":             {"label": "pH Level",       "unit": "",     "icon": "⚗️",
                       "safe": (6.5,  8.5),  "poor": (5,    9.5)},
    "ammonia":        {"label": "Ammonia",         "unit": "mg/L", "icon": "🧪",
                       "safe": (None, 0.5),  "poor": (None, 1.0)},
    "temp":           {"label": "Temperature",     "unit": "°C",   "icon": "🌡️",
                       "safe": (None, 28),   "poor": (None, 35)},
    "bod":            {"label": "BOD",             "unit": "mg/L", "icon": "🌊",
                       "safe": (None, 3),    "poor": (None, 6)},
    "nitrogen":       {"label": "Nitrogen",        "unit": "mg/L", "icon": "🌿",
                       "safe": (None, 1),    "poor": (None, 5)},
    "nitrate":        {"label": "Nitrate",         "unit": "mg/L", "icon": "🔬",
                       "safe": (None, 10),   "poor": (None, 50)},
    "orthophosphate": {"label": "Orthophosphate",  "unit": "mg/L", "icon": "⚡",
                       "safe": (None, 0.1),  "poor": (None, 0.5)},
}


def _status(key: str, value: float) -> tuple[str, str]:
    cfg     = METRIC_CONFIG[key]
    lo_safe, hi_safe = cfg["safe"]
    lo_poor, hi_poor = cfg["poor"]

    is_poor = (lo_poor is not None and value <= lo_poor) or \
              (hi_poor is not None and value >= hi_poor)
    is_safe = (lo_safe is None or value >= lo_safe) and \
              (hi_safe is None or value <= hi_safe)

    if is_poor: return "Poor",     "#ef4444"
    if is_safe: return "Safe",     "#22c55e"
    return          "Moderate",    "#f59e0b"


def render(stats: dict):
    if not stats:
        st.info("No metrics available. Upload data or generate sample data first.")
        return

    st.markdown("""
    <style>
      .cw-card {
        border-radius: 10px; padding: 16px 14px 12px;
        background: #0f172a; border: 1px solid #1e293b;
        text-align: center; transition: border-color .2s;
      }
      .cw-card:hover { border-color: #334155; }
      .cw-card-icon  { font-size: 1.5rem; margin-bottom: 4px; }
      .cw-card-label {
        font-size: 0.72rem; font-weight: 600; letter-spacing: .07em;
        text-transform: uppercase; color: #64748b; margin-bottom: 8px;
      }
      .cw-card-value { font-size: 1.6rem; font-weight: 800; line-height: 1; margin-bottom: 2px; }
      .cw-card-unit  { font-size: 0.72rem; color: #475569; margin-bottom: 8px; }
      .cw-pill {
        display: inline-block; font-size: 0.68rem; font-weight: 700;
        letter-spacing: .05em; text-transform: uppercase;
        padding: 2px 9px; border-radius: 99px;
      }
    </style>
    """, unsafe_allow_html=True)

    keys = list(METRIC_CONFIG.keys())
    # render in two rows of 4
    for row_keys in [keys[:4], keys[4:]]:
        cols = st.columns(len(row_keys))
        for col, key in zip(cols, row_keys):
            cfg      = METRIC_CONFIG[key]
            raw_stat = stats.get(key)
            raw      = raw_stat.get("latest") if isinstance(raw_stat, dict) else None

            with col:
                if raw is None:
                    st.markdown(
                        f'<div class="cw-card">'
                        f'<div class="cw-card-icon">{cfg["icon"]}</div>'
                        f'<div class="cw-card-label">{cfg["label"]}</div>'
                        f'<div class="cw-card-value" style="color:#475569">—</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                    continue

                value         = round(raw, 2)
                status, color = _status(key, value)
                pill_bg       = color + "22"

                st.markdown(f"""
                <div class="cw-card" style="border-top:3px solid {color}">
                  <div class="cw-card-icon">{cfg['icon']}</div>
                  <div class="cw-card-label">{cfg['label']}</div>
                  <div class="cw-card-value" style="color:{color}">{value}</div>
                  <div class="cw-card-unit">{cfg['unit']}</div>
                  <span class="cw-pill" style="background:{pill_bg};color:{color}">{status}</span>
                </div>
                """, unsafe_allow_html=True)
