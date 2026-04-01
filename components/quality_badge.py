import streamlit as st

STATUS_CONFIG = {
    "Safe": {
        "fn":      st.success,
        "icon":    "✅",
        "label":   "Safe Water",
        "color":   "#22c55e",
        "bg":      "#d9f99d",
        "desc":    "Water quality meets all safety standards.",
    },
    "Moderate": {
        "fn":      st.warning,
        "icon":    "⚠️",
        "label":   "Moderate Quality",
        "color":   "#f59e0b",
        "bg":      "#fde68a",
        "desc":    "Some parameters are outside optimal ranges.",
    },
    "Poor": {
        "fn":      st.error,
        "icon":    "❌",
        "label":   "Poor Quality",
        "color":   "#ef4444",
        "bg":      "#fecaca",
        "desc":    "Water quality is below safe thresholds.",
    },
}

# Map CCME WQI labels → internal status
_CCME_TO_STATUS = {
    "Excellent": "Safe",
    "Good":      "Safe",
    "Marginal":  "Moderate",
    "Fair":      "Moderate",
    "Poor":      "Poor",
    "Unknown":   "Poor",
}


def render(classification: dict):
    """
    Render an enhanced water quality badge.

    Expected classification dict keys:
        status   → CCME label (Excellent/Good/Marginal/Fair/Poor)
                   OR internal label (Safe/Moderate/Poor)
        messages → list[str]   (optional)
        score    → float       (optional CCME score)
    """
    if not classification:
        st.info("No classification data available.")
        return

    raw_status = classification.get("status", "Poor")
    # normalise CCME labels to Safe/Moderate/Poor
    status   = _CCME_TO_STATUS.get(raw_status, raw_status)
    status   = status if status in STATUS_CONFIG else "Poor"
    messages = classification.get("messages", [])
    score    = classification.get("score")
    cfg      = STATUS_CONFIG.get(status, STATUS_CONFIG["Poor"])

    score_text = f" · CCME Score: {score:.1f}" if score is not None else ""
    ccme_label  = f" ({raw_status})" if raw_status not in ("Safe", "Moderate", "Poor") else ""

    # ── Main badge ───────────────────────────────────────────────────────────
    st.markdown(f"""
    <style>
      .cw-badge {{
        display: flex;
        align-items: center;
        gap: 14px;
        padding: 16px 20px;
        border-radius: 10px;
        border: 1px solid {cfg['color']}44;
        background: {cfg['bg']};
        margin-bottom: 8px;
      }}
      .cw-badge-icon  {{ font-size: 2rem; line-height:1; }}
      .cw-badge-title {{
        font-size: 1.15rem; font-weight: 700;
        color: #000000; margin: 0 0 2px;
      }}
      .cw-badge-desc  {{ font-size: 0.82rem; color: #000000; margin: 0; }}
      .cw-msg-list {{
        list-style: none; padding: 0; margin: 6px 0 0;
        display: flex; flex-direction: column; gap: 5px;
      }}
      .cw-msg-item {{
        font-size: 0.81rem;
        background: {cfg['bg']};
        border-left: 3px solid {cfg['color']};
        padding: 5px 10px;
        border-radius: 0 6px 6px 0;
        color: #000000;
      }}
    </style>

    <div class="cw-badge">
      <span class="cw-badge-icon">{cfg['icon']}</span>
      <div>
        <p class="cw-badge-title">{cfg['label']}{ccme_label}{score_text}</p>
        <p class="cw-badge-desc">{cfg['desc']}</p>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-parameter messages (only if present) ─────────────────────────────
    if messages:
        items = "".join(
            f'<li class="cw-msg-item">⚡ {msg}</li>'
            for msg in messages
        )
        st.markdown(
            f'<ul class="cw-msg-list">{items}</ul>',
            unsafe_allow_html=True
        )
