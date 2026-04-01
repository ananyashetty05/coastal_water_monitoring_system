import streamlit as st

st.set_page_config(
    page_title="CoastalWatch",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)


def load_css():
    try:
        with open("assets/style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        pass


load_css()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown(
    """
    <div style="
        background: linear-gradient(135deg, #0a2342 0%, #1a5276 60%, #0e6655 100%);
        border-radius: 16px;
        padding: 3rem 3rem 2.5rem;
        margin-bottom: 2rem;
        color: white;
    ">
        <h1 style="font-size:2.6rem;margin:0;font-weight:800;letter-spacing:-1px;">
            🌊 CoastalWatch
        </h1>
        <p style="font-size:1.1rem;margin-top:.6rem;opacity:.85;max-width:680px;">
            Water quality monitoring across coastal, transitional, and estuarine
            waterbodies in various locations.
            Data powered by the CCME Water Quality Index.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ── Navigation cards ──────────────────────────────────────────────────────────
cols = st.columns(4)
cards = [
    ("📤", "Upload Data",    "Load the dataset CSV or re-upload an updated version."),
    ("🗺️", "Map View",      "Explore monitoring stations geographically by WQI rating."),
    ("📊", "Analytics",      "Dive into parameter trends, CCME scores, and quality breakdowns."),
    ("🔮", "Predictions",    "7-day parameter forecasts using linear regression."),
]
for col, (icon, title, desc) in zip(cols, cards):
    with col:
        st.markdown(
            f"""
            <div style="
                background:#f0f7ff;border:1px solid #d0e8ff;
                border-radius:12px;padding:1.4rem 1.2rem;
                height:165px;display:flex;flex-direction:column;gap:.5rem;
            ">
                <span style="font-size:2rem;">{icon}</span>
                <strong style="font-size:.97rem;color:#0a2342;">{title}</strong>
                <p style="font-size:.82rem;color:#4a6580;margin:0;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ── Dataset overview ──────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("📦 About the Dataset")

c1, c2, c3, c4 = st.columns(4)
c1.info("**Countries**\nIreland · England · China · USA")
c2.info("**Parameters**\nDO · pH · Ammonia · BOD\nOrthophosphate · Temp\nNitrogen · Nitrate")
c3.info("**Quality Index**\nCCME WQI\nExcellent → Good → Marginal → Fair → Poor")
c4.info("**Waterbody Types**\nCoastal · Transitional\nEstuarine · Sea Water")

# ── Quick-start ───────────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("🚀 Quick Start")
steps = st.columns(4)
for col, num, label in zip(
    steps, ["1", "2", "3", "4"],
    [
        "Go to Upload Data and load `data.csv`.",
        "Open Map View to see stations colour-coded by CCME WQI.",
        "Visit Analytics — pick a location to explore all parameters.",
        "Head to Predictions for a 7-day CCME score forecast.",
    ],
):
    with col:
        st.markdown(
            f"""<div style="display:flex;gap:.7rem;align-items:flex-start;">
                <div style="background:#0a2342;color:white;border-radius:50%;
                    min-width:28px;height:28px;display:flex;align-items:center;
                    justify-content:center;font-weight:700;font-size:.85rem;
                    margin-top:.1rem;">{num}</div>
                <p style="margin:0;font-size:.88rem;color:#2c3e50;">{label}</p>
            </div>""",
            unsafe_allow_html=True,
        )

st.markdown(
    "<br><p style='text-align:center;color:#aaa;font-size:.78rem;'>"
    "CoastalWatch · CCME Water Quality Index "
    "</p>",
    unsafe_allow_html=True,
)