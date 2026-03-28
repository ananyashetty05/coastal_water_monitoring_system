import streamlit as st

st.set_page_config(page_title="CoastalWatch", layout="wide")

# Load CSS
def load_css():
    with open("assets/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css()

st.title("🌊 Coastal Water Quality Monitoring System")

st.markdown("""
Welcome to CoastalWatch!

Use the sidebar to:
- Upload data
- Explore map
- Analyze water quality
- View predictions
""")