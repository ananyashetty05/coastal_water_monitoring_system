import streamlit as st

def render(stats):
    cols = st.columns(5)

    cols[0].metric("DO", round(stats["do"], 2))
    cols[1].metric("pH", round(stats["ph"], 2))
    cols[2].metric("Sulphur", round(stats["sulphur"], 2))
    cols[3].metric("Temp", round(stats["temp"], 2))
    cols[4].metric("Turbidity", round(stats["turbidity"], 2))