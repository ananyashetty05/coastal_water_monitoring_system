import streamlit as st
from core.state import get_df
from core.processor import get_stats
from core.classifier import classify
from components.metric_row import render as metrics
from components.quality_badge import render as badge

st.title("Analytics")

df = get_df()

if df is None:
    st.warning("Upload data first")
else:
    location = st.selectbox("Select Location", df["location"].unique())

    stats = get_stats(df, location)
    metrics(stats)

    result = classify(**stats)
    badge(result)

    d = df[df["location"] == location]
    st.line_chart(d.set_index("date")[["do", "ph"]])