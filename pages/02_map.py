import streamlit as st
from core.state import get_df
from core.processor import get_location_summaries

st.title("Map View")

df = get_df()

if df is None:
    st.warning("Upload data first")
else:
    locs = get_location_summaries(df)
    st.map(locs.rename(columns={"lat": "latitude", "lon": "longitude"}))