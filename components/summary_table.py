import streamlit as st
import pandas as pd

def render(stats):
    df = pd.DataFrame(stats.items(), columns=["Metric", "Value"])
    st.dataframe(df)