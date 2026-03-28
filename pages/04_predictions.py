import streamlit as st
from core.state import get_df
from core.predictor import predict

st.title("Predictions")

df = get_df()

if df is None:
    st.warning("Upload data first")
else:
    location = st.selectbox("Select Location", df["location"].unique())

    pred = predict(df, location)

    st.line_chart(pred.set_index("date"))