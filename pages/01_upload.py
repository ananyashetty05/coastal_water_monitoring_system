import streamlit as st
from core.processor import parse_csv, generate_sample_data
from core.state import set_df

st.title("Upload Data")

file = st.file_uploader("Upload CSV")

if file:
    df = parse_csv(file)
    set_df(df)
    st.success("File uploaded!")

if st.button("Generate Sample Data"):
    df = generate_sample_data()
    set_df(df)
    st.success("Sample data loaded!")