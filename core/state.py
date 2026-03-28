import streamlit as st

def set_df(df):
    st.session_state["data"] = df

def get_df():
    return st.session_state.get("data", None)