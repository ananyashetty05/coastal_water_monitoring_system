import streamlit as st

def render(classification):
    status = classification["status"]

    if status == "Safe":
        st.success("✅ Safe Water")
    elif status == "Moderate":
        st.warning("⚠️ Moderate Quality")
    else:
        st.error("❌ Poor Quality")