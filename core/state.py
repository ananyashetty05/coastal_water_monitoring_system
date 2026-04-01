"""
core/state.py
Centralized Streamlit session-state helpers.
"""

from __future__ import annotations

import streamlit as st

_DATA_KEY = "data"
_ML_BUNDLE_KEY = "ml_bundle"
_ML_BENCHMARK_KEY = "ml_benchmark"
_INTERPRETABILITY_KEY = "interpretability"


def set_df(df):
    """
    Store dataframe and invalidate model-dependent cached artifacts.
    """
    st.session_state[_DATA_KEY] = df
    clear_ml_state()


def get_df():
    return st.session_state.get(_DATA_KEY, None)


def clear_ml_state():
    """
    Clear all ML artifacts so they can be rebuilt against the current data.
    """
    for key in (_ML_BUNDLE_KEY, _ML_BENCHMARK_KEY, _INTERPRETABILITY_KEY):
        st.session_state.pop(key, None)


def set_ml_bundle(bundle):
    st.session_state[_ML_BUNDLE_KEY] = bundle
    if isinstance(bundle, dict):
        st.session_state[_ML_BENCHMARK_KEY] = bundle.get("benchmark")
        st.session_state[_INTERPRETABILITY_KEY] = {
            "feature_importance": bundle.get("feature_importance", {}),
            "correlation_matrix": bundle.get("correlation_matrix"),
        }


def get_ml_bundle():
    return st.session_state.get(_ML_BUNDLE_KEY, None)


def get_ml_benchmark():
    return st.session_state.get(_ML_BENCHMARK_KEY, None)


def get_interpretability():
    return st.session_state.get(_INTERPRETABILITY_KEY, None)
