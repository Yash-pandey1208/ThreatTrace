
import streamlit as st
import pandas as pd
import plotly.express as px


def render_local_shap_chart(explanation, max_features=10):
    """Display SHAP contributions for one traffic record."""

    if explanation is None or explanation.empty:
        st.warning("No local SHAP explanation is available.")
        return

    required = ["Feature", "SHAP value"]

    if not all(column in explanation.columns for column in required):
        st.error("Invalid local SHAP explanation format.")
        return

    chart_data = explanation.copy()

    chart_data["SHAP value"] = pd.to_numeric(
        chart_data["SHAP value"], errors="coerce"
    )

    chart_data = chart_data.dropna(subset=["SHAP value"])
    chart_data["Absolute impact"] = chart_data["SHAP value"].abs()

    chart_data = chart_data.nlargest(
        max_features, "Absolute impact"
    ).sort_values("SHAP value", ascending=True)

    fig = px.bar(
        chart_data,
        x="SHAP value",
        y="Feature",
        orientation="h",
        title="Feature Contributions for This Prediction",
        color="SHAP value",
        color_continuous_midpoint=0
    )

    fig.update_layout(
        xaxis_title="SHAP value",
        yaxis_title="Network feature"
    )

    st.plotly_chart(fig, use_container_width=True)


def render_global_shap_chart(importance, max_features=15):
    """Display global SHAP feature importance."""

    if importance is None or importance.empty:
        st.warning("No global feature importance is available.")
        return

    required = ["Feature", "Mean |SHAP|"]

    if not all(column in importance.columns for column in required):
        st.error("Invalid global SHAP importance format.")
        return

    chart_data = importance.copy()

    chart_data["Mean |SHAP|"] = pd.to_numeric(
        chart_data["Mean |SHAP|"], errors="coerce"
    )

    chart_data = chart_data.dropna(subset=["Mean |SHAP|"])

    chart_data = chart_data.nlargest(
        max_features, "Mean |SHAP|"
    ).sort_values("Mean |SHAP|", ascending=True)

    fig = px.bar(
        chart_data,
        x="Mean |SHAP|",
        y="Feature",
        orientation="h",
        title="Global Network Feature Importance"
    )

    fig.update_layout(
        xaxis_title="Mean absolute SHAP value",
        yaxis_title="Network feature"
    )

    st.plotly_chart(fig, use_container_width=True)
