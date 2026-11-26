
import streamlit as st
import pandas as pd
import plotly.express as px


def render_home(models=None, result=None):
    """
    Display the ThreatTrace dashboard overview.

    Parameters:
        models: Loaded model information (optional).
        result: DataFrame containing analyzed network traffic (optional).
    """

    st.title("ThreatTrace - Network Security Overview")

    st.write(
        "Monitor network traffic, identify suspicious activity, "
        "and understand potential cyber threats."
    )

    # -------------------------------------------------
    # CASE 1: No traffic analysis has been performed
    # -------------------------------------------------
    if result is None or result.empty:

        st.info(
            "No traffic analysis is available yet. "
            "Go to the Analyze Traffic page and upload a compatible CSV file."
        )

        st.subheader("System Status")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Dashboard", "Ready")

        with col2:
            st.metric("Binary Model", "Loaded")

        with col3:
            st.metric("Multiclass Model", "Loaded")

        st.subheader("Available Features")

        st.markdown(
            """
            - **Traffic Analysis:** Upload network traffic CSV files.
            - **Threat Detection:** Identify benign and malicious traffic.
            - **Attack Classification:** Predict the type of detected attack.
            - **Explainable AI:** Understand why the model made a prediction.
            """
        )

        return

    # -------------------------------------------------
    # CASE 2: Display analysis results
    # -------------------------------------------------

    required_columns = ["Threat Status", "Attack Type"]

    missing_columns = [
        column for column in required_columns
        if column not in result.columns
    ]

    if missing_columns:
        st.error(
            "Analysis results are missing required columns: "
            + ", ".join(missing_columns)
        )
        return

    total_flows = len(result)

    suspicious_flows = (
        result["Threat Status"].astype(str).str.lower() != "benign"
    ).sum()

    benign_flows = total_flows - suspicious_flows

    attack_rate = (
        suspicious_flows / total_flows * 100
        if total_flows > 0 else 0
    )

    # KPI cards
    st.subheader("Traffic Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Flows", f"{total_flows:,}")

    with col2:
        st.metric("Suspicious Flows", f"{suspicious_flows:,}")

    with col3:
        st.metric("Benign Flows", f"{benign_flows:,}")

    with col4:
        st.metric("Threat Rate", f"{attack_rate:.2f}%")

    # -------------------------------------------------
    # Charts
    # -------------------------------------------------

    st.subheader("Traffic Distribution")

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        traffic_data = pd.DataFrame({
            "Traffic Type": ["Benign", "Suspicious"],
            "Count": [benign_flows, suspicious_flows]
        })

        fig_pie = px.pie(
            traffic_data,
            names="Traffic Type",
            values="Count",
            title="Benign vs Suspicious Traffic",
            hole=0.45
        )

        st.plotly_chart(fig_pie, use_container_width=True)

    with chart_col2:
        attack_data = (
            result["Attack Type"]
            .fillna("Unknown")
            .astype(str)
            .value_counts()
            .reset_index()
        )

        attack_data.columns = ["Attack Type", "Count"]

        fig_bar = px.bar(
            attack_data,
            x="Attack Type",
            y="Count",
            title="Traffic by Predicted Class",
            color="Count"
        )

        fig_bar.update_layout(xaxis_tickangle=-35)

        st.plotly_chart(fig_bar, use_container_width=True)

    # -------------------------------------------------
    # Recent detection records
    # -------------------------------------------------

    st.subheader("Recent Traffic Records")

    display_columns = [
        column for column in [
            "Threat Status",
            "Attack Type",
            "Attack Probability",
            "Classification Confidence"
        ]
        if column in result.columns
    ]

    st.dataframe(
        result[display_columns].head(10),
        use_container_width=True,
        hide_index=True
    )
