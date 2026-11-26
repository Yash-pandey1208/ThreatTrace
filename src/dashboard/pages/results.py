
import streamlit as st
import pandas as pd

from src.dashboard.model_service import analyze_dataframe


def render_results(models=None):
    """
    Traffic analysis page:
    - Upload network traffic CSV
    - Validate input
    - Analyze traffic using trained models
    - Display predictions
    - Download results
    """

    st.title("Network Traffic Analysis")

    st.write(
        "Upload a CSV file containing the 65 engineered network traffic "
        "features required by the trained models."
    )

    uploaded_file = st.file_uploader(
        "Upload Network Traffic CSV",
        type=["csv"],
        help="Upload a CSV containing the model's required feature columns."
    )

    if uploaded_file is None:
        st.info("Please upload a CSV file to begin traffic analysis.")
        return

    # Read uploaded CSV
    try:
        input_df = pd.read_csv(uploaded_file)

    except Exception as error:
        st.error(f"Unable to read CSV file: {error}")
        return

    if input_df.empty:
        st.warning("The uploaded CSV file is empty.")
        return

    st.subheader("Uploaded Data Preview")

    st.write(f"Total records: {len(input_df):,}")
    st.dataframe(input_df.head(10), use_container_width=True)

    # Run analysis
    if st.button("Analyze Network Traffic", type="primary"):

        if models is None:
            st.error("Models are not loaded. Please check model configuration.")
            return

        try:
            with st.spinner("Analyzing network traffic..."):

                analysis_result, warnings = analyze_dataframe(
                    input_df,
                    models
                )

            # Store results in session state
            st.session_state["analysis_result"] = analysis_result

            st.session_state["analysis_warnings"] = warnings

            st.success("Traffic analysis completed successfully!")

        except Exception as error:
            st.error(f"Traffic analysis failed: {error}")
            return

    # Display saved results
    analysis_result = st.session_state.get("analysis_result")

    if analysis_result is None:
        return

    st.divider()

    st.subheader("Traffic Analysis Results")

    # Display warnings
    warnings = st.session_state.get("analysis_warnings", [])

    if warnings:
        with st.expander("Data Preparation Warnings"):
            for warning in warnings:
                st.warning(str(warning))

    # Display prediction metrics
    total_records = len(analysis_result)

    if "Threat Status" in analysis_result.columns:

        suspicious = (
            analysis_result["Threat Status"]
            .astype(str)
            .str.lower() != "benign"
        ).sum()

        benign = total_records - suspicious

        col1, col2, col3 = st.columns(3)

        col1.metric("Total Records", f"{total_records:,}")
        col2.metric("Suspicious Records", f"{suspicious:,}")
        col3.metric("Benign Records", f"{benign:,}")

    # Results table
    st.dataframe(
        analysis_result,
        use_container_width=True,
        hide_index=True
    )

    # Download results
    csv_data = analysis_result.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Analysis Results",
        data=csv_data,
        file_name="threattrace_analysis_results.csv",
        mime="text/csv"
    )
