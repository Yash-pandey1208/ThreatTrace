
import streamlit as st
import pandas as pd

from src.explainability import shap_explainer

from src.dashboard.components.shap_chart import (
    render_local_shap_chart,
    render_global_shap_chart
)


# =====================================================
# SHAP CALCULATION HELPERS
# =====================================================

def calculate_local_explanation(model, row, features):
    """
    Calculate SHAP values for one network traffic record.

    Supports both explain_row() and explain_prediction()
    implementations in shap_explainer.py.
    """

    if hasattr(shap_explainer, "explain_row"):

        output = shap_explainer.explain_row(
            model,
            row,
            features,
            class_index=1
        )

    elif hasattr(shap_explainer, "explain_prediction"):

        output = shap_explainer.explain_prediction(
            model,
            row,
            feature_names=features,
            class_index=1
        )

    else:
        raise AttributeError(
            "No supported local SHAP function was found "
            "in shap_explainer.py."
        )

    # Some implementations return (DataFrame, expected_value).
    if isinstance(output, tuple):
        output = output[0]

    if not isinstance(output, pd.DataFrame):
        raise TypeError("SHAP explanation must be a pandas DataFrame.")

    output = output.copy()

    # Standardize column names for the dashboard.
    output = output.rename(
        columns={
            "SHAP Value": "SHAP value",
            "Absolute Impact": "Absolute impact",
            "Feature Value": "Value"
        }
    )

    required = ["Feature", "SHAP value"]

    if not all(column in output.columns for column in required):
        raise ValueError(
            "The local SHAP result is missing required columns."
        )

    output["SHAP value"] = pd.to_numeric(
        output["SHAP value"],
        errors="coerce"
    )

    output = output.dropna(subset=["SHAP value"])

    if "Absolute impact" not in output.columns:
        output["Absolute impact"] = output["SHAP value"].abs()

    if "Value" not in output.columns:
        output["Value"] = None

    output["Absolute impact"] = output["SHAP value"].abs()

    return output.sort_values(
        "Absolute impact",
        ascending=False
    ).reset_index(drop=True)


def calculate_global_explanation(model, sample, features):
    """
    Calculate global SHAP feature importance.

    Supports explain_global() and
    get_global_feature_importance().
    """

    if hasattr(shap_explainer, "explain_global"):

        importance = shap_explainer.explain_global(
            model,
            sample,
            features
        )

    elif hasattr(shap_explainer, "get_global_feature_importance"):

        importance = shap_explainer.get_global_feature_importance(
            model,
            sample,
            feature_names=features,
            max_samples=500
        )

    else:
        raise AttributeError(
            "No supported global SHAP function was found."
        )

    if not isinstance(importance, pd.DataFrame):
        raise TypeError(
            "Global SHAP importance must be a pandas DataFrame."
        )

    importance = importance.copy()

    importance = importance.rename(
        columns={
            "SHAP Importance": "Mean |SHAP|"
        }
    )

    if not all(
        column in importance.columns
        for column in ["Feature", "Mean |SHAP|"]
    ):
        raise ValueError(
            "Global SHAP result is missing required columns."
        )

    importance["Mean |SHAP|"] = pd.to_numeric(
        importance["Mean |SHAP|"],
        errors="coerce"
    )

    importance = importance.dropna(
        subset=["Mean |SHAP|"]
    )

    return importance.sort_values(
        "Mean |SHAP|",
        ascending=False
    ).reset_index(drop=True)


# =====================================================
# PLAIN-LANGUAGE LOCAL EXPLANATION
# =====================================================

def create_directional_summary(explanation, predicted_status):
    """
    Explain whether individual feature contributions
    push the binary model towards or away from ATTACK.

    Assumes the binary model's positive class is ATTACK (class 1).
    Positive SHAP values push the ATTACK-class model output higher.
    Negative SHAP values push it lower.
    """

    attack_features = explanation[
        explanation["SHAP value"] > 0
    ].head(3)

    benign_features = explanation[
        explanation["SHAP value"] < 0
    ].head(3)

    st.subheader("What influenced this prediction?")

    st.write(
        f"The binary model classified this traffic record as "
        f"**{predicted_status}**."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Features pushing towards ATTACK**")

        if attack_features.empty:
            st.success(
                "No positive SHAP contributions were found "
                "among the displayed features."
            )
        else:
            for _, item in attack_features.iterrows():
                st.write(
                    f"**{item['Feature']}** "
                    f"({item['SHAP value']:+.4f})"
                )

    with col2:
        st.markdown("**Features pushing away from ATTACK**")

        if benign_features.empty:
            st.info(
                "No negative SHAP contributions were found "
                "among the displayed features."
            )
        else:
            for _, item in benign_features.iterrows():
                st.write(
                    f"**{item['Feature']}** "
                    f"({item['SHAP value']:+.4f})"
                )

    st.caption(
        "Positive SHAP values push the binary model's ATTACK-class "
        "output higher. Negative values push it lower. These values "
        "describe model influence, not proof of malicious activity."
    )


# =====================================================
# MAIN EXPLAINABILITY PAGE
# =====================================================

def render_explanation(models=None, result=None):

    st.title("Explainable AI - SHAP Analysis")

    st.write(
        "Understand why ThreatTrace classified a network flow "
        "as benign or malicious."
    )

    # -------------------------------------------------
    # CHECK ANALYSIS RESULTS
    # -------------------------------------------------

    if result is None or result.empty:

        st.info(
            "No analysis results are available. "
            "Go to Analyze Traffic and upload a CSV file first."
        )
        return

    if models is None or "binary" not in models:

        st.error("The binary XGBoost model is not available.")
        return

    # -------------------------------------------------
    # EXTRACT BINARY MODEL
    # -------------------------------------------------

    try:
        binary_model = models["binary"]["booster"]
        features = models["binary"]["features"]

    except (KeyError, TypeError) as error:

        st.error(f"Could not access the binary model: {error}")
        return

    # Verify the original model features exist.
    missing_features = [
        column for column in features
        if column not in result.columns
    ]

    if missing_features:

        st.error(
            f"{len(missing_features)} required model features "
            "are missing from the analysis results."
        )
        return

    # -------------------------------------------------
    # PREPARE FEATURE DATA
    # -------------------------------------------------

    feature_data = result.loc[:, features].copy()

    feature_data = feature_data.apply(
        pd.to_numeric,
        errors="coerce"
    )

    feature_data = (
        feature_data
        .replace(
            [float("inf"), float("-inf")],
            float("nan")
        )
        .fillna(0.0)
    )

    # -------------------------------------------------
    # SECTION 1: SELECT TRAFFIC RECORD
    # -------------------------------------------------

    st.divider()
    st.subheader("1. Individual Prediction Explanation")

    selected_row = st.selectbox(
        "Select a traffic record",
        options=list(range(len(result))),
        format_func=lambda index: f"Traffic Record {index + 1}",
        key="shap_selected_row"
    )

    selected_data = result.iloc[selected_row]

    threat_status = str(
        selected_data.get("Threat Status", "Unknown")
    )

    attack_type = str(
        selected_data.get("Attack Type", "Unknown")
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Threat Status",
            threat_status
        )

    with col2:
        st.metric(
            "Predicted Attack Type",
            attack_type
        )

    with col3:
        probability = selected_data.get(
            "Attack Probability",
            None
        )

        if probability is not None:

            try:
                probability = float(probability)

                if 0 <= probability <= 1:
                    probability *= 100

                st.metric(
                    "Attack Probability",
                    f"{probability:.2f}%"
                )

            except (ValueError, TypeError):
                st.metric("Attack Probability", "N/A")

        else:
            st.metric("Attack Probability", "N/A")

    # -------------------------------------------------
    # SECTION 2: LOCAL SHAP EXPLANATION
    # -------------------------------------------------

    st.divider()

    st.subheader("2. Local SHAP Explanation")

    st.write(
        "This explanation shows how individual network features "
        "influenced the selected traffic record's prediction."
    )

    row_data = feature_data.iloc[[selected_row]]

    try:

        with st.spinner("Calculating individual SHAP values..."):

            explanation = calculate_local_explanation(
                binary_model,
                row_data,
                features
            )

        if explanation.empty:

            st.warning("No SHAP values were returned.")

        else:

            # Plain-language directional explanation.
            create_directional_summary(
                explanation,
                threat_status
            )

            # SHAP feature contribution chart.
            st.markdown("### Feature Contribution Chart")

            render_local_shap_chart(
                explanation,
                max_features=10
            )

            # Detailed numerical explanation.
            st.markdown("### Detailed SHAP Values")

            detailed_table = explanation.copy()

            detailed_table = detailed_table.rename(
                columns={
                    "Value": "Feature Value"
                }
            )

            detailed_table["Direction"] = detailed_table[
                "SHAP value"
            ].apply(
                lambda value:
                    "Towards ATTACK"
                    if value > 0
                    else (
                        "Away from ATTACK"
                        if value < 0
                        else "Neutral"
                    )
            )

            display_columns = [
                column for column in [
                    "Feature",
                    "Feature Value",
                    "SHAP value",
                    "Absolute impact",
                    "Direction"
                ]
                if column in detailed_table.columns
            ]

            with st.expander("View detailed feature contributions"):

                st.dataframe(
                    detailed_table[display_columns],
                    use_container_width=True,
                    hide_index=True
                )

    except Exception as error:

        st.error(
            f"Individual SHAP calculation failed: {error}"
        )

    # -------------------------------------------------
    # SECTION 3: GLOBAL SHAP IMPORTANCE
    # -------------------------------------------------

    st.divider()

    st.subheader("3. Global Feature Importance")

    st.write(
        "Global SHAP identifies the network features that have "
        "the greatest overall influence across the analyzed traffic."
    )

    max_rows = min(200, len(feature_data))

    if max_rows == 0:
        st.info("No traffic records are available for global analysis.")
        return

    sample_size = st.slider(
        "Number of records for global SHAP analysis",
        min_value=1,
        max_value=max_rows,
        value=min(50, max_rows),
        step=1,
        key="global_shap_sample_size"
    )

    if st.button(
        "Calculate Global SHAP Importance",
        type="primary"
    ):

        try:

            sample = feature_data.sample(
                n=sample_size,
                random_state=42
            )

            with st.spinner(
                "Calculating global SHAP importance..."
            ):

                importance = calculate_global_explanation(
                    binary_model,
                    sample,
                    features
                )

            st.session_state["global_shap"] = importance

            st.success(
                "Global SHAP feature importance calculated successfully."
            )

        except Exception as error:

            st.error(
                f"Global SHAP calculation failed: {error}"
            )

    # Display global results if already calculated.
    importance = st.session_state.get("global_shap")

    if importance is not None:

        st.markdown("### Top 15 Important Network Features")

        render_global_shap_chart(
            importance,
            max_features=15
        )

        with st.expander("View global feature importance values"):

            st.dataframe(
                importance,
                use_container_width=True,
                hide_index=True
            )

    # -------------------------------------------------
    # EXPLANATION NOTE
    # -------------------------------------------------

    st.divider()

    st.caption(
        "ThreatTrace uses SHAP to explain its binary XGBoost model. "
        "SHAP values show how features influence the model output. "
        "They do not establish causation or independently confirm "
        "that network traffic is malicious."
    )
