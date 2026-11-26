
"""Main Streamlit entry point for ThreatTrace-129."""

import sys
from pathlib import Path

import streamlit as st


# =====================================================
# 1. PROJECT PATH
# =====================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# =====================================================
# 2. PROJECT IMPORTS
# =====================================================

from src.dashboard.model_service import load_models

from src.dashboard.views.home import render_home
from src.dashboard.views.results import render_results
from src.dashboard.views.explanation import render_explanation


# =====================================================
# 3. STREAMLIT CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="ThreatTrace | Network Threat Intelligence",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =====================================================
# 4. DASHBOARD STYLING
# =====================================================

st.markdown(
    """
    <style>
        .stApp {
            background-color: #0b1220;
        }

        [data-testid="stSidebar"] {
            background-color: #111b2e;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2rem;
        }

        div[data-testid="stMetric"] {
            background-color: #142238;
            border: 1px solid #263b57;
            padding: 14px;
            border-radius: 12px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# =====================================================
# 5. SESSION STATE
# =====================================================

if "analysis_result" not in st.session_state:
    st.session_state["analysis_result"] = None

if "analysis_warnings" not in st.session_state:
    st.session_state["analysis_warnings"] = []

if "global_shap" not in st.session_state:
    st.session_state["global_shap"] = None


# =====================================================
# 6. LOAD TRAINED MODELS
# =====================================================

@st.cache_resource(show_spinner="Loading trained XGBoost models...")
def get_models():
    return load_models()


try:
    models = get_models()

except Exception as error:

    st.error(
        f"Unable to load trained models: {error}"
    )

    st.info(
        """
        Check that these model files exist:

        - models/XGBoost_Binary/xgboost_binary.json
        - models/XGBoost_Binary/feature_columns.json
        - models/XGBoost_Binary/metadata.json
        - models/xgboost/xgboost_multiclass.json
        - models/xgboost/feature_columns.json
        - models/xgboost/metadata.json
        """
    )

    st.stop()


# =====================================================
# 7. DASHBOARD HEADER
# =====================================================

st.title("🛡️ ThreatTrace-129")

st.caption(
    "Autonomous Network Intrusion Forensics "
    "& Real-Time Threat Classification"
)


# =====================================================
# 8. SIDEBAR NAVIGATION
# =====================================================

with st.sidebar:

    st.title("ThreatTrace")

    st.caption("Network Threat Intelligence System")

    st.divider()

    page = st.radio(
        "Dashboard Navigation",
        [
            "Analyze Traffic",
            "Overview",
            "Explainability",
            "Model Information"
        ],
        index=0,
        label_visibility="collapsed",
        key="dashboard_navigation"
    )

    st.divider()

    st.success("Binary Model Loaded")

    st.success("Multiclass Model Loaded")

    st.divider()

    st.caption("ThreatTrace-129 | Academic Project")


# =====================================================
# 9. GET ANALYSIS RESULTS
# =====================================================

result = st.session_state.get("analysis_result")


# =====================================================
# 10. PAGE 1: ANALYZE TRAFFIC
# =====================================================

if page == "Analyze Traffic":

    render_results(
        models=models
    )


# =====================================================
# 11. PAGE 2: OVERVIEW
# =====================================================

elif page == "Overview":

    render_home(
        models=models,
        result=result
    )


# =====================================================
# 12. PAGE 3: EXPLAINABILITY
# =====================================================

elif page == "Explainability":

    render_explanation(
        models=models,
        result=result
    )


# =====================================================
# 13. PAGE 4: MODEL INFORMATION
# =====================================================

elif page == "Model Information":

    st.title("Trained Model Information")

    # -----------------------------------------------
    # Binary model
    # -----------------------------------------------

    binary_metadata = models["binary"]["metadata"]

    multiclass_metadata = models["multiclass"]["metadata"]

    st.subheader("Binary XGBoost Model")

    st.write(
        "Purpose: Classify network traffic as BENIGN or ATTACK."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test Accuracy",
        f"{binary_metadata.get('test_accuracy', 0) * 100:.2f}%"
    )

    col2.metric(
        "Test Precision",
        f"{binary_metadata.get('test_precision', 0) * 100:.2f}%"
    )

    col3.metric(
        "Test Recall",
        f"{binary_metadata.get('test_recall', 0) * 100:.2f}%"
    )

    with st.expander("View Binary Model Metadata"):

        st.json(binary_metadata)

    st.divider()

    # -----------------------------------------------
    # Multiclass model
    # -----------------------------------------------

    st.subheader("Multiclass XGBoost Model")

    st.write(
        "Purpose: Identify the predicted category of an attack."
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Test Accuracy",
        f"{multiclass_metadata.get('test_accuracy', 0) * 100:.2f}%"
    )

    col2.metric(
        "Number of Classes",
        multiclass_metadata.get("number_of_classes", "N/A")
    )

    col3.metric(
        "Input Features",
        len(models["multiclass"]["features"])
    )

    with st.expander("View Multiclass Model Metadata"):

        st.json(multiclass_metadata)

    st.divider()

    # -----------------------------------------------
    # Model features
    # -----------------------------------------------

    st.subheader("Model Input Features")

    st.write(
        f"Both models use "
        f"{len(models['binary']['features'])} "
        "engineered network traffic features."
    )

    with st.expander("Show all model input features"):

        st.code(
            "\n".join(models["binary"]["features"]),
            language="text"
        )

    st.warning(
        "The displayed accuracy and evaluation metrics are "
        "saved model evaluation results. They are not "
        "measurements from the currently uploaded CSV."
    )


# =====================================================
# 14. DASHBOARD FOOTER
# =====================================================

st.divider()

st.caption(
    "ThreatTrace-129 | Network Intrusion Detection "
    "| XGBoost | Explainable AI with SHAP"
)
