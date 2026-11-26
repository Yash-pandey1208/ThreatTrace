
import streamlit as st


def render_alert_banner(
    threat_status,
    attack_type=None,
    confidence=None
):
    """
    Display a security alert based on the predicted threat status.

    Parameters:
        threat_status: Predicted status (Benign or Malicious).
        attack_type: Predicted attack category.
        confidence: Model classification confidence.
    """

    status = str(threat_status).strip().lower()

    if status in ["benign", "normal", "0"]:
        st.success(
            "BENIGN TRAFFIC: No malicious activity detected."
        )

    else:
        st.error(
            "SECURITY ALERT: Suspicious network activity detected!"
        )

        if attack_type is not None:
            st.write(f"**Predicted Attack Type:** {attack_type}")

    if confidence is not None:
        try:
            confidence_value = float(confidence)

            # Display confidence as a percentage when supplied
            # as a probability between 0 and 1.
            if 0 <= confidence_value <= 1:
                confidence_value *= 100

            st.caption(
                f"Classification Confidence: {confidence_value:.2f}%"
            )

        except (ValueError, TypeError):
            st.caption(f"Classification Confidence: {confidence}")
