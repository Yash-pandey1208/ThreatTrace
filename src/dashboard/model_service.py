
"""Local inference helpers for the ThreatTrace Streamlit dashboard."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MULTICLASS_DIR = PROJECT_ROOT / "models" / "xgboost"
BINARY_DIR = PROJECT_ROOT / "models" / "XGBoost_Binary"


class ModelLoadError(RuntimeError):
    """Raised when a trained model or its metadata cannot be loaded."""


def _read_json(path: Path) -> Any:
    if not path.exists():
        raise ModelLoadError(f"Required model file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_model_bundle(model_dir: Path) -> dict[str, Any]:
    """Load an XGBoost JSON model and training feature order."""

    model_path = model_dir / (
        "xgboost_multiclass.json"
        if model_dir.name.lower() == "xgboost"
        else "xgboost_binary.json"
    )

    feature_columns = _read_json(
        model_dir / "feature_columns.json"
    )

    metadata = _read_json(
        model_dir / "metadata.json"
    )

    if not isinstance(feature_columns, list) or not feature_columns:
        raise ModelLoadError("Invalid feature list.")

    if not model_path.exists():
        raise ModelLoadError(f"Trained model not found: {model_path}")

    booster = xgb.Booster()

    try:
        booster.load_model(str(model_path))
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load {model_path.name}: {exc}"
        ) from exc

    return {
        "booster": booster,
        "features": feature_columns,
        "metadata": metadata,
        "model_path": model_path,
    }


def load_models() -> dict[str, Any]:
    """Load both trained models."""

    return {
        "binary": load_model_bundle(BINARY_DIR),
        "multiclass": load_model_bundle(MULTICLASS_DIR),
    }


def prepare_input(
    data: pd.DataFrame,
    feature_columns: list[str]
) -> tuple[pd.DataFrame, list[str]]:
    """Validate input and arrange features in training order."""

    if not isinstance(data, pd.DataFrame) or data.empty:
        raise ValueError("The uploaded dataset is empty.")

    missing = [
        column for column in feature_columns
        if column not in data.columns
    ]

    if missing:
        preview = ", ".join(missing[:8])
        suffix = " ..." if len(missing) > 8 else ""

        raise ValueError(
            f"CSV is missing {len(missing)} required model "
            f"feature(s): {preview}{suffix}. "
            "Use a flow CSV with the same 65 feature columns "
            "used during training."
        )

    matrix = data.loc[:, feature_columns].copy()

    for column in feature_columns:
        matrix[column] = pd.to_numeric(
            matrix[column],
            errors="coerce"
        )

    matrix = matrix.replace(
        [np.inf, -np.inf],
        np.nan
    )

    bad_counts = matrix.isna().sum()
    bad_columns = bad_counts[bad_counts > 0]

    warnings = []

    if not bad_columns.empty:
        warnings.append(
            f"{int(bad_columns.sum())} missing/non-numeric/"
            "infinite feature value(s) were replaced with 0. "
            "Review the source CSV if this is unexpected."
        )
        matrix = matrix.fillna(0.0)

    return matrix.astype(float), warnings


def predict_bundle(
    bundle: dict[str, Any],
    matrix: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Return predicted class IDs and scores."""

    dmatrix = xgb.DMatrix(
        matrix,
        feature_names=bundle["features"]
    )

    raw = np.asarray(
        bundle["booster"].predict(dmatrix)
    )

    if raw.ndim == 1:

        positive = np.clip(
            raw.astype(float),
            0.0,
            1.0
        )

        scores = np.column_stack([
            1.0 - positive,
            positive
        ])

        threshold = float(
            bundle["metadata"].get("threshold", 0.5)
        )

        predicted = (
            positive >= threshold
        ).astype(int)

    elif raw.ndim == 2:

        scores = raw.astype(float)
        predicted = np.argmax(scores, axis=1)

    else:
        raise ValueError(
            f"Unexpected prediction output shape: {raw.shape}"
        )

    return predicted, scores


def decode_multiclass_labels(
    class_ids: np.ndarray,
    project_root: Path = PROJECT_ROOT
) -> list[str]:
    """Decode class IDs using the saved LabelEncoder."""

    encoder_path = (
        project_root
        / "models"
        / "xgboost"
        / "label_encoder.joblib"
    )

    if encoder_path.exists():
        try:
            import joblib

            encoder = joblib.load(encoder_path)

            return [
                str(value)
                for value in encoder.inverse_transform(
                    class_ids.astype(int)
                )
            ]

        except Exception:
            pass

    return [
        str(int(value))
        for value in class_ids
    ]


def analyze_dataframe(
    data: pd.DataFrame,
    models: dict[str, Any]
) -> tuple[pd.DataFrame, list[str]]:
    """Run binary and multiclass predictions."""

    features = models["binary"]["features"]

    matrix, warnings = prepare_input(
        data,
        features
    )

    binary_ids, binary_scores = predict_bundle(
        models["binary"],
        matrix
    )

    multi_ids, multi_scores = predict_bundle(
        models["multiclass"],
        matrix
    )

    multi_labels = decode_multiclass_labels(multi_ids)

    result = data.copy()

    result["Threat Status"] = np.where(
        binary_ids == 1,
        "ATTACK",
        "BENIGN"
    )

    result["Attack Type"] = [
        label if binary_ids[i] == 1 else "BENIGN"
        for i, label in enumerate(multi_labels)
    ]

    result["Attack Probability"] = binary_scores[:, 1]

    result["Classification Confidence"] = np.max(
        multi_scores,
        axis=1
    )

    result["Predicted Class ID"] = multi_ids

    return result, warnings
