from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd

from src.explainability import explain_prediction
from src.inference import (
    FAILURE_FEATURES_PATH,
    FAILURE_MODEL_PATH,
    build_inference_features,
    predict_from_telemetry,
)
from src.subsystem_health import (
    add_live_health_statuses,
    calculate_live_group_health,
)


# ============================================================
# Diagnostic configuration
# ============================================================

RISK_WATCH_THRESHOLD = 0.15
RISK_HIGH_THRESHOLD = 0.30

RUL_MEDIUM_THRESHOLD = 30
RUL_HIGH_THRESHOLD = 15
RUL_CRITICAL_THRESHOLD = 7

SHAP_TOP_K = 10

EXPECTED_FEATURE_COUNT = 143


# ============================================================
# Risk status
# ============================================================

def determine_risk_status(
    failure_probability: float,
) -> str:
    """
    Convert failure probability into an operational risk state.

    The 0.30 threshold is the selected operating threshold from
    the official 100-engine FD001 evaluation:

        Precision: 0.8846
        Recall:    0.9200
        F1:         0.9020

    The project prioritizes detecting potential failures because
    missed failures are more costly than additional warnings.
    """

    if failure_probability >= RISK_HIGH_THRESHOLD:
        return "HIGH_RISK"

    if failure_probability >= RISK_WATCH_THRESHOLD:
        return "WATCH"

    return "LOW_RISK"


# ============================================================
# Diagnostic severity
# ============================================================

def determine_severity(
    failure_probability: float,
    predicted_rul: float,
) -> str:

    if (
        failure_probability >= 0.90
        or predicted_rul <= RUL_CRITICAL_THRESHOLD
    ):
        return "CRITICAL"

    if (
        failure_probability >= 0.65
        or predicted_rul <= RUL_HIGH_THRESHOLD
    ):
        return "HIGH"

    if (
        failure_probability >= 0.35
        or predicted_rul <= RUL_MEDIUM_THRESHOLD
    ):
        return "MEDIUM"

    return "LOW"


# ============================================================
# Load current model feature schema
# ============================================================

def load_failure_feature_columns() -> list[str]:
    """
    Load the feature schema saved alongside the current
    leakage-free XGBoost failure-risk model.

    The current artifacts contain the model directly, rather
    than an older {'model': ..., 'feature_columns': ...} bundle.
    """

    if not FAILURE_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Failure-risk feature metadata not found: "
            f"{FAILURE_FEATURES_PATH}"
        )

    with open(
        FAILURE_FEATURES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        metadata = json.load(file)

    feature_columns = metadata.get("features")

    if not isinstance(feature_columns, list):
        raise ValueError(
            "Failure-risk feature metadata must contain "
            "a 'features' list."
        )

    feature_columns = [
        str(column)
        for column in feature_columns
    ]

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Failure-risk model metadata does not contain "
            f"{EXPECTED_FEATURE_COUNT} features. "
            f"Found {len(feature_columns)}."
        )

    forbidden_features = {
        "cycle",
        "cycle_progress",
        "final_cycle",
        "max_cycle",
        "unit_max_cycle",
    }

    forbidden_found = sorted(
        set(feature_columns) & forbidden_features
    )

    if forbidden_found:
        raise ValueError(
            "Leakage-related features found in failure-risk "
            "model metadata: "
            + ", ".join(forbidden_found)
        )

    return feature_columns


# ============================================================
# Build complete diagnostic report
# ============================================================

def build_diagnostic_report(
    telemetry_df: pd.DataFrame,
    top_k_shap: int = SHAP_TOP_K,
    include_shap: bool = True,
) -> dict:

    if telemetry_df.empty:
        raise ValueError(
            "Telemetry input cannot be empty."
        )

    telemetry_df = (
        telemetry_df
        .sort_values(
            ["unit", "cycle"]
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # 1. ML prediction
    # --------------------------------------------------------

    prediction = predict_from_telemetry(
        telemetry_df
    )

    failure_probability = float(
        prediction["failure_probability"]
    )

    predicted_rul = float(
        prediction["predicted_rul"]
    )

    risk_status = determine_risk_status(
        failure_probability
    )

    severity = determine_severity(
        failure_probability,
        predicted_rul,
    )

    # --------------------------------------------------------
    # 2. Build inference features
    # --------------------------------------------------------

    inference_features = (
        build_inference_features(
            telemetry_df
        )
    )

    # --------------------------------------------------------
    # 3. Load current failure-risk model
    # --------------------------------------------------------

    if not FAILURE_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Failure-risk model not found: "
            f"{FAILURE_MODEL_PATH}"
        )

    # Current artifacts store the XGBClassifier directly.
    # They are NOT the older {'model': ..., ...} bundle format.
    failure_model = joblib.load(
        FAILURE_MODEL_PATH
    )

    feature_columns = (
        load_failure_feature_columns()
    )

    # --------------------------------------------------------
    # 4. Validate feature alignment
    # --------------------------------------------------------

    missing_features = [
        column
        for column in feature_columns
        if column not in inference_features.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing engineered model features: "
            + ", ".join(missing_features)
        )

    X = inference_features[
        feature_columns
    ].copy()

    if X.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            "Diagnostic SHAP input contains "
            f"{X.shape[1]} features. "
            f"Expected {EXPECTED_FEATURE_COUNT}."
        )


    # --------------------------------------------------------
    # 5. SHAP explanation
    # --------------------------------------------------------

    shap_contributors = []

    if include_shap:
        shap_df = explain_prediction(
            failure_model,
            X,
            top_k=top_k_shap,
        )

        for _, row in shap_df.iterrows():
            shap_contributors.append(
                {
                    "feature": str(row["feature"]),
                    "impact": round(float(row["shap_value"]), 4),
                    "absolute_impact": round(
                        float(row["absolute_shap"]), 4
                    ),
                    "direction": str(row["direction"]),
                    "description": str(row["feature"]),
                }
            )
    # --------------------------------------------------------
    # 6. Live subsystem health
    # --------------------------------------------------------

    health_df = (
        calculate_live_group_health(
            telemetry_df
        )
    )

    health_df = (
        add_live_health_statuses(
            health_df
        )
    )

    latest_health = health_df.iloc[-1]

    subsystem_health = {}

    for group in [
        "temperature",
        "pressure",
        "speed_flow",
        "thermal_bleed",
    ]:

        subsystem_health[group] = {
            "health": round(
                float(
                    latest_health[
                        f"{group}_health"
                    ]
                ),
                2,
            ),
            "status": str(
                latest_health[
                    f"{group}_status"
                ]
            ),
        }

    overall_health = round(
        float(
            latest_health[
                "overall_sensor_health"
            ]
        ),
        2,
    )

    overall_health_status = str(
        latest_health[
            "overall_health_status"
        ]
    )

    # --------------------------------------------------------
    # 7. Latest telemetry
    # --------------------------------------------------------

    latest = telemetry_df.iloc[-1]

    # --------------------------------------------------------
    # 8. Final diagnostic report
    # --------------------------------------------------------

    return {
        "unit": int(
            latest["unit"]
        ),
        "current_cycle": int(
            latest["cycle"]
        ),
        "failure_probability": round(
            failure_probability,
            4,
        ),
        "predicted_rul": round(
            predicted_rul,
            2,
        ),
        "risk_status": risk_status,
        "severity": severity,
        "overall_sensor_health": (
            overall_health
        ),
        "overall_health_status": (
            overall_health_status
        ),
        "subsystem_health": (
            subsystem_health
        ),
        "shap_contributors": (
            shap_contributors
        ),
        "feature_count": len(
            feature_columns
        ),
    }
