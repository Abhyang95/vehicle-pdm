from pathlib import Path

import joblib
import pandas as pd


# ============================================================
# VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM
# UNIFIED DIAGNOSTIC INFERENCE ENGINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = PROJECT_ROOT / "models"
SPLIT_DIR = PROJECT_ROOT / "data" / "processed" / "splits"
DIAGNOSTICS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "diagnostics"
)

DIAGNOSTICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Model loading
# ============================================================

def load_model_bundle(filename):
    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found: {path}"
        )

    bundle = joblib.load(path)

    if not isinstance(bundle, dict):
        raise ValueError(
            f"Expected model bundle dictionary: {path}"
        )

    required_keys = {
        "model",
        "feature_columns",
        "target",
    }

    missing = required_keys - set(bundle.keys())

    if missing:
        raise ValueError(
            f"Missing keys in {path}: {missing}"
        )

    return bundle


# ============================================================
# Severity engine
# ============================================================

def determine_severity(
    failure_probability,
    predicted_rul,
):
    """
    Determine diagnostic severity using both
    failure probability and predicted RUL.

    This is a project-level decision rule,
    not a manufacturer-defined standard.
    """

    if (
        failure_probability >= 0.90
        or predicted_rul <= 7
    ):
        return "CRITICAL"

    if (
        failure_probability >= 0.65
        or predicted_rul <= 15
    ):
        return "HIGH"

    if (
        failure_probability >= 0.35
        or predicted_rul <= 30
    ):
        return "MEDIUM"

    return "LOW"


# ============================================================
# Risk status
# ============================================================

def determine_risk_status(
    failure_probability,
):
    if failure_probability >= 0.65:
        return "HIGH_RISK"

    if failure_probability >= 0.35:
        return "WATCH"

    return "LOW_RISK"


# ============================================================
# Recommendation engine
# ============================================================

def generate_recommendation(
    severity,
    predicted_rul,
):
    """
    Generate a high-level maintenance recommendation.

    These are decision-support recommendations,
    not direct mechanical instructions.
    """

    if severity == "CRITICAL":
        return {
            "priority": "IMMEDIATE",
            "action": (
                "Schedule immediate diagnostic inspection "
                "and assess whether the vehicle should "
                "remain in service."
            ),
            "inspection": (
                "Inspect abnormal telemetry trends, "
                "temperature, pressure, and speed/flow "
                "signals."
            ),
        }

    if severity == "HIGH":
        return {
            "priority": "URGENT",
            "action": (
                "Schedule maintenance inspection at "
                "the earliest practical opportunity."
            ),
            "inspection": (
                "Review degraded sensor groups and "
                "recent telemetry trends."
            ),
        }

    if severity == "MEDIUM":
        return {
            "priority": "PLANNED",
            "action": (
                "Plan preventive maintenance and "
                "continue monitoring."
            ),
            "inspection": (
                "Monitor sensor trends and investigate "
                "persistent abnormalities."
            ),
        }

    return {
        "priority": "ROUTINE",
        "action": (
            "Continue normal monitoring and "
            "scheduled maintenance."
        ),
        "inspection": (
            "No immediate diagnostic intervention "
            "indicated."
        ),
    }


# ============================================================
# Single-observation prediction
# ============================================================

def predict_observation(
    observation,
    failure_bundle,
    rul_bundle,
):
    failure_model = failure_bundle["model"]
    failure_features = failure_bundle["feature_columns"]

    rul_model = rul_bundle["model"]
    rul_features = rul_bundle["feature_columns"]

    # --------------------------------------------------------
    # Verify both models expect the same feature set
    # --------------------------------------------------------

    if failure_features != rul_features:
        raise ValueError(
            "Failure-risk and RUL models use different "
            "feature columns."
        )

    feature_columns = failure_features

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in observation.index
    ]

    if missing_features:
        raise ValueError(
            "Observation is missing model features: "
            f"{missing_features}"
        )

    X = observation[
        feature_columns
    ].to_frame().T

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    failure_probability = float(
        failure_model
        .predict_proba(X)[0, 1]
    )

    predicted_rul = float(
        rul_model
        .predict(X)[0]
    )

    # RUL is capped at 125 cycles in this project.
    predicted_rul = max(
        0.0,
        min(125.0, predicted_rul),
    )

    risk_status = determine_risk_status(
        failure_probability
    )

    severity = determine_severity(
        failure_probability,
        predicted_rul,
    )

    recommendation = generate_recommendation(
        severity,
        predicted_rul,
    )

    return {
        "unit": int(observation["unit"]),
        "cycle": int(observation["cycle"]),
        "failure_probability": failure_probability,
        "risk_status": risk_status,
        "predicted_rul": predicted_rul,
        "severity": severity,
        "priority": recommendation["priority"],
        "maintenance_action": recommendation["action"],
        "suggested_inspection": recommendation["inspection"],
    }


# ============================================================
# Main demonstration
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM"
    )
    print("UNIFIED DIAGNOSTIC INFERENCE ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    failure_bundle = load_model_bundle(
        "xgboost_failure_risk.joblib"
    )

    rul_bundle = load_model_bundle(
        "rul_xgboost.joblib"
    )

    print(
        "\nFailure-risk model loaded."
    )

    print(
        f"Failure features: "
        f"{len(failure_bundle['feature_columns'])}"
    )

    print(
        "RUL model loaded."
    )

    print(
        f"RUL features: "
        f"{len(rul_bundle['feature_columns'])}"
    )

    # --------------------------------------------------------
    # Load held-out test data
    # --------------------------------------------------------

    test_path = (
        SPLIT_DIR
        / "test.parquet"
    )

    test_df = pd.read_parquet(
        test_path
    )

    print(
        f"\nTest observations: "
        f"{len(test_df):,}"
    )

    # --------------------------------------------------------
    # Find highest-risk observation
    # --------------------------------------------------------

    failure_model = failure_bundle["model"]
    feature_columns = failure_bundle[
        "feature_columns"
    ]

    X_test = test_df[
        feature_columns
    ]

    probabilities = (
        failure_model
        .predict_proba(X_test)[:, 1]
    )

    highest_risk_index = int(
        probabilities.argmax()
    )

    observation = test_df.iloc[
        highest_risk_index
    ]

    # --------------------------------------------------------
    # Generate diagnosis
    # --------------------------------------------------------

    diagnosis = predict_observation(
        observation,
        failure_bundle,
        rul_bundle,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print("DIAGNOSTIC RESULT")
    print("=" * 70)

    print(
        f"\nVehicle / Unit ID     : "
        f"{diagnosis['unit']}"
    )

    print(
        f"Current cycle        : "
        f"{diagnosis['cycle']}"
    )

    print(
        f"Failure probability  : "
        f"{diagnosis['failure_probability']:.4f}"
    )

    print(
        f"Risk status          : "
        f"{diagnosis['risk_status']}"
    )

    print(
        f"Predicted RUL        : "
        f"{diagnosis['predicted_rul']:.2f} cycles"
    )

    print(
        f"Severity             : "
        f"{diagnosis['severity']}"
    )

    print(
        f"Maintenance priority : "
        f"{diagnosis['priority']}"
    )

    print(
        f"\nMaintenance action:"
    )

    print(
        diagnosis["maintenance_action"]
    )

    print(
        f"\nSuggested inspection:"
    )

    print(
        diagnosis["suggested_inspection"]
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    output = pd.DataFrame([
        diagnosis
    ])

    output_path = (
        DIAGNOSTICS_DIR
        / "sample_diagnostic_result.csv"
    )

    output.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "DIAGNOSTIC INFERENCE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSaved result:"
    )

    print(output_path)