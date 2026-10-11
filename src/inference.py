from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

try:
    from src.feature_engineering import (
        SENSOR_COLUMNS,
        build_features,
        get_engineered_feature_columns,
        validate_feature_input,
        validate_features,
    )
except ModuleNotFoundError:
    from feature_engineering import (
        SENSOR_COLUMNS,
        build_features,
        get_engineered_feature_columns,
        validate_feature_input,
        validate_features,
    )


ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
DATA_DIR = ROOT / "data" / "processed"

FAILURE_MODEL_PATH = MODEL_DIR / "failure_risk_xgboost.joblib"
FAILURE_FEATURES_PATH = MODEL_DIR / "failure_risk_features.json"

RUL_MODEL_PATH = MODEL_DIR / "rul_xgboost.joblib"
RUL_FEATURES_PATH = MODEL_DIR / "rul_features.json"

MAX_RUL = 125


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(
            f"Model metadata not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


def load_models():
    if not FAILURE_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Failure-risk model not found: {FAILURE_MODEL_PATH}"
        )

    if not RUL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RUL model not found: {RUL_MODEL_PATH}"
        )

    failure_model = joblib.load(FAILURE_MODEL_PATH)
    rul_model = joblib.load(RUL_MODEL_PATH)

    failure_metadata = load_json(FAILURE_FEATURES_PATH)
    rul_metadata = load_json(RUL_FEATURES_PATH)

    failure_features = failure_metadata["features"]
    rul_features = rul_metadata["features"]

    if len(failure_features) != 143:
        raise ValueError(
            "Failure-risk model metadata does not contain "
            f"143 features. Found {len(failure_features)}."
        )

    if len(rul_features) != 143:
        raise ValueError(
            "RUL model metadata does not contain "
            f"143 features. Found {len(rul_features)}."
        )

    if failure_features != rul_features:
        raise ValueError(
            "Failure-risk and RUL models use different feature schemas."
        )

    forbidden = {
        "cycle",
        "cycle_progress",
        "final_cycle",
        "max_cycle",
        "unit_max_cycle",
    }

    forbidden_found = sorted(
        forbidden.intersection(failure_features)
    )

    if forbidden_found:
        raise ValueError(
            "Forbidden/leakage feature(s) found in model metadata: "
            f"{forbidden_found}"
        )

    return (
        failure_model,
        rul_model,
        failure_features,
    )


def validate_raw_telemetry(df: pd.DataFrame) -> None:
    validate_feature_input(df)

    required = {
        "unit",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
    }

    missing = sorted(required - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required telemetry columns: {missing}"
        )

    if df.empty:
        raise ValueError("Telemetry dataframe is empty.")

    if df["unit"].isna().any():
        raise ValueError("Telemetry contains missing unit values.")

    if df["cycle"].isna().any():
        raise ValueError("Telemetry contains missing cycle values.")

    if not np.isfinite(df["cycle"].to_numpy(dtype=float)).all():
        raise ValueError(
            "Telemetry contains non-finite cycle values."
        )


def build_inference_features(
    telemetry: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build exactly the same causal feature representation
    used during model training.

    Important:
    - No final-cycle information is used.
    - No cycle_progress feature is created.
    - cycle is retained only for ordering/replay metadata.
    """
    validate_raw_telemetry(telemetry)

    selected_columns = [
        "unit",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
        *SENSOR_COLUMNS,
    ]

    df = telemetry[selected_columns].copy()

    df = df.sort_values(
        ["unit", "cycle"]
    ).reset_index(drop=True)

    engineered = build_features(df)
    return engineered

def validate_model_schema(
    features_df: pd.DataFrame,
    expected_features: list[str],
) -> None:
    actual_features = [
        column
        for column in features_df.columns
        if column not in {
            "unit",
            "cycle",
            "RUL",
            "failure_risk",
        }
    ]

    if actual_features != expected_features:
        actual_set = set(actual_features)
        expected_set = set(expected_features)

        missing = sorted(expected_set - actual_set)
        extra = sorted(actual_set - expected_set)

        raise ValueError(
            "Inference feature schema does not match the "
            "training feature schema.\n"
            f"Missing features: {missing}\n"
            f"Extra features: {extra}"
        )

    if len(actual_features) != 143:
        raise ValueError(
            f"Expected 143 inference features, "
            f"found {len(actual_features)}"
        )


def predict_from_telemetry(
    telemetry: pd.DataFrame,
) -> dict:
    """
    Predict failure probability and RUL using telemetry
    available up to the latest supplied cycle.

    The caller should provide telemetry only up to the
    cycle being analyzed during replay.
    """
    (
        failure_model,
        rul_model,
        feature_columns,
    ) = load_models()

    features_df = build_inference_features(
        telemetry
    )

    validate_model_schema(
        features_df,
        feature_columns,
    )

    latest_row = features_df.iloc[[-1]]

    X_latest = latest_row[feature_columns]

    failure_probability = float(
        failure_model.predict_proba(X_latest)[0, 1]
    )

    predicted_rul = float(
        rul_model.predict(X_latest)[0]
    )

    predicted_rul = float(
        np.clip(
            predicted_rul,
            0,
            MAX_RUL,
        )
    )

    current_cycle = int(
        latest_row["cycle"].iloc[0]
    )

    unit = int(
        latest_row["unit"].iloc[0]
    )

    return {
        "unit": unit,
        "current_cycle": current_cycle,
        "failure_probability": failure_probability,
        "predicted_rul": predicted_rul,
        "feature_count": len(feature_columns),
        "leakage_free": True,
    }


def compare_replay_consistency(
    full_telemetry: pd.DataFrame,
    unit: int,
    cycle: int,
    tolerance: float = 1e-9,
) -> dict:
    """
    Verify that the features for cycle t are identical whether
    they are calculated from:

        1. the complete telemetry history, or
        2. telemetry truncated at cycle t.

    This is a key deployment/replay consistency check.
    """
    full_unit = (
        full_telemetry[
            full_telemetry["unit"] == unit
        ]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    if full_unit.empty:
        raise ValueError(
            f"Unit {unit} not found in telemetry."
        )

    truncated = full_unit[
        full_unit["cycle"] <= cycle
    ].copy()

    if truncated.empty:
        raise ValueError(
            f"No telemetry exists for unit {unit} "
            f"at or before cycle {cycle}."
        )

    full_features = build_inference_features(
        full_unit
    )

    truncated_features = build_inference_features(
        truncated
    )

    full_at_cycle = full_features[
        full_features["cycle"] == cycle
    ]

    truncated_at_cycle = truncated_features[
        truncated_features["cycle"] == cycle
    ]

    if full_at_cycle.empty:
        raise ValueError(
            f"Cycle {cycle} not found for unit {unit}."
        )

    if truncated_at_cycle.empty:
        raise ValueError(
            f"Truncated telemetry does not contain "
            f"cycle {cycle}."
        )

    expected_features = [
        column
        for column in full_features.columns
        if column not in {
            "unit",
            "cycle",
            "RUL",
            "failure_risk",
        }
    ]

    full_values = (
        full_at_cycle[expected_features]
        .to_numpy(dtype=float)
    )

    truncated_values = (
        truncated_at_cycle[expected_features]
        .to_numpy(dtype=float)
    )

    max_difference = float(
        np.max(
            np.abs(
                full_values - truncated_values
            )
        )
    )

    passed = max_difference <= tolerance

    return {
        "unit": unit,
        "cycle": cycle,
        "passed": passed,
        "max_absolute_difference": max_difference,
        "tolerance": tolerance,
    }


def main() -> None:
    print("=" * 70)
    print("LEAKAGE-FREE INFERENCE CHECK")
    print("=" * 70)

    telemetry_path = (
        DATA_DIR / "fd001_train_labeled.parquet"
    )

    if not telemetry_path.exists():
        raise FileNotFoundError(
            f"Telemetry file not found: {telemetry_path}"
        )

    df = pd.read_parquet(
        telemetry_path
    )

    first_unit = int(
        df["unit"].iloc[0]
    )

    unit_df = (
        df[df["unit"] == first_unit]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    if len(unit_df) < 30:
        raise ValueError(
            "Not enough telemetry rows for inference check."
        )

    test_cycle = int(
        unit_df["cycle"].iloc[
            min(29, len(unit_df) - 1)
        ]
    )

    result = compare_replay_consistency(
        df,
        unit=first_unit,
        cycle=test_cycle,
    )

    print()
    print(
        f"Truncated-feature consistency: "
        f"{'PASS' if result['passed'] else 'FAIL'}"
    )
    print(
        f"Unit: {result['unit']}"
    )
    print(
        f"Cycle: {result['cycle']}"
    )
    print(
        f"Maximum absolute feature difference: "
        f"{result['max_absolute_difference']:.12g}"
    )

    if not result["passed"]:
        raise RuntimeError(
            "Inference consistency check failed."
        )

    replay_df = unit_df[
        unit_df["cycle"] <= test_cycle
    ].copy()

    prediction = predict_from_telemetry(
        replay_df
    )

    print()
    print("Prediction:")
    print(
        f"  Unit: {prediction['unit']}"
    )
    print(
        f"  Current cycle: "
        f"{prediction['current_cycle']}"
    )
    print(
        f"  Failure probability: "
        f"{prediction['failure_probability']:.4f}"
    )
    print(
        f"  Predicted RUL: "
        f"{prediction['predicted_rul']:.2f}"
    )
    print(
        f"  Feature count: "
        f"{prediction['feature_count']}"
    )
    print(
        f"  Leakage-free: "
        f"{prediction['leakage_free']}"
    )


if __name__ == "__main__":
    main()