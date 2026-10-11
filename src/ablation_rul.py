from pathlib import Path

import pandas as pd

from xgboost import XGBRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 42

TARGET = "RUL"

# Lifecycle features we intentionally remove.
LIFECYCLE_FEATURES = [
    "cycle",
    "cycle_progress",
]

NON_FEATURE_COLUMNS = [
    "unit",
    "RUL",
    "failure_risk",
]


# ============================================================
# Load data
# ============================================================

def load_split(filename: str) -> pd.DataFrame:

    path = SPLIT_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found:\n{path}\n\n"
            "Run split_data.py first."
        )

    return pd.read_parquet(path)


# ============================================================
# Feature selection
# ============================================================

def get_sensor_only_features(df):

    return [
        column
        for column in df.columns
        if (
            column not in NON_FEATURE_COLUMNS
            and column not in LIFECYCLE_FEATURES
        )
    ]


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model,
    X,
    y,
    split_name,
):

    predictions = model.predict(X)

    # RUL cannot be negative.
    predictions = predictions.clip(
        min=0
    )

    mae = mean_absolute_error(
        y,
        predictions,
    )

    rmse = mean_squared_error(
        y,
        predictions,
    ) ** 0.5

    r2 = r2_score(
        y,
        predictions,
    )

    print("\n" + "=" * 70)
    print(f"{split_name} — SENSOR-ONLY RUL MODEL")
    print("=" * 70)

    print(f"\nMAE : {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²  : {r2:.4f}")

    return {
        "split": split_name,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM")
    print("RUL SENSOR-ONLY ABLATION EXPERIMENT")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    train_df = load_split(
        "train.parquet"
    )

    validation_df = load_split(
        "validation.parquet"
    )

    test_df = load_split(
        "test.parquet"
    )

    feature_columns = get_sensor_only_features(
        train_df
    )

    print(
        f"\nSensor-derived features: "
        f"{len(feature_columns)}"
    )

    print(
        "Removed lifecycle features:"
    )

    for feature in LIFECYCLE_FEATURES:
        print(f"  - {feature}")

    print(
        f"\nTraining observations  : {len(train_df):,}"
    )

    print(
        f"Validation observations: {len(validation_df):,}"
    )

    print(
        f"Test observations       : {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X_train = train_df[
        feature_columns
    ]

    y_train = train_df[
        TARGET
    ]

    X_validation = validation_df[
        feature_columns
    ]

    y_validation = validation_df[
        TARGET
    ]

    X_test = test_df[
        feature_columns
    ]

    y_test = test_df[
        TARGET
    ]

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nTraining sensor-only XGBoost..."
    )

    model = XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=RANDOM_SEED,
        n_jobs=-1,
        tree_method="hist",
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (
                X_validation,
                y_validation,
            )
        ],
        verbose=False,
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    validation_metrics = evaluate(
        model,
        X_validation,
        y_validation,
        "VALIDATION",
    )

    test_metrics = evaluate(
        model,
        X_test,
        y_test,
        "TEST",
    )

    # --------------------------------------------------------
    # Feature importance
    # --------------------------------------------------------

    importance = pd.DataFrame({
        "feature": feature_columns,
        "importance": model.feature_importances_,
    })

    importance = importance.sort_values(
        "importance",
        ascending=False,
    )

    importance_path = (
        MODEL_DIR
        / "rul_sensor_only_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP SENSOR-ONLY FEATURES"
    )

    print(
        "=" * 70
    )

    print(
        importance.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = (
        MODEL_DIR
        / "rul_sensor_only_metrics.csv"
    )

    pd.DataFrame([
        validation_metrics,
        test_metrics,
    ]).to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Compare against full model
    # --------------------------------------------------------

    full_metrics_path = (
        MODEL_DIR
        / "rul_xgboost_metrics.csv"
    )

    if full_metrics_path.exists():

        full_metrics = pd.read_csv(
            full_metrics_path
        )

        full_test = full_metrics[
            full_metrics["split"] == "TEST"
        ].iloc[0]

        print(
            "\n" + "=" * 70
        )

        print(
            "FULL MODEL vs SENSOR-ONLY"
        )

        print(
            "=" * 70
        )

        comparison = pd.DataFrame([
            {
                "model": "Full XGBoost",
                "MAE": full_test["mae"],
                "RMSE": full_test["rmse"],
                "R2": full_test["r2"],
            },
            {
                "model": "Sensor-only XGBoost",
                "MAE": test_metrics["mae"],
                "RMSE": test_metrics["rmse"],
                "R2": test_metrics["r2"],
            },
        ])

        print(
            comparison.to_string(
                index=False
            )
        )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RUL ABLATION COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"""
Metrics saved:
{metrics_path}

Feature importance saved:
{importance_path}
"""
    )