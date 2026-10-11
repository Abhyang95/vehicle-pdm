from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
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
# Features
# ============================================================

def get_feature_columns(df):

    return [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]


# ============================================================
# Evaluation
# ============================================================

def evaluate_regressor(
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
    print(f"{split_name} EVALUATION")
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
    print("RANDOM FOREST RUL REGRESSOR")
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

    feature_columns = get_feature_columns(
        train_df
    )

    print(
        f"\nModel features: {len(feature_columns)}"
    )

    print(
        f"Training observations  : {len(train_df):,}"
    )

    print(
        f"Validation observations: {len(validation_df):,}"
    )

    print(
        f"Test observations       : {len(test_df):,}"
    )

    # --------------------------------------------------------
    # Prepare X / y
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
        "\nTraining Random Forest RUL regressor..."
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=18,
        min_samples_leaf=2,
        max_features="sqrt",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    validation_metrics = evaluate_regressor(
        model,
        X_validation,
        y_validation,
        "VALIDATION",
    )

    test_metrics = evaluate_regressor(
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
        / "rul_random_forest_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP RUL RANDOM FOREST FEATURES"
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
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "rul_random_forest.joblib"
    )

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "target": TARGET,
        },
        model_path,
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = (
        MODEL_DIR
        / "rul_random_forest_metrics.csv"
    )

    pd.DataFrame([
        validation_metrics,
        test_metrics,
    ]).to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RUL RANDOM FOREST COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"""
Model saved:
{model_path}

Metrics saved:
{metrics_path}

Feature importance saved:
{importance_path}
"""
    )