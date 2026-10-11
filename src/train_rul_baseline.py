from pathlib import Path

import pandas as pd

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

TARGET = "RUL"


# ============================================================
# Load split
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
# Evaluation
# ============================================================

def evaluate_predictions(
    y_true,
    y_pred,
    split_name,
):

    mae = mean_absolute_error(
        y_true,
        y_pred,
    )

    rmse = mean_squared_error(
        y_true,
        y_pred,
    ) ** 0.5

    r2 = r2_score(
        y_true,
        y_pred,
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
    print("RUL REGRESSION BASELINE")
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
    # Target
    # --------------------------------------------------------

    y_train = train_df[TARGET]
    y_validation = validation_df[TARGET]
    y_test = test_df[TARGET]

    # --------------------------------------------------------
    # Mean baseline
    # --------------------------------------------------------

    mean_rul = y_train.mean()

    print(
        f"\nTraining mean RUL: {mean_rul:.4f}"
    )

    print(
        "\nPredicting the training mean for every observation..."
    )

    validation_predictions = pd.Series(
        mean_rul,
        index=y_validation.index,
    )

    test_predictions = pd.Series(
        mean_rul,
        index=y_test.index,
    )

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------

    validation_metrics = evaluate_predictions(
        y_validation,
        validation_predictions,
        "VALIDATION",
    )

    test_metrics = evaluate_predictions(
        y_test,
        test_predictions,
        "TEST",
    )

    # --------------------------------------------------------
    # Save metrics
    # --------------------------------------------------------

    metrics_path = (
        MODEL_DIR
        / "rul_baseline_metrics.csv"
    )

    pd.DataFrame([
        validation_metrics,
        test_metrics,
    ]).to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Save baseline value
    # --------------------------------------------------------

    baseline_path = (
        MODEL_DIR
        / "rul_baseline_value.txt"
    )

    baseline_path.write_text(
        f"{mean_rul:.10f}\n",
        encoding="utf-8",
    )

    # --------------------------------------------------------
    # Final
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "RUL BASELINE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"""
Baseline RUL:
{mean_rul:.4f}

Metrics saved:
{metrics_path}

Baseline value saved:
{baseline_path}
"""
    )