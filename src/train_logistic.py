from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
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


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42

TARGET = "failure_risk"

NON_FEATURE_COLUMNS = [
    "unit",
    "RUL",
    "failure_risk",
]


# ============================================================
# Load datasets
# ============================================================

def load_split(
    filename: str,
) -> pd.DataFrame:

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

def get_feature_columns(
    df: pd.DataFrame,
):

    return [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]


# ============================================================
# Evaluation
# ============================================================

def evaluate_classifier(
    model,
    X,
    y,
    split_name: str,
):

    probabilities = model.predict_proba(X)[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    roc_auc = roc_auc_score(
        y,
        probabilities,
    )

    pr_auc = average_precision_score(
        y,
        probabilities,
    )

    f1 = f1_score(
        y,
        predictions,
        zero_division=0,
    )

    precision = precision_score(
        y,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y,
        predictions,
        zero_division=0,
    )

    accuracy = accuracy_score(
        y,
        predictions,
    )

    cm = confusion_matrix(
        y,
        predictions,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        f"{split_name} EVALUATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nROC-AUC  : {roc_auc:.4f}"
    )

    print(
        f"PR-AUC   : {pr_auc:.4f}"
    )

    print(
        f"F1       : {f1:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"Accuracy : {accuracy:.4f}"
    )

    print(
        "\nConfusion Matrix"
    )

    print(
        "----------------"
    )

    print(cm)

    print(
        "\nClassification Report"
    )

    print(
        "---------------------"
    )

    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return {
        "split": split_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "accuracy": accuracy,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM")
    print("LOGISTIC REGRESSION FAILURE-RISK BASELINE")
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

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = get_feature_columns(
        train_df
    )

    print(
        f"\nModel features: {len(feature_columns)}"
    )

    print(
        f"Training rows : {len(train_df):,}"
    )

    print(
        f"Validation rows: {len(validation_df):,}"
    )

    print(
        f"Test rows     : {len(test_df):,}"
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
    # Class weighting
    # --------------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        solver="lbfgs",
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "Training complete."
    )

    # --------------------------------------------------------
    # Evaluate validation
    # --------------------------------------------------------

    validation_metrics = evaluate_classifier(
        model,
        X_validation,
        y_validation,
        "VALIDATION",
    )

    # --------------------------------------------------------
    # Evaluate test
    # --------------------------------------------------------

    test_metrics = evaluate_classifier(
        model,
        X_test,
        y_test,
        "TEST",
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    model_path = (
        MODEL_DIR
        / "logistic_failure_risk.joblib"
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
        / "logistic_metrics.csv"
    )

    metrics_df = pd.DataFrame([
        validation_metrics,
        test_metrics,
    ])

    metrics_df.to_csv(
        metrics_path,
        index=False,
    )

    # --------------------------------------------------------
    # Coefficients
    # --------------------------------------------------------

    coefficients = pd.DataFrame({
        "feature": feature_columns,
        "coefficient": model.coef_[0],
    })

    coefficients["absolute_coefficient"] = (
        coefficients["coefficient"]
        .abs()
    )

    coefficients = coefficients.sort_values(
        "absolute_coefficient",
        ascending=False,
    )

    coefficient_path = (
        MODEL_DIR
        / "logistic_feature_coefficients.csv"
    )

    coefficients.to_csv(
        coefficient_path,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP LOGISTIC REGRESSION FEATURES"
    )

    print(
        "=" * 70
    )

    print(
        coefficients.head(15).to_string(
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
        "LOGISTIC REGRESSION COMPLETE"
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

Coefficients saved:
{coefficient_path}
"""
    )