from pathlib import Path

import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

TRAIN_FILE = (
    PROCESSED_DIR / "train_scaled.parquet"
)

VALIDATION_FILE = (
    PROCESSED_DIR / "validation_scaled.parquet"
)

TEST_FILE = (
    PROCESSED_DIR / "test_scaled.parquet"
)

MODEL_FILE = (
    PROCESSED_DIR
    / "logistic_regression_classifier.joblib"
)


# ============================================================
# Columns
# ============================================================

IDENTIFIER_COLUMNS = [
    "unit",
    "cycle",
]

TARGET_COLUMN = "failure_risk"

NON_FEATURE_COLUMNS = (
    IDENTIFIER_COLUMNS
    + [
        "RUL",
        TARGET_COLUMN,
    ]
)


# ============================================================
# Load data
# ============================================================

def load_data():

    train_df = pd.read_parquet(
        TRAIN_FILE
    )

    validation_df = pd.read_parquet(
        VALIDATION_FILE
    )

    test_df = pd.read_parquet(
        TEST_FILE
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# Prepare X / y
# ============================================================

def prepare_features(
    df: pd.DataFrame,
):

    feature_columns = [
        column
        for column in df.columns
        if column not in NON_FEATURE_COLUMNS
    ]

    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    return X, y, feature_columns


# ============================================================
# Evaluation
# ============================================================

def evaluate_classifier(
    model,
    X,
    y,
    dataset_name: str,
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

    cm = confusion_matrix(
        y,
        predictions,
    )

    print("\n" + "=" * 70)
    print(
        f"{dataset_name.upper()} CLASSIFIER RESULTS"
    )
    print("=" * 70)

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

    print("\nConfusion Matrix")
    print("-" * 70)

    print(cm)

    print("\nClassification Report")
    print("-" * 70)

    print(
        classification_report(
            y,
            predictions,
            digits=4,
            zero_division=0,
        )
    )

    return {
        "dataset": dataset_name,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM"
    )
    print("LOGISTIC REGRESSION FAILURE-RISK BASELINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    (
        train_df,
        validation_df,
        test_df,
    ) = load_data()

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_train, y_train, feature_columns = (
        prepare_features(train_df)
    )

    X_validation, y_validation, _ = (
        prepare_features(validation_df)
    )

    X_test, y_test, _ = (
        prepare_features(test_df)
    )

    print(
        f"\nTraining observations : {len(X_train):,}"
    )

    print(
        f"Validation observations: "
        f"{len(X_validation):,}"
    )

    print(
        f"Test observations      : "
        f"{len(X_test):,}"
    )

    print(
        f"Model features         : "
        f"{len(feature_columns)}"
    )

    print(
        f"\nTraining positive rate : "
        f"{y_train.mean() * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    print(
        "\nTraining Logistic Regression..."
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
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

    train_metrics = evaluate_classifier(
        model,
        X_train,
        y_train,
        "Train",
    )

    validation_metrics = evaluate_classifier(
        model,
        X_validation,
        y_validation,
        "Validation",
    )

    test_metrics = evaluate_classifier(
        model,
        X_test,
        y_test,
        "Test",
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
        },
        MODEL_FILE,
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(
        f"\n{MODEL_FILE}"
    )

    print("\n" + "=" * 70)
    print("CLASSIFIER TRAINING COMPLETE")
    print("=" * 70)