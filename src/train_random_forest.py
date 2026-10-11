from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

RANDOM_SEED = 42

TARGET = "failure_risk"

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

def evaluate_classifier(
    model,
    X,
    y,
    split_name,
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

    print("\n" + "=" * 70)
    print(f"{split_name} EVALUATION")
    print("=" * 70)

    print(f"\nROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"Accuracy : {accuracy:.4f}")

    print("\nConfusion Matrix")
    print("----------------")
    print(cm)

    print("\nClassification Report")
    print("---------------------")

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
    print("RANDOM FOREST FAILURE-RISK CLASSIFIER")
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
        "\nTraining Random Forest..."
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=16,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
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

    validation_metrics = evaluate_classifier(
        model,
        X_validation,
        y_validation,
        "VALIDATION",
    )

    test_metrics = evaluate_classifier(
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
        / "random_forest_feature_importance.csv"
    )

    importance.to_csv(
        importance_path,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP RANDOM FOREST FEATURES"
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
        / "random_forest_failure_risk.joblib"
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
        / "random_forest_metrics.csv"
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
        "RANDOM FOREST COMPLETE"
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