from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    average_precision_score,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = PROCESSED_DIR / "train_scaled.parquet"
VALIDATION_FILE = PROCESSED_DIR / "validation_scaled.parquet"
TEST_FILE = PROCESSED_DIR / "test_scaled.parquet"

MODEL_FILE = PROCESSED_DIR / "xgboost_classifier.joblib"
THRESHOLD_FILE = PROCESSED_DIR / "xgboost_failure_threshold.joblib"

IDENTIFIER_COLUMNS = ["unit", "cycle"]
TARGET_COLUMN = "failure_risk"
NON_FEATURE_COLUMNS = IDENTIFIER_COLUMNS + ["RUL", TARGET_COLUMN]

MIN_RECALL = 0.95


def load_data():
    validation_df = pd.read_parquet(VALIDATION_FILE)
    test_df = pd.read_parquet(TEST_FILE)

    return validation_df, test_df


def prepare_features(df, feature_columns):
    X = df[feature_columns]
    y = df[TARGET_COLUMN]

    return X, y


def evaluate_threshold(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    precision = precision_score(
        y_true,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(y_true, predictions)

    tn, fp, fn, tp = cm.ravel()

    false_positive_rate = fp / (fp + tn)

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_positive_rate": false_positive_rate,
        "true_negatives": tn,
        "false_positives": fp,
        "false_negatives": fn,
        "true_positives": tp,
    }


def print_threshold_table(results):
    print("\n" + "=" * 90)
    print("VALIDATION THRESHOLD ANALYSIS")
    print("=" * 90)

    print(
        f"\n{'Threshold':<12}"
        f"{'Precision':<12}"
        f"{'Recall':<12}"
        f"{'F1':<12}"
        f"{'FP Rate':<12}"
    )

    print("-" * 90)

    for result in results:
        print(
            f"{result['threshold']:<12.2f}"
            f"{result['precision']:<12.4f}"
            f"{result['recall']:<12.4f}"
            f"{result['f1']:<12.4f}"
            f"{result['false_positive_rate']:<12.4f}"
        )


def evaluate_final_test(model, test_df, feature_columns, threshold):
    X_test, y_test = prepare_features(
        test_df,
        feature_columns,
    )

    probabilities = model.predict_proba(X_test)[:, 1]

    predictions = (
        probabilities >= threshold
    ).astype(int)

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    pr_auc = average_precision_score(
        y_test,
        probabilities,
    )

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        y_test,
        predictions,
    )

    print("\n" + "=" * 90)
    print("FINAL TEST EVALUATION")
    print("=" * 90)

    print(f"\nLocked threshold: {threshold:.2f}")

    print(f"\nROC-AUC  : {roc_auc:.4f}")
    print(f"PR-AUC   : {pr_auc:.4f}")
    print(f"F1       : {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")

    print("\nConfusion Matrix")
    print("-" * 90)
    print(cm)

    tn, fp, fn, tp = cm.ravel()

    print("\nPrediction Summary")
    print("-" * 90)
    print(f"True negatives : {tn:,}")
    print(f"False positives: {fp:,}")
    print(f"False negatives: {fn:,}")
    print(f"True positives : {tp:,}")

    return {
        "threshold": threshold,
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "f1": f1,
        "precision": precision,
        "recall": recall,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    print("=" * 90)
    print("VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM")
    print("XGBOOST FAILURE-RISK THRESHOLD OPTIMIZATION")
    print("=" * 90)

    validation_df, test_df = load_data()

    checkpoint = joblib.load(MODEL_FILE)

    model = checkpoint["model"]
    feature_columns = checkpoint["feature_columns"]

    X_validation, y_validation = prepare_features(
        validation_df,
        feature_columns,
    )

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    print(f"\nValidation observations: {len(X_validation):,}")
    print(f"Model features         : {len(feature_columns)}")
    print(f"Required minimum recall: {MIN_RECALL:.2%}")

    print("\nSearching thresholds...")

    thresholds = np.arange(
        0.10,
        0.91,
        0.01,
    )

    all_results = []

    for threshold in thresholds:
        result = evaluate_threshold(
            y_validation,
            validation_probabilities,
            threshold,
        )

        all_results.append(result)

    eligible_results = [
        result
        for result in all_results
        if result["recall"] >= MIN_RECALL
    ]

    if not eligible_results:
        raise RuntimeError(
            "No threshold satisfies the minimum recall requirement."
        )

    best_result = max(
        eligible_results,
        key=lambda result: (
            result["f1"],
            result["precision"],
        ),
    )

    print_threshold_table(all_results)

    print("\n" + "=" * 90)
    print("SELECTED VALIDATION THRESHOLD")
    print("=" * 90)

    print(f"\nThreshold          : {best_result['threshold']:.2f}")
    print(f"Validation F1      : {best_result['f1']:.4f}")
    print(f"Validation Recall  : {best_result['recall']:.4f}")
    print(f"Validation Precision: {best_result['precision']:.4f}")
    print(
        f"Validation FP Rate : "
        f"{best_result['false_positive_rate']:.4f}"
    )

    threshold_artifact = {
        "threshold": best_result["threshold"],
        "minimum_recall": MIN_RECALL,
        "selection_rule": (
            "Maximum validation F1 subject to "
            "minimum recall constraint"
        ),
    }

    joblib.dump(
        threshold_artifact,
        THRESHOLD_FILE,
    )

    print("\nThreshold saved:")
    print(THRESHOLD_FILE)

    final_test_metrics = evaluate_final_test(
        model,
        test_df,
        feature_columns,
        best_result["threshold"],
    )

    print("\n" + "=" * 90)
    print("THRESHOLD OPTIMIZATION COMPLETE")
    print("=" * 90)