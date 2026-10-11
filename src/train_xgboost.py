from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from xgboost import XGBClassifier


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
MODEL_DIR = ROOT / "models"

TRAIN_PATH = SPLIT_DIR / "train.parquet"
VAL_PATH = SPLIT_DIR / "validation.parquet"
TEST_PATH = SPLIT_DIR / "test.parquet"

MODEL_PATH = MODEL_DIR / "failure_risk_xgboost.joblib"
METRICS_PATH = MODEL_DIR / "failure_risk_metrics.csv"
FEATURES_PATH = MODEL_DIR / "failure_risk_features.json"

TARGET = "failure_risk"

LEAKAGE_COLUMNS = {
    "cycle_progress",
    "final_cycle",
    "max_cycle",
    "unit_max_cycle",
}


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")

    df = pd.read_parquet(path)

    if TARGET not in df.columns:
        raise ValueError(f"{TARGET!r} not found in {path}")

    return df


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    excluded = {
        "unit",
        "cycle",
        "RUL",
        "failure_risk",
    }

    features = [
        column
        for column in df.columns
        if column not in excluded
    ]

    leakage_found = sorted(
        set(features).intersection(LEAKAGE_COLUMNS)
    )

    if leakage_found:
        raise ValueError(
            f"Leakage features detected: {leakage_found}"
        )

    if len(features) != 143:
        raise ValueError(
            f"Expected exactly 143 model features, "
            f"found {len(features)}"
        )

    if not features:
        raise ValueError("No model features found.")

    return features


def evaluate_split(
    model: XGBClassifier,
    df: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
) -> dict:
    X = df[feature_columns]
    y = df[TARGET].astype(int)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y,
        predictions,
        labels=[0, 1],
    ).ravel()

    metrics = {
        "split": split_name,
        "rows": len(df),
        "units": df["unit"].nunique(),
        "positive_rate": float(y.mean()),
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "pr_auc": float(average_precision_score(y, probabilities)),
        "precision": float(
            precision_score(y, predictions, zero_division=0)
        ),
        "recall": float(
            recall_score(y, predictions, zero_division=0)
        ),
        "f1": float(
            f1_score(y, predictions, zero_division=0)
        ),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }

    print()
    print("=" * 70)
    print(f"{split_name.upper()} RESULTS")
    print("=" * 70)

    for key, value in metrics.items():
        if key in {"split", "rows", "units"}:
            print(f"{key}: {value}")
        elif isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    if split_name == "test":
        print()
        print("Classification report:")
        print(
            classification_report(
                y,
                predictions,
                digits=4,
                zero_division=0,
            )
        )

    return metrics


def main() -> None:
    print("=" * 70)
    print("LEAKAGE-SAFE XGBOOST FAILURE-RISK TRAINING")
    print("=" * 70)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_split(TRAIN_PATH)
    val_df = load_split(VAL_PATH)
    test_df = load_split(TEST_PATH)

    feature_columns = get_feature_columns(train_df)

    for name, df in [
        ("train", train_df),
        ("validation", val_df),
        ("test", test_df),
    ]:
        missing = sorted(set(feature_columns) - set(df.columns))

        if missing:
            raise ValueError(
                f"{name} split is missing features: {missing}"
            )

        if len(feature_columns) != 143:
            raise ValueError(
                "Unexpected feature count before training."
            )

    train_units = set(train_df["unit"].unique())
    val_units = set(val_df["unit"].unique())
    test_units = set(test_df["unit"].unique())

    if train_units & val_units:
        raise ValueError("Unit leakage between train and validation.")

    if train_units & test_units:
        raise ValueError("Unit leakage between train and test.")

    if val_units & test_units:
        raise ValueError("Unit leakage between validation and test.")

    X_train = train_df[feature_columns]
    y_train = train_df[TARGET].astype(int)

    print()
    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Training units: {len(train_units)}")
    print(f"Validation units: {len(val_units)}")
    print(f"Test units: {len(test_units)}")
    print(f"Model features: {len(feature_columns)}")
    print(f"Positive samples: {int(y_train.sum()):,}")
    print(f"Negative samples: {int((y_train == 0).sum()):,}")

    positive = int(y_train.sum())
    negative = int((y_train == 0).sum())

    if positive == 0:
        raise ValueError("Training set contains no positive samples.")

    scale_pos_weight = negative / positive

    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    model = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=2,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.05,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )

    print()
    print("Training XGBoost classifier...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (val_df[feature_columns], val_df[TARGET].astype(int))
        ],
        verbose=False,
    )

    print("Training complete.")

    metrics = []

    metrics.append(
        evaluate_split(
            model,
            train_df,
            feature_columns,
            "train",
        )
    )

    metrics.append(
        evaluate_split(
            model,
            val_df,
            feature_columns,
            "validation",
        )
    )

    metrics.append(
        evaluate_split(
            model,
            test_df,
            feature_columns,
            "test",
        )
    )

    joblib.dump(model, MODEL_PATH)

    with open(FEATURES_PATH, "w", encoding="utf-8") as file:
        json.dump(
            {
                "model_type": "XGBClassifier",
                "feature_count": len(feature_columns),
                "features": feature_columns,
                "target": TARGET,
                "threshold": 0.5,
                "leakage_free": True,
                "random_state": 42,
            },
            file,
            indent=2,
        )

    pd.DataFrame(metrics).to_csv(
        METRICS_PATH,
        index=False,
    )

    importance = pd.DataFrame(
        {
            "feature": feature_columns,
            "importance": model.feature_importances_,
        }
    ).sort_values(
        "importance",
        ascending=False,
    )

    importance.to_csv(
        MODEL_DIR / "failure_risk_feature_importance.csv",
        index=False,
    )

    print()
    print("=" * 70)
    print("MODEL SAVED")
    print("=" * 70)
    print(f"Model: {MODEL_PATH}")
    print(f"Metrics: {METRICS_PATH}")
    print(f"Features: {FEATURES_PATH}")
    print(
        f"Feature importance: "
        f"{MODEL_DIR / 'failure_risk_feature_importance.csv'}"
    )


if __name__ == "__main__":
    main()