from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[1]
SPLIT_DIR = ROOT / "data" / "processed" / "splits"
MODEL_DIR = ROOT / "models"

TRAIN_PATH = SPLIT_DIR / "train.parquet"
VAL_PATH = SPLIT_DIR / "validation.parquet"
TEST_PATH = SPLIT_DIR / "test.parquet"

MODEL_PATH = MODEL_DIR / "rul_xgboost.joblib"
METRICS_PATH = MODEL_DIR / "rul_metrics.csv"
FEATURES_PATH = MODEL_DIR / "rul_features.json"

TARGET = "RUL"
MAX_RUL = 125

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

    return features


def evaluate_split(
    model: XGBRegressor,
    df: pd.DataFrame,
    feature_columns: list[str],
    split_name: str,
) -> dict:
    X = df[feature_columns]
    y = df[TARGET].astype(float)

    predictions = model.predict(X)
    predictions = np.clip(
        predictions,
        0,
        MAX_RUL,
    )

    mae = mean_absolute_error(y, predictions)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)

    metrics = {
        "split": split_name,
        "rows": len(df),
        "units": df["unit"].nunique(),
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
    }

    print()
    print("=" * 70)
    print(f"{split_name.upper()} RUL RESULTS")
    print("=" * 70)
    print(f"rows: {len(df):,}")
    print(f"units: {df['unit'].nunique()}")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")

    return metrics


def main() -> None:
    print("=" * 70)
    print("LEAKAGE-SAFE XGBOOST RUL TRAINING")
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
    y_train = train_df[TARGET].astype(float)

    print()
    print(f"Training rows: {len(train_df):,}")
    print(f"Validation rows: {len(val_df):,}")
    print(f"Test rows: {len(test_df):,}")
    print(f"Training units: {len(train_units)}")
    print(f"Validation units: {len(val_units)}")
    print(f"Test units: {len(test_units)}")
    print(f"Model features: {len(feature_columns)}")

    model = XGBRegressor(
    n_estimators=700,
    max_depth=4,
    learning_rate=0.03,
    min_child_weight=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=5.0,
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    random_state=42,
    n_jobs=-1,
)

    print()
    print("Training XGBoost RUL regressor...")

    model.fit(
        X_train,
        y_train,
        eval_set=[
            (val_df[feature_columns], val_df[TARGET].astype(float))
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
                "model_type": "XGBRegressor",
                "feature_count": len(feature_columns),
                "features": feature_columns,
                "target": TARGET,
                "max_rul": MAX_RUL,
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
        MODEL_DIR / "rul_feature_importance.csv",
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
        f"{MODEL_DIR / 'rul_feature_importance.csv'}"
    )


if __name__ == "__main__":
    main()