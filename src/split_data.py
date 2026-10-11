from pathlib import Path

import joblib
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from feature_engineering import build_features, validate_features


ROOT = Path(__file__).resolve().parents[1]

LABELED_PATH = ROOT / "data" / "processed" / "fd001_train_labeled.parquet"
FEATURE_PATH = ROOT / "data" / "processed" / "fd001_train_features.parquet"
SPLIT_DIR = ROOT / "data" / "processed" / "splits"

RANDOM_STATE = 42

TRAIN_UNITS = 70
VALIDATION_UNITS = 15
TEST_UNITS = 15

EXPECTED_FEATURE_COUNT = 143


def build_training_features():
    """Build leakage-free engineered features without feature scaling."""

    if not LABELED_PATH.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found: {LABELED_PATH}"
        )

    df = pd.read_parquet(LABELED_PATH)

    print("Building leakage-free engineered features...")
    features = build_features(df)

    validate_features(features)

    feature_columns = [
        column
        for column in features.columns
        if column not in {
            "unit",
            "cycle",
            "RUL",
            "failure_risk",
        }
    ]

    forbidden = {
        "cycle_progress",
        "final_cycle",
        "max_cycle",
        "unit_max_cycle",
    }

    leakage_columns = forbidden.intersection(feature_columns)

    if leakage_columns:
        raise ValueError(
            f"Forbidden leakage columns detected: {sorted(leakage_columns)}"
        )

    if len(feature_columns) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} model features, "
            f"found {len(feature_columns)}."
        )

    FEATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURE_PATH, index=False)

    return features, feature_columns


def split_by_unit(df):
    """Create train/validation/test splits with complete unit separation."""

    units = df["unit"].unique()

    if len(units) != 100:
        raise ValueError(
            f"Expected 100 FD001 training units, found {len(units)}."
        )

    splitter_1 = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_UNITS,
        random_state=RANDOM_STATE,
    )

    train_val_idx, test_idx = next(
        splitter_1.split(
            df,
            groups=df["unit"],
        )
    )

    train_val = df.iloc[train_val_idx].copy()
    test = df.iloc[test_idx].copy()

    splitter_2 = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_UNITS,
        random_state=RANDOM_STATE,
    )

    train_idx, validation_idx = next(
        splitter_2.split(
            train_val,
            groups=train_val["unit"],
        )
    )

    train = train_val.iloc[train_idx].copy()
    validation = train_val.iloc[validation_idx].copy()

    return train, validation, test


def validate_unit_separation(train, validation, test):
    """Verify that no engine appears in more than one split."""

    train_units = set(train["unit"].unique())
    validation_units = set(validation["unit"].unique())
    test_units = set(test["unit"].unique())

    if train_units & validation_units:
        raise ValueError("Unit leakage between train and validation.")

    if train_units & test_units:
        raise ValueError("Unit leakage between train and test.")

    if validation_units & test_units:
        raise ValueError("Unit leakage between validation and test.")

    if len(train_units) != TRAIN_UNITS:
        raise ValueError(
            f"Expected {TRAIN_UNITS} training units, "
            f"found {len(train_units)}."
        )

    if len(validation_units) != VALIDATION_UNITS:
        raise ValueError(
            f"Expected {VALIDATION_UNITS} validation units, "
            f"found {len(validation_units)}."
        )

    if len(test_units) != TEST_UNITS:
        raise ValueError(
            f"Expected {TEST_UNITS} test units, "
            f"found {len(test_units)}."
        )


def save_splits(train, validation, test):
    """Save raw engineered features without scaling."""

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)

    train_path = SPLIT_DIR / "train.parquet"
    validation_path = SPLIT_DIR / "validation.parquet"
    test_path = SPLIT_DIR / "test.parquet"

    train.to_parquet(train_path, index=False)
    validation.to_parquet(validation_path, index=False)
    test.to_parquet(test_path, index=False)

    return train_path, validation_path, test_path


def main():
    print("=" * 70)
    print("LEAKAGE-SAFE FEATURE + UNIT SPLIT")
    print("=" * 70)

    features, feature_columns = build_training_features()

    train, validation, test = split_by_unit(features)

    validate_unit_separation(
        train,
        validation,
        test,
    )

    save_splits(
        train,
        validation,
        test,
    )

    print()
    print(f"Total rows: {len(features):,}")
    print(f"Total units: {features['unit'].nunique()}")

    print()
    print(
        f"Train: {len(train):,} rows / "
        f"{train['unit'].nunique()} units"
    )

    print(
        f"Validation: {len(validation):,} rows / "
        f"{validation['unit'].nunique()} units"
    )

    print(
        f"Test: {len(test):,} rows / "
        f"{test['unit'].nunique()} units"
    )

    print()
    print(f"Raw sensors: 14")
    print(f"Settings: 3")
    print(f"Rolling features: 84")
    print(f"Slope features: 42")
    print(f"Model feature count: {len(feature_columns)}")

    print()
    print("Scaling: NOT USED")
    print("Reason: XGBoost tree models do not require feature scaling.")
    print("Linear/logistic baselines will scale inside their own pipeline.")

    print()
    print("cycle included as model feature: NO")
    print("cycle_progress included: NO")
    print("final-cycle-derived feature included: NO")
    print("Unit leakage across splits: NO")

    print()
    print(f"Saved engineered dataset: {FEATURE_PATH}")
    print(f"Saved splits: {SPLIT_DIR}")

    print()
    print("Feature schema:")
    print(f"  {len(feature_columns)} model features")
    print(f"  First 10: {feature_columns[:10]}")
    print(f"  Last 10: {feature_columns[-10:]}")

    print()
    print("SCALING CONSISTENCY FIX: COMPLETE")


if __name__ == "__main__":
    main()