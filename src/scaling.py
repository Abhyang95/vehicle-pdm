from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

TRAIN_FILE = (
    PROCESSED_DIR / "train.parquet"
)

VALIDATION_FILE = (
    PROCESSED_DIR / "validation.parquet"
)

TEST_FILE = (
    PROCESSED_DIR / "test.parquet"
)

SCALER_FILE = (
    PROCESSED_DIR / "feature_scaler.joblib"
)


# ============================================================
# Columns that must NOT be scaled
# ============================================================

IDENTIFIER_COLUMNS = [
    "unit",
    "cycle",
]

TARGET_COLUMNS = [
    "RUL",
    "failure_risk",
]


# ============================================================
# Load datasets
# ============================================================

def load_splits():
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
# Determine model features
# ============================================================

def get_feature_columns(
    df: pd.DataFrame,
) -> list[str]:

    excluded = set(
        IDENTIFIER_COLUMNS
        + TARGET_COLUMNS
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in excluded
    ]

    return feature_columns


# ============================================================
# Fit scaler on training data only
# ============================================================

def fit_scaler(
    train_df: pd.DataFrame,
    feature_columns: list[str],
) -> StandardScaler:

    scaler = StandardScaler()

    scaler.fit(
        train_df[feature_columns]
    )

    return scaler


# ============================================================
# Transform dataset
# ============================================================

def transform_dataset(
    df: pd.DataFrame,
    feature_columns: list[str],
    scaler: StandardScaler,
) -> pd.DataFrame:

    result = df.copy()

    result[feature_columns] = scaler.transform(
        result[feature_columns]
    )

    return result


# ============================================================
# Validation
# ============================================================

def validate_scaled_data(
    df: pd.DataFrame,
    feature_columns: list[str],
) -> None:

    if df[feature_columns].isna().any().any():
        raise ValueError(
            "Scaled dataset contains NaN values."
        )

    if df[feature_columns].isna().any().any():
        raise ValueError(
            "Scaled dataset contains missing values."
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM"
    )
    print("FEATURE SCALING")
    print("=" * 70)

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    print("\nLoading datasets...")

    (
        train_df,
        validation_df,
        test_df,
    ) = load_splits()

    print(
        f"Train      : {len(train_df):,} rows"
    )

    print(
        f"Validation : {len(validation_df):,} rows"
    )

    print(
        f"Test       : {len(test_df):,} rows"
    )

    # --------------------------------------------------------
    # Features
    # --------------------------------------------------------

    feature_columns = get_feature_columns(
        train_df
    )

    print(
        f"\nNumber of model features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Fit scaler ONLY on train
    # --------------------------------------------------------

    print(
        "\nFitting StandardScaler "
        "on training data only..."
    )

    scaler = fit_scaler(
        train_df,
        feature_columns,
    )

    # --------------------------------------------------------
    # Transform
    # --------------------------------------------------------

    train_scaled = transform_dataset(
        train_df,
        feature_columns,
        scaler,
    )

    validation_scaled = transform_dataset(
        validation_df,
        feature_columns,
        scaler,
    )

    test_scaled = transform_dataset(
        test_df,
        feature_columns,
        scaler,
    )

    # --------------------------------------------------------
    # Validate
    # --------------------------------------------------------

    validate_scaled_data(
        train_scaled,
        feature_columns,
    )

    validate_scaled_data(
        validation_scaled,
        feature_columns,
    )

    validate_scaled_data(
        test_scaled,
        feature_columns,
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    train_output = (
        PROCESSED_DIR
        / "train_scaled.parquet"
    )

    validation_output = (
        PROCESSED_DIR
        / "validation_scaled.parquet"
    )

    test_output = (
        PROCESSED_DIR
        / "test_scaled.parquet"
    )

    train_scaled.to_parquet(
        train_output,
        index=False,
    )

    validation_scaled.to_parquet(
        validation_output,
        index=False,
    )

    test_scaled.to_parquet(
        test_output,
        index=False,
    )

    joblib.dump(
        scaler,
        SCALER_FILE,
    )

    # --------------------------------------------------------
    # Scaling sanity check
    # --------------------------------------------------------

    train_means = (
        train_scaled[feature_columns]
        .mean()
        .abs()
        .mean()
    )

    train_stds = (
        train_scaled[feature_columns]
        .std()
        .mean()
    )

    print("\n" + "=" * 70)
    print("SCALING SUMMARY")
    print("=" * 70)

    print(
        f"\nFeatures scaled : "
        f"{len(feature_columns)}"
    )

    print(
        f"Mean absolute training feature mean : "
        f"{train_means:.6f}"
    )

    print(
        f"Average training feature std        : "
        f"{train_stds:.6f}"
    )

    print(
        "\nScaler fitted on: TRAIN ONLY"
    )

    print(
        "\nSaved:"
    )

    print(train_output)
    print(validation_output)
    print(test_output)
    print(SCALER_FILE)

    print("\n" + "=" * 70)
    print("FEATURE SCALING COMPLETE")
    print("=" * 70)