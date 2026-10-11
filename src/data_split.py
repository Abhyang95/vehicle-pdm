from pathlib import Path

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT / "data" / "processed"
)

INPUT_FILE = (
    PROCESSED_DIR
    / "fd001_train_features.parquet"
)

RANDOM_STATE = 42

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15


# ============================================================
# Unit-level splitting
# ============================================================

def split_by_unit(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset by complete unit trajectories.

    No unit can appear in more than one split.
    """

    if not abs(
        TRAIN_SIZE
        + VALIDATION_SIZE
        + TEST_SIZE
        - 1.0
    ) < 1e-9:
        raise ValueError(
            "Split proportions must sum to 1."
        )

    groups = df["unit"]

    # --------------------------------------------------------
    # First split: train vs temporary
    # --------------------------------------------------------

    train_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=TRAIN_SIZE,
        random_state=RANDOM_STATE,
    )

    train_idx, temp_idx = next(
        train_splitter.split(
            df,
            groups=groups,
        )
    )

    train_df = (
        df.iloc[train_idx]
        .copy()
    )

    temp_df = (
        df.iloc[temp_idx]
        .copy()
    )

    # --------------------------------------------------------
    # Second split: validation vs test
    #
    # Temporary set = 30%
    #
    # Validation = 15%
    # Test       = 15%
    #
    # Therefore:
    # validation fraction of temp = 0.5
    # --------------------------------------------------------

    validation_splitter = GroupShuffleSplit(
        n_splits=1,
        train_size=0.5,
        random_state=RANDOM_STATE,
    )

    validation_idx, test_idx = next(
        validation_splitter.split(
            temp_df,
            groups=temp_df["unit"],
        )
    )

    validation_df = (
        temp_df.iloc[validation_idx]
        .copy()
    )

    test_df = (
        temp_df.iloc[test_idx]
        .copy()
    )

    return (
        train_df,
        validation_df,
        test_df,
    )


# ============================================================
# Leakage validation
# ============================================================

def validate_no_unit_overlap(
    train_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:

    train_units = set(
        train_df["unit"].unique()
    )

    validation_units = set(
        validation_df["unit"].unique()
    )

    test_units = set(
        test_df["unit"].unique()
    )

    if train_units & validation_units:
        raise ValueError(
            "Unit overlap between train and validation."
        )

    if train_units & test_units:
        raise ValueError(
            "Unit overlap between train and test."
        )

    if validation_units & test_units:
        raise ValueError(
            "Unit overlap between validation and test."
        )


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM"
    )
    print("FD001 UNIT-LEVEL DATA SPLIT")
    print("=" * 70)

    print(
        f"\nLoading:\n{INPUT_FILE}"
    )

    df = pd.read_parquet(
        INPUT_FILE
    )

    print(
        f"\nLoaded {len(df):,} rows "
        f"across {df['unit'].nunique()} units."
    )

    train_df, validation_df, test_df = (
        split_by_unit(df)
    )

    validate_no_unit_overlap(
        train_df,
        validation_df,
        test_df,
    )

    # --------------------------------------------------------
    # Save datasets
    # --------------------------------------------------------

    train_path = (
        PROCESSED_DIR
        / "train.parquet"
    )

    validation_path = (
        PROCESSED_DIR
        / "validation.parquet"
    )

    test_path = (
        PROCESSED_DIR
        / "test.parquet"
    )

    train_df.to_parquet(
        train_path,
        index=False,
    )

    validation_df.to_parquet(
        validation_path,
        index=False,
    )

    test_df.to_parquet(
        test_path,
        index=False,
    )

    # --------------------------------------------------------
    # Report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SPLIT SUMMARY")
    print("=" * 70)

    print("\nTrain")
    print("-" * 70)
    print(f"Rows  : {len(train_df):,}")
    print(
        f"Units : {train_df['unit'].nunique()}"
    )

    print("\nValidation")
    print("-" * 70)
    print(f"Rows  : {len(validation_df):,}")
    print(
        f"Units : {validation_df['unit'].nunique()}"
    )

    print("\nTest")
    print("-" * 70)
    print(f"Rows  : {len(test_df):,}")
    print(
        f"Units : {test_df['unit'].nunique()}"
    )

    print("\nUnit overlap check")
    print("-" * 70)
    print("Train ∩ Validation :", len(
        set(train_df["unit"])
        & set(validation_df["unit"])
    ))

    print("Train ∩ Test       :", len(
        set(train_df["unit"])
        & set(test_df["unit"])
    ))

    print("Validation ∩ Test  :", len(
        set(validation_df["unit"])
        & set(test_df["unit"])
    ))

    print("\nSaved datasets:")
    print(train_path)
    print(validation_path)
    print(test_path)

    print("\n" + "=" * 70)
    print("UNIT-LEVEL SPLIT COMPLETE")
    print("=" * 70)