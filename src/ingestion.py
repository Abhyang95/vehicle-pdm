from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "cmapss"
)


# ============================================================
# C-MAPSS column definitions
# ============================================================

COLUMN_NAMES = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
] + [
    f"sensor_{i}"
    for i in range(1, 22)
]


# ============================================================
# Generic loader
# ============================================================

def _load_txt_file(filename: str) -> pd.DataFrame:
    """
    Load a whitespace-separated C-MAPSS file.

    Parameters
    ----------
    filename : str
        Name of the file inside data/raw/cmapss.

    Returns
    -------
    pd.DataFrame
        Parsed telemetry dataframe.
    """

    file_path = RAW_DATA_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_path}"
        )

    df = pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=COLUMN_NAMES,
    )

    return df


# ============================================================
# FD001 training data
# ============================================================

def load_fd001_train() -> pd.DataFrame:
    """
    Load FD001 training telemetry.

    Training trajectories run until simulated failure.
    """

    return _load_txt_file("train_FD001.txt")


# ============================================================
# FD001 test data
# ============================================================

def load_fd001_test() -> pd.DataFrame:
    """
    Load FD001 test telemetry.

    Test trajectories stop before simulated failure.
    """

    return _load_txt_file("test_FD001.txt")


# ============================================================
# FD001 ground-truth RUL
# ============================================================

def load_fd001_rul() -> pd.DataFrame:
    """
    Load the ground-truth RUL values for FD001 test units.
    """

    file_path = RAW_DATA_DIR / "RUL_FD001.txt"

    if not file_path.exists():
        raise FileNotFoundError(
            f"RUL file not found: {file_path}"
        )

    return pd.read_csv(
        file_path,
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )


# ============================================================
# Dataset validation
# ============================================================

def validate_fd001_dataset(
    train: pd.DataFrame,
    test: pd.DataFrame,
    rul: pd.DataFrame,
) -> None:
    """
    Validate basic structural expectations for FD001.
    """

    expected_columns = 26

    if train.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns, "
            f"got {train.shape[1]}"
        )

    if test.shape[1] != expected_columns:
        raise ValueError(
            f"Expected {expected_columns} columns, "
            f"got {test.shape[1]}"
        )

    if train["unit"].nunique() != 100:
        raise ValueError(
            "FD001 training data should contain 100 units."
        )

    if test["unit"].nunique() != 100:
        raise ValueError(
            "FD001 test data should contain 100 units."
        )

    if len(rul) != 100:
        raise ValueError(
            "FD001 RUL file should contain 100 values."
        )

    if train.isna().any().any():
        raise ValueError(
            "Training data contains unexpected missing values."
        )

    if test.isna().any().any():
        raise ValueError(
            "Test data contains unexpected missing values."
        )


# ============================================================
# Command-line validation
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM")
    print("C-MAPSS FD001 DATA INGESTION")
    print("=" * 70)

    train_df = load_fd001_train()
    test_df = load_fd001_test()
    rul_df = load_fd001_rul()

    validate_fd001_dataset(
        train_df,
        test_df,
        rul_df,
    )

    print("\nDataset validation: PASSED")

    print("\nTraining data")
    print("-" * 70)
    print(f"Rows       : {len(train_df):,}")
    print(f"Columns    : {len(train_df.columns)}")
    print(f"Units      : {train_df['unit'].nunique()}")
    print(
        f"Cycle range: "
        f"{train_df['cycle'].min()} - "
        f"{train_df['cycle'].max()}"
    )

    print("\nTest data")
    print("-" * 70)
    print(f"Rows       : {len(test_df):,}")
    print(f"Columns    : {len(test_df.columns)}")
    print(f"Units      : {test_df['unit'].nunique()}")
    print(
        f"Cycle range: "
        f"{test_df['cycle'].min()} - "
        f"{test_df['cycle'].max()}"
    )

    print("\nGround-truth RUL")
    print("-" * 70)
    print(f"Values     : {len(rul_df)}")
    print(
        f"RUL range : "
        f"{rul_df['RUL'].min()} - "
        f"{rul_df['RUL'].max()}"
    )

    print("\nColumn names")
    print("-" * 70)

    for index, column in enumerate(
        train_df.columns,
        start=1,
    ):
        print(f"{index:02d}. {column}")

    print("\nFirst five training rows")
    print("-" * 70)
    print(train_df.head().to_string(index=False))

    print("\n" + "=" * 70)
    print("INGESTION CHECK COMPLETE")
    print("=" * 70)