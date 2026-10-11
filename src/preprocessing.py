
from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = (
    BASE_DIR
    / "data"
    / "raw"
    / "cmapss"
)

PROCESSED_DATA_DIR = (
    BASE_DIR
    / "data"
    / "processed"
)


# ============================================================
# C-MAPSS columns
# ============================================================

ALL_SENSORS = [
    f"sensor_{index}"
    for index in range(1, 22)
]

SETTING_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
]

LOW_INFORMATION_SENSORS = [
    "sensor_1",
    "sensor_5",
    "sensor_6",
    "sensor_10",
    "sensor_16",
    "sensor_18",
    "sensor_19",
]

SELECTED_SENSORS = [
    sensor
    for sensor in ALL_SENSORS
    if sensor not in LOW_INFORMATION_SENSORS
]


# ============================================================
# Target configuration
# ============================================================

MAX_RUL = 125
FAILURE_RUL_THRESHOLD = 30


# ============================================================
# Expected raw columns
# ============================================================

RAW_COLUMNS = [
    "unit",
    "cycle",
    *SETTING_COLUMNS,
    *ALL_SENSORS,
]


# ============================================================
# Data loading
# ============================================================

def load_fd001_train() -> pd.DataFrame:
    """
    Load the official FD001 training telemetry.

    The expected raw file is:
        train_FD001.txt
    """

    path = RAW_DATA_DIR / "train_FD001.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"FD001 training file not found: {path}"
        )

    df = pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
    )

    return df


def load_fd001_test() -> pd.DataFrame:
    """
    Load the official FD001 test telemetry.
    """

    path = RAW_DATA_DIR / "test_FD001.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"FD001 test file not found: {path}"
        )

    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
    )


def load_fd001_rul() -> pd.Series:
    """
    Load the official FD001 test-set RUL values.

    Each value represents the number of cycles remaining after
    the final observed cycle of the corresponding test engine.
    """

    path = RAW_DATA_DIR / "RUL_FD001.txt"

    if not path.exists():
        raise FileNotFoundError(
            f"FD001 RUL file not found: {path}"
        )

    rul_df = pd.read_csv(
        path,
        header=None,
    )

    if rul_df.empty:
        raise ValueError(
            "RUL_FD001.txt is empty."
        )

    return (
        pd.to_numeric(
            rul_df.iloc[:, 0],
            errors="raise",
        )
        .astype(float)
    )


# ============================================================
# Validation
# ============================================================

def validate_raw_dataframe(
    df: pd.DataFrame,
) -> None:
    """
    Validate raw C-MAPSS structure.
    """

    missing = [
        column
        for column in RAW_COLUMNS
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing raw columns: "
            + ", ".join(missing)
        )

    if df.empty:
        raise ValueError(
            "Input dataframe is empty."
        )

    if df["unit"].isna().any():
        raise ValueError(
            "Unit contains missing values."
        )

    if df["cycle"].isna().any():
        raise ValueError(
            "Cycle contains missing values."
        )

    if (df["unit"] < 1).any():
        raise ValueError(
            "Unit identifiers must be >= 1."
        )

    if (df["cycle"] < 1).any():
        raise ValueError(
            "Cycle values must be >= 1."
        )

    for column in (
        SETTING_COLUMNS
        + ALL_SENSORS
    ):
        if df[column].isna().any():
            raise ValueError(
                f"{column} contains missing values."
            )


# ============================================================
# RUL labels
# ============================================================

def add_rul_labels(
    df: pd.DataFrame,
    max_rul: int = MAX_RUL,
) -> pd.DataFrame:
    """
    Add training RUL labels.

    IMPORTANT:

    The final cycle is used ONLY to construct the target label.

    It is immediately removed after label construction and is
    never passed into feature engineering as a model feature.

    This is valid supervised-learning target construction.

    Example:

        final_cycle(unit) - current_cycle

    is the target.

    The model must never receive:
        final_cycle
        max_cycle
        cycle_progress
        or any other feature derived from future observations.
    """

    validate_raw_dataframe(df)

    result = (
        df.copy()
        .sort_values(
            ["unit", "cycle"],
        )
        .reset_index(drop=True)
    )

    final_cycles = (
        result
        .groupby("unit")["cycle"]
        .max()
        .rename("final_cycle")
    )

    result = result.merge(
        final_cycles,
        on="unit",
        how="left",
        validate="many_to_one",
    )

    result["RUL"] = (
        result["final_cycle"]
        - result["cycle"]
    ).clip(
        lower=0,
        upper=max_rul,
    )

    # Critical leakage boundary:
    #
    # final_cycle has served its sole purpose as the target
    # construction value. It must never survive into feature
    # engineering/model training.
    result.drop(
        columns=["final_cycle"],
        inplace=True,
    )

    return result


# ============================================================
# Failure-risk labels
# ============================================================

def add_failure_risk_label(
    df: pd.DataFrame,
    threshold: int = FAILURE_RUL_THRESHOLD,
) -> pd.DataFrame:
    """
    Create binary failure-risk labels.

    failure_risk = 1 when RUL <= threshold.
    """

    if "RUL" not in df.columns:
        raise ValueError(
            "RUL must exist before failure_risk is created."
        )

    result = df.copy()

    result["failure_risk"] = (
        result["RUL"] <= threshold
    ).astype("int8")

    return result


# ============================================================
# Sensor selection
# ============================================================

def remove_low_information_sensors(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove sensors identified as low-information for FD001.
    """

    result = df.copy()

    columns_to_drop = [
        column
        for column in LOW_INFORMATION_SENSORS
        if column in result.columns
    ]

    return result.drop(
        columns=columns_to_drop
    )


# ============================================================
# Complete FD001 training preparation
# ============================================================

def prepare_fd001_training_data(
    max_rul: int = MAX_RUL,
    failure_threshold: int = FAILURE_RUL_THRESHOLD,
) -> pd.DataFrame:
    """
    Prepare the complete FD001 training dataframe.

    Pipeline:

        raw telemetry
            ↓
        RUL target construction
            ↓
        failure-risk target construction
            ↓
        low-information sensor removal

    No model features are generated here.
    """

    df = load_fd001_train()

    df = add_rul_labels(
        df,
        max_rul=max_rul,
    )

    df = add_failure_risk_label(
        df,
        threshold=failure_threshold,
    )

    df = remove_low_information_sensors(
        df
    )

    return df


# ============================================================
# Label validation
# ============================================================

def validate_labels(
    df: pd.DataFrame,
) -> None:
    """
    Validate target columns.
    """

    required = {
        "RUL",
        "failure_risk",
    }

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing target columns: "
            + ", ".join(missing)
        )

    if df["RUL"].isna().any():
        raise ValueError(
            "RUL contains missing values."
        )

    if (
        df["RUL"] < 0
    ).any():
        raise ValueError(
            "RUL contains negative values."
        )

    if (
        df["RUL"] > MAX_RUL
    ).any():
        raise ValueError(
            f"RUL exceeds configured cap of {MAX_RUL}."
        )

    unique_risk_values = set(
        df["failure_risk"].unique()
    )

    if not unique_risk_values.issubset(
        {0, 1}
    ):
        raise ValueError(
            "failure_risk must contain only 0 and 1."
        )


# ============================================================
# Save processed training data
# ============================================================

def save_fd001_training_data(
    df: pd.DataFrame,
) -> Path:
    """
    Save the labeled FD001 training dataframe.
    """

    PROCESSED_DATA_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        PROCESSED_DATA_DIR
        / "fd001_train_labeled.parquet"
    )

    df.to_parquet(
        output_path,
        index=False,
    )

    return output_path


# ============================================================
# Standalone preprocessing
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("FD001 LEAKAGE-SAFE PREPROCESSING")
    print("=" * 70)

    data = prepare_fd001_training_data()

    validate_labels(data)

    output_path = save_fd001_training_data(
        data
    )

    print(
        f"Rows: {len(data):,}"
    )

    print(
        f"Units: {data['unit'].nunique():,}"
    )

    print(
        f"Sensors retained: {len(SELECTED_SENSORS)}"
    )

    print(
        f"RUL cap: {MAX_RUL}"
    )

    print(
        f"Failure threshold: {FAILURE_RUL_THRESHOLD}"
    )

    print(
        f"Failure-risk positives: "
        f"{int(data['failure_risk'].sum()):,}"
    )

    print(
        f"Saved: {output_path}"
    )

    print()
    print(
        "Leakage boundary verified:"
    )
    print(
        "  final_cycle -> target construction only"
    )
    print(
        "  final_cycle -> removed"
    )
    print(
        "  cycle_progress -> not created"
    )
    print(
        "  future-derived model features -> none"
    )