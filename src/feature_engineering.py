
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


# ============================================================
# Feature configuration
# ============================================================

SENSOR_COLUMNS = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_17",
    "sensor_20",
    "sensor_21",
]

ROLLING_WINDOWS = [5, 10, 20]


# IMPORTANT:
# There must be NO feature here that requires knowledge of the
# future or the final cycle of a unit.
#
# In particular:
#   - cycle_progress has been removed.
#   - max(unit cycle) must never be used as an input feature.
#
# The raw cycle column is retained in the dataframe for:
#   - ordering
#   - labels
#   - diagnostics
#   - replay
#
# The model-training split code explicitly excludes "cycle".


# ============================================================
# Validation helpers
# ============================================================

def validate_required_columns(
    df: pd.DataFrame,
    required_columns: Iterable[str],
) -> None:
    """
    Validate that all required columns exist.
    """

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing required columns: "
            + ", ".join(missing)
        )


def validate_feature_input(df: pd.DataFrame) -> None:
    """
    Validate the minimum structure required to build features.
    """

    required = [
        "unit",
        "cycle",
        *SENSOR_COLUMNS,
    ]

    validate_required_columns(
        df,
        required,
    )

    if df.empty:
        raise ValueError(
            "Cannot build features from an empty dataframe."
        )

    if df["unit"].isna().any():
        raise ValueError(
            "Unit column contains missing values."
        )

    if df["cycle"].isna().any():
        raise ValueError(
            "Cycle column contains missing values."
        )

    if (df["cycle"] < 1).any():
        raise ValueError(
            "Cycle values must be >= 1."
        )

    for sensor in SENSOR_COLUMNS:
        if df[sensor].isna().any():
            raise ValueError(
                f"{sensor} contains missing values."
            )


# ============================================================
# Rolling features
# ============================================================



def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add causal rolling features per unit with batched column insertion."""
    result = df.copy()
    feature_data = {}

    for sensor in SENSOR_COLUMNS:
        grouped = result.groupby("unit", sort=False)[sensor]

        for window in ROLLING_WINDOWS:
            rolling = grouped.rolling(window=window, min_periods=1)

            feature_data[f"{sensor}_roll_mean_{window}"] = (
                rolling.mean()
                .reset_index(level=0, drop=True)
                .reindex(result.index)
            )

            feature_data[f"{sensor}_roll_std_{window}"] = (
                grouped.rolling(window=window, min_periods=2)
                .std()
                .reset_index(level=0, drop=True)
                .reindex(result.index)
                .fillna(0.0)
            )

    rolling_features = pd.DataFrame(feature_data, index=result.index)
    return pd.concat([result, rolling_features], axis=1)




# ============================================================
# Slope features
# ============================================================


def calculate_slope(
    values: np.ndarray,
) -> float:
    """Calculate the linear-regression slope for equally spaced observations."""

    n = len(values)

    if n < 2:
        return 0.0

    centered_x = (
        np.arange(n, dtype=float)
        - (n - 1) / 2.0
    )

    denominator = float(
        np.dot(centered_x, centered_x)
    )

    if denominator == 0.0:
        return 0.0

    return float(
        np.dot(centered_x, values) / denominator
    )





def add_slope_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Add causal rolling slope features."""
    result = df.copy()

    for sensor in SENSOR_COLUMNS:
        for window in ROLLING_WINDOWS:
            column = f"{sensor}_slope_{window}"

            result[column] = (
                result.groupby("unit", sort=False)[sensor]
                .transform(
                    lambda series: (
                        series.rolling(
                            window=window,
                            min_periods=2,
                        )
                        .apply(calculate_slope, raw=True)
                        .fillna(0.0)
                    )
                )
            )

    return result




# ============================================================
# Feature construction
# ============================================================

def build_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build the complete leakage-free feature dataframe.

    Important guarantees:

    1. Data is sorted by unit/cycle.
    2. Rolling features are causal.
    3. Slope features are causal.
    4. No feature uses a unit's final cycle.
    5. No cycle_progress feature is created.
    6. The raw cycle column remains available for ordering,
       labels, replay and diagnostics.

    Model-training code is responsible for excluding raw cycle
    from the actual feature matrix.
    """

    validate_feature_input(df)

    result = (
        df.copy()
        .sort_values(
            ["unit", "cycle"],
        )
        .reset_index(drop=True)
    )

    result = add_rolling_features(
        result
    )

    result = add_slope_features(
        result
    )

    return result


# ============================================================
# Feature metadata
# ============================================================

def get_engineered_feature_columns(
    df: pd.DataFrame,
) -> list[str]:
    """
    Return all columns created by the feature pipeline.

    This is useful for diagnostics and tests.

    The function deliberately does not include:
        unit
        cycle
        RUL
        failure_risk
    """

    excluded = {
        "unit",
        "cycle",
        "RUL",
        "failure_risk",
    }

    return [
        column
        for column in df.columns
        if column not in excluded
    ]


# ============================================================
# Feature validation
# ============================================================

def validate_features(
    df: pd.DataFrame,
) -> None:
    """
    Validate the structural properties of the leakage-free
    feature dataframe.

    This intentionally avoids hard-coding a particular row count
    or number of units so that the same feature pipeline can be
    used for:
        - training
        - validation
        - test data
        - replay
        - unit tests
    """

    validate_feature_input(df)

    if "cycle_progress" in df.columns:
        raise ValueError(
            "Leakage detected: cycle_progress must not exist."
        )

    forbidden_future_features = [
        "final_cycle",
        "max_cycle",
        "unit_max_cycle",
    ]

    present_forbidden = [
        column
        for column in forbidden_future_features
        if column in df.columns
    ]

    if present_forbidden:
        raise ValueError(
            "Future-information feature(s) detected: "
            + ", ".join(present_forbidden)
        )

    engineered_columns = (
        get_engineered_feature_columns(df)
    )

    if not engineered_columns:
        raise ValueError(
            "No model features were generated."
        )

    for column in engineered_columns:

        if df[column].isna().any():
            raise ValueError(
                f"Feature '{column}' contains NaN values."
            )

        if not np.isfinite(
            df[column].to_numpy(
                dtype=float,
            )
        ).all():
            raise ValueError(
                f"Feature '{column}' contains "
                "non-finite values."
            )


# ============================================================
# Truncation consistency check
# ============================================================

def compare_truncated_features(
    full_df: pd.DataFrame,
    unit: int,
    cycle: int,
    tolerance: float = 1e-10,
) -> tuple[bool, str]:
    """
    Verify that features calculated from:

        full unit history truncated to cycle t

    are identical to features calculated from:

        the original full unit history, evaluated through t.

    This protects against accidental look-ahead leakage.

    Returns:
        (passed, message)
    """

    validate_feature_input(full_df)

    unit_full = (
        full_df[
            full_df["unit"] == unit
        ]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    if unit_full.empty:
        return (
            False,
            f"Unit {unit} was not found.",
        )

    available_cycles = unit_full["cycle"].tolist()

    if cycle not in available_cycles:
        return (
            False,
            f"Cycle {cycle} was not found for unit {unit}.",
        )

    truncated_raw = unit_full[
        unit_full["cycle"] <= cycle
    ].copy()

    full_features = build_features(
        unit_full
    )

    truncated_features = build_features(
        truncated_raw
    )

    full_row = full_features[
        full_features["cycle"] == cycle
    ]

    truncated_row = truncated_features[
        truncated_features["cycle"] == cycle
    ]

    if full_row.empty or truncated_row.empty:
        return (
            False,
            "Unable to locate comparison cycle.",
        )

    comparison_columns = [
        column
        for column in truncated_features.columns
        if column not in {
            "RUL",
            "failure_risk",
        }
    ]

    left = (
        full_row[comparison_columns]
        .reset_index(drop=True)
    )

    right = (
        truncated_row[comparison_columns]
        .reset_index(drop=True)
    )

    if list(left.columns) != list(right.columns):
        return (
            False,
            "Feature column sets differ.",
        )

    for column in comparison_columns:

        if pd.api.types.is_numeric_dtype(
            left[column]
        ):

            if not np.allclose(
                left[column].to_numpy(
                    dtype=float,
                ),
                right[column].to_numpy(
                    dtype=float,
                ),
                rtol=tolerance,
                atol=tolerance,
            ):
                return (
                    False,
                    (
                        "Mismatch detected in "
                        f"feature '{column}'."
                    ),
                )

        else:

            if not left[column].equals(
                right[column]
            ):
                return (
                    False,
                    (
                        "Mismatch detected in "
                        f"column '{column}'."
                    ),
                )

    return (
        True,
        (
            f"Unit {unit}, cycle {cycle}: "
            "truncated/full feature consistency passed."
        ),
    )


# ============================================================
# Standalone verification
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("LEAKAGE-FREE FEATURE ENGINEERING CHECK")
    print("=" * 70)

    print(
        f"Sensors: {len(SENSOR_COLUMNS)}"
    )

    print(
        f"Rolling windows: {ROLLING_WINDOWS}"
    )

    print(
        "cycle_progress: REMOVED"
    )

    print(
        "final-cycle model features: REMOVED"
    )

    print(
        "raw cycle retained for ordering/replay only"
    )
