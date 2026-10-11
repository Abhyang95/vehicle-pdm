
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

LABELED_FILE = PROCESSED_DIR / "fd001_train_labeled.parquet"
FEATURE_FILE = PROCESSED_DIR / "fd001_train_features.parquet"

DIAGNOSTICS_DIR = PROCESSED_DIR / "diagnostics"
OUTPUT_FILE = DIAGNOSTICS_DIR / "subsystem_health_results.csv"

# Engineering-oriented sensor groups.
# These are NOT manufacturer-defined component-health labels.

SENSOR_GROUPS = {
    "temperature": [
        "sensor_2",
        "sensor_3",
        "sensor_4",
    ],
    "pressure": [
        "sensor_7",
        "sensor_11",
    ],
    "speed_flow": [
        "sensor_8",
        "sensor_9",
        "sensor_12",
        "sensor_13",
        "sensor_14",
        "sensor_15",
    ],
    "thermal_bleed": [
        "sensor_17",
        "sensor_20",
        "sensor_21",
    ],
}

GROUP_DISPLAY_NAMES = {
    "temperature": "Temperature",
    "pressure": "Pressure",
    "speed_flow": "Speed / Flow",
    "thermal_bleed": "Thermal / Bleed",
}

# Shared scoring configuration.
MAD_SCALE = 1.4826
MAD_EPSILON = 1e-8
HEALTH_DEVIATION_RANGE = 4.0

# Number of initial observations per unit used for the live baseline.
# This is a count, not a percentage of the supplied history.
LIVE_BASELINE_CYCLES = 20


def load_data():
    """Load processed training labels and engineered sensor features."""
    labeled_df = pd.read_parquet(LABELED_FILE)
    features_df = pd.read_parquet(FEATURE_FILE)

    return labeled_df, features_df


def validate_sensor_groups(features_df):
    """Raise a clear error if any required sensor column is missing."""
    missing = sorted({
        sensor
        for sensors in SENSOR_GROUPS.values()
        for sensor in sensors
        if sensor not in features_df.columns
    })

    if missing:
        raise ValueError(
            "Missing required sensor columns: "
            + ", ".join(missing)
        )


def _robust_reference(values):
    """Calculate a robust median and MAD reference."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    if values.size == 0:
        raise ValueError(
            "Cannot calculate a sensor baseline without finite values."
        )

    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    if not np.isfinite(mad) or mad < MAD_EPSILON:
        mad = 1.0

    return median, mad


def _health_from_values(values, reference_median, reference_mad):
    """Convert robust sensor deviation into an approximate 0–100 score."""
    values = np.asarray(values, dtype=float)

    deviation = (
        np.abs(values - reference_median)
        / (MAD_SCALE * reference_mad)
    )

    health = 100.0 * (
        1.0
        - np.clip(
            (deviation - 1.0) / HEALTH_DEVIATION_RANGE,
            0.0,
            1.0,
        )
    )

    # Keep missing sensor values missing rather than silently treating
    # them as healthy.
    health[~np.isfinite(values)] = np.nan

    return health


def _add_group_scores(df, reference_values_by_sensor):
    """Add sensor-group scores and overall approximate health."""
    result = df.copy()

    for group, sensors in SENSOR_GROUPS.items():
        sensor_scores = []

        for sensor in sensors:
            if sensor not in result.columns:
                continue

            reference_values = reference_values_by_sensor.get(sensor)

            if reference_values is None:
                continue

            reference_median, reference_mad = reference_values

            score = _health_from_values(
                result[sensor].astype(float).to_numpy(),
                reference_median,
                reference_mad,
            )

            sensor_scores.append(
                pd.Series(score, index=result.index)
            )

        if sensor_scores:
            result[f"{group}_health"] = (
                pd.concat(sensor_scores, axis=1)
                .mean(axis=1)
                .round(2)
            )

    health_columns = [
        f"{group}_health"
        for group in SENSOR_GROUPS
        if f"{group}_health" in result.columns
    ]

    if not health_columns:
        raise ValueError(
            "No subsystem health scores could be calculated."
        )

    result["overall_sensor_health"] = (
        result[health_columns]
        .mean(axis=1)
        .round(2)
    )

    return result


def calculate_group_health(features_df, labeled_df):
    """
    Calculate offline approximate sensor-group health.

    A global reference distribution is estimated from the first 20%
    of each training unit's trajectory. This function is intended
    for retrospective dataset analysis, not live replay.

    Scores represent sensor deviation, not physical component health.
    """
    validate_sensor_groups(features_df)

    required_label_columns = {
        "unit",
        "cycle",
        "RUL",
        "failure_risk",
    }

    missing_labels = required_label_columns - set(labeled_df.columns)

    if missing_labels:
        raise ValueError(
            "Missing label columns: "
            + ", ".join(sorted(missing_labels))
        )

    required_feature_columns = {"unit", "cycle"}
    missing_features = required_feature_columns - set(features_df.columns)

    if missing_features:
        raise ValueError(
            "Feature data is missing columns: "
            + ", ".join(sorted(missing_features))
        )

    # Align feature and label records explicitly by unit and cycle.
    feature_columns = [
        "unit",
        "cycle",
        *[
            sensor
            for sensors in SENSOR_GROUPS.values()
            for sensor in sensors
        ],
    ]

    features = features_df[feature_columns].copy()
    labels = labeled_df[
        ["unit", "cycle", "RUL", "failure_risk"]
    ].copy()

    features["unit"] = features["unit"].astype(int)
    features["cycle"] = features["cycle"].astype(int)
    labels["unit"] = labels["unit"].astype(int)
    labels["cycle"] = labels["cycle"].astype(int)

    if features.duplicated(["unit", "cycle"]).any():
        raise ValueError(
            "Feature data contains duplicate unit/cycle records."
        )

    if labels.duplicated(["unit", "cycle"]).any():
        raise ValueError(
            "Label data contains duplicate unit/cycle records."
        )

    merged = labels.merge(
        features,
        on=["unit", "cycle"],
        how="left",
        validate="one_to_one",
        indicator=True,
    )

    unmatched = int((merged["_merge"] != "both").sum())

    if unmatched:
        raise ValueError(
            f"{unmatched} label records could not be matched "
            "to sensor features by unit and cycle."
        )

    merged = (
        merged.drop(columns="_merge")
        .sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )

    reference_values_by_sensor = {}

    for sensors in SENSOR_GROUPS.values():
        for sensor in sensors:
            early_parts = []

            for _, unit_df in merged.groupby("unit", sort=False):
                unit_df = unit_df.sort_values("cycle")

                cycle_limit = (
                    unit_df["cycle"].max() * 0.20
                )

                early = unit_df.loc[
                    unit_df["cycle"] <= cycle_limit,
                    sensor,
                ]

                if not early.empty:
                    early_parts.append(early.to_numpy(dtype=float))

            if not early_parts:
                raise ValueError(
                    f"No early-life reference values for {sensor}."
                )

            reference_values_by_sensor[sensor] = _robust_reference(
                np.concatenate(early_parts)
            )

    result = _add_group_scores(
        merged,
        reference_values_by_sensor,
    )

    return result


def calculate_live_group_health(
    telemetry_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate approximate subsystem health for supplied telemetry.

    The first LIVE_BASELINE_CYCLES observations of each unit in the
    supplied history are used to estimate its baseline.

    Important: this baseline is stable only while the same initial
    observations remain available. For strict replay consistency
    across separate requests, persist and reuse calibration statistics.

    This is an engineering abstraction, not a physical component-health
    model.
    """
    if telemetry_df.empty:
        raise ValueError(
            "Telemetry input cannot be empty."
        )

    required_columns = {"unit", "cycle"}
    missing_columns = required_columns - set(telemetry_df.columns)

    if missing_columns:
        raise ValueError(
            "Telemetry is missing columns: "
            + ", ".join(sorted(missing_columns))
        )

    validate_sensor_groups(telemetry_df)

    df = telemetry_df.copy()

    df["unit"] = df["unit"].astype(int)
    df["cycle"] = df["cycle"].astype(int)

    if df.duplicated(["unit", "cycle"]).any():
        raise ValueError(
            "Telemetry contains duplicate unit/cycle records."
        )

    df = (
        df.sort_values(["unit", "cycle"])
        .reset_index(drop=True)
    )

    reference_values_by_sensor = {}

    for sensors in SENSOR_GROUPS.values():
        for sensor in sensors:
            reference_parts = []

            for _, unit_df in df.groupby("unit", sort=False):
                unit_df = unit_df.sort_values("cycle")

                # Fixed count, not 20% of the current history length.
                baseline = unit_df[sensor].iloc[
                    :LIVE_BASELINE_CYCLES
                ].to_numpy(dtype=float)

                reference_parts.append(baseline)

            if not reference_parts:
                raise ValueError(
                    f"No baseline observations available for {sensor}."
                )

            reference_values_by_sensor[sensor] = _robust_reference(
                np.concatenate(reference_parts)
            )

    result = _add_group_scores(
        df[["unit", "cycle", *[
            sensor
            for sensors in SENSOR_GROUPS.values()
            for sensor in sensors
        ]]],
        reference_values_by_sensor,
    )

    return result[
        [
            "unit",
            "cycle",
            *[
                f"{group}_health"
                for group in SENSOR_GROUPS
            ],
            "overall_sensor_health",
        ]
    ]


def assign_health_status(score):
    """Map an approximate health score to its display status."""
    if pd.isna(score):
        return "UNKNOWN"

    if score >= 80:
        return "HEALTHY"

    if score >= 60:
        return "WATCH"

    if score >= 40:
        return "DEGRADED"

    return "CRITICAL"


def add_health_statuses(diagnostics_df):
    """Add subsystem and overall status columns in place."""
    for group in SENSOR_GROUPS:
        health_column = f"{group}_health"
        status_column = f"{group}_status"

        diagnostics_df[status_column] = (
            diagnostics_df[health_column]
            .apply(assign_health_status)
        )

    diagnostics_df["overall_health_status"] = (
        diagnostics_df["overall_sensor_health"]
        .apply(assign_health_status)
    )

    return diagnostics_df


def add_live_health_statuses(
    diagnostics_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return live diagnostics with subsystem and overall statuses."""
    result = diagnostics_df.copy()

    for group in SENSOR_GROUPS:
        health_column = f"{group}_health"
        status_column = f"{group}_status"

        result[status_column] = (
            result[health_column]
            .apply(assign_health_status)
        )

    result["overall_health_status"] = (
        result["overall_sensor_health"]
        .apply(assign_health_status)
    )

    return result


def print_summary(diagnostics_df):
    """Print aggregate health statistics and status distribution."""
    print("\n" + "=" * 70)
    print("APPROXIMATE SENSOR-GROUP HEALTH SUMMARY")
    print("=" * 70)

    print("\nAverage health scores:")

    for group, column in (
        (group, f"{group}_health")
        for group in SENSOR_GROUPS
    ):
        display_name = GROUP_DISPLAY_NAMES[group]
        value = diagnostics_df[column].mean()

        print(
            f"  {display_name:<20} : {value:.2f}/100"
        )

    print("\nOverall sensor health:")
    print(
        diagnostics_df["overall_sensor_health"]
        .describe()
        .round(2)
        .to_string()
    )

    print("\nOverall health status distribution:")
    print(
        diagnostics_df["overall_health_status"]
        .value_counts(dropna=False)
        .to_string()
    )


def show_highest_risk_cases(diagnostics_df):
    """Display the lowest-health observations among labeled high-risk rows."""
    health_columns = [
        f"{group}_health"
        for group in SENSOR_GROUPS
    ]

    display_columns = [
        "unit",
        "cycle",
        "RUL",
        "failure_risk",
        "overall_sensor_health",
        *health_columns,
    ]

    print("\n" + "=" * 70)
    print("LOWEST-HEALTH HIGH-RISK OBSERVATIONS")
    print("=" * 70)

    high_risk = diagnostics_df[
        diagnostics_df["failure_risk"] == 1
    ]

    if high_risk.empty:
        print("\nNo high-risk observations found.")
        return

    print(
        high_risk[display_columns]
        .sort_values("overall_sensor_health")
        .head(10)
        .round(2)
        .to_string(index=False)
    )


if __name__ == "__main__":
    print("=" * 70)
    print("VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM")
    print("APPROXIMATE SUBSYSTEM HEALTH")
    print("=" * 70)

    labeled_df, features_df = load_data()

    print(f"\nObservations : {len(features_df):,}")
    print(f"Units        : {features_df['unit'].nunique():,}")

    print("\nSensor groups:")
    for group, sensors in SENSOR_GROUPS.items():
        print(
            f"  {GROUP_DISPLAY_NAMES[group]:<20} "
            f": {', '.join(sensors)}"
        )

    diagnostics_df = calculate_group_health(
        features_df,
        labeled_df,
    )

    diagnostics_df = add_health_statuses(
        diagnostics_df
    )

    print_summary(diagnostics_df)
    show_highest_risk_cases(diagnostics_df)

    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"\nSaved: {OUTPUT_FILE}")

    print("\nNOTE:")
    print(
        "These scores represent approximate sensor-group health "
        "derived from telemetry abnormality."
    )
    print(
        "They are not manufacturer-defined component-health measurements."
    )

    print("\n" + "=" * 70)
    print("SUBSYSTEM HEALTH COMPLETE")
    print("=" * 70)
