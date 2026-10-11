from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# Project setup
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

sys.path.insert(0, str(SRC_DIR))

from ingestion import load_fd001_train


PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# Configuration
# ============================================================

SENSOR_COLUMNS = [
    f"sensor_{i}"
    for i in range(1, 22)
]

SETTING_COLUMNS = [
    "setting_1",
    "setting_2",
    "setting_3",
]


# ============================================================
# Load data
# ============================================================

df = load_fd001_train()

print("=" * 70)
print("VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM")
print("FD001 EXPLORATORY DATA ANALYSIS")
print("=" * 70)

print(f"\nDataset shape: {df.shape}")
print(f"Number of units: {df['unit'].nunique()}")


# ============================================================
# 1. Missing-value analysis
# ============================================================

print("\n" + "=" * 70)
print("1. MISSING VALUE ANALYSIS")
print("=" * 70)

missing = df.isna().sum()

missing_table = pd.DataFrame({
    "column": missing.index,
    "missing_count": missing.values,
})

missing_table["missing_percentage"] = (
    missing_table["missing_count"]
    / len(df)
    * 100
)

print(
    missing_table[
        missing_table["missing_count"] > 0
    ]
)

if missing.sum() == 0:
    print("\nNo missing values detected.")


# ============================================================
# 2. Constant / near-constant sensor analysis
# ============================================================

print("\n" + "=" * 70)
print("2. SENSOR VARIABILITY ANALYSIS")
print("=" * 70)

sensor_stats = []

for sensor in SENSOR_COLUMNS:

    series = df[sensor]

    sensor_stats.append({
        "sensor": sensor,
        "unique_values": series.nunique(),
        "mean": series.mean(),
        "std": series.std(),
        "min": series.min(),
        "max": series.max(),
        "range": series.max() - series.min(),
    })

sensor_stats = pd.DataFrame(sensor_stats)

sensor_stats = sensor_stats.sort_values(
    "std"
)

print(
    sensor_stats.to_string(
        index=False
    )
)


# ============================================================
# 3. Identify constant sensors
# ============================================================

constant_sensors = sensor_stats[
    sensor_stats["std"] == 0
]["sensor"].tolist()

near_constant_sensors = sensor_stats[
    sensor_stats["unique_values"] <= 2
]["sensor"].tolist()

print("\nConstant sensors:")
print(
    constant_sensors
    if constant_sensors
    else "None"
)

print("\nNear-constant sensors (<=2 unique values):")
print(
    near_constant_sensors
    if near_constant_sensors
    else "None"
)


# ============================================================
# 4. Save sensor statistics
# ============================================================

sensor_stats_path = (
    PROCESSED_DIR
    / "sensor_statistics.csv"
)

sensor_stats.to_csv(
    sensor_stats_path,
    index=False,
)

print(
    f"\nSensor statistics saved to:\n"
    f"{sensor_stats_path}"
)


# ============================================================
# 5. Vehicle lifetime distribution
# ============================================================

print("\n" + "=" * 70)
print("3. VEHICLE LIFETIME ANALYSIS")
print("=" * 70)

lifetime = (
    df.groupby("unit")["cycle"]
    .max()
)

print(lifetime.describe())

plt.figure(figsize=(10, 6))

plt.hist(
    lifetime,
    bins=20,
)

plt.title(
    "FD001 Vehicle / Engine Lifetime Distribution"
)

plt.xlabel("Cycles Until Simulated Failure")
plt.ylabel("Number of Units")

plt.tight_layout()

lifetime_path = (
    PROCESSED_DIR
    / "vehicle_lifetime_distribution.png"
)

plt.savefig(
    lifetime_path,
    dpi=150,
)

plt.close()

print(
    f"\nLifetime plot saved to:\n"
    f"{lifetime_path}"
)


# ============================================================
# 6. Sensor trajectories for representative units
# ============================================================

print("\n" + "=" * 70)
print("4. SENSOR TRAJECTORY ANALYSIS")
print("=" * 70)

representative_units = [
    1,
    50,
    100,
]

selected_sensors = [
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_7",
    "sensor_11",
    "sensor_15",
]

for unit_id in representative_units:

    unit_df = df[
        df["unit"] == unit_id
    ]

    for sensor in selected_sensors:

        plt.figure(figsize=(10, 5))

        plt.plot(
            unit_df["cycle"],
            unit_df[sensor],
        )

        plt.title(
            f"Unit {unit_id} — {sensor}"
        )

        plt.xlabel("Cycle")
        plt.ylabel(sensor)

        plt.tight_layout()

        output_path = (
            PROCESSED_DIR
            / f"unit_{unit_id}_{sensor}.png"
        )

        plt.savefig(
            output_path,
            dpi=120,
        )

        plt.close()


print(
    "\nRepresentative sensor trajectory plots generated."
)


# ============================================================
# 7. Sensor correlation
# ============================================================

print("\n" + "=" * 70)
print("5. SENSOR CORRELATION ANALYSIS")
print("=" * 70)

correlation = df[
    SENSOR_COLUMNS
].corr()

correlation_path = (
    PROCESSED_DIR
    / "sensor_correlation.csv"
)

correlation.to_csv(
    correlation_path
)

print(
    f"Correlation matrix saved to:\n"
    f"{correlation_path}"
)


# ============================================================
# 8. Summary
# ============================================================

print("\n" + "=" * 70)
print("EDA COMPLETE")
print("=" * 70)

print(
    f"""
Dataset
-------
Rows: {len(df):,}
Units: {df['unit'].nunique()}
Sensors: {len(SENSOR_COLUMNS)}

Missing values
--------------
{missing.sum()}

Constant sensors
----------------
{constant_sensors if constant_sensors else "None"}

Near-constant sensors
---------------------
{near_constant_sensors if near_constant_sensors else "None"}

Outputs
-------
{PROCESSED_DIR}
"""
)