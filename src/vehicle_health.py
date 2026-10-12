
from pathlib import Path

import pandas as pd

from src.diagnostic_report import build_diagnostic_report
from src.inference import build_inference_features
from src.subsystem_health import (
    LIVE_BASELINE_CYCLES,
    add_live_health_statuses,
    calculate_live_group_health,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fd001_train_labeled.parquet"
)


def load_vehicle_history(unit: int) -> pd.DataFrame:
    """Load historical telemetry for one vehicle/unit."""

    if not LABELED_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Historical dataset not found: {LABELED_DATA_PATH}"
        )

    df = pd.read_parquet(LABELED_DATA_PATH)

    vehicle_df = (
        df[df["unit"] == unit]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    if vehicle_df.empty:
        raise ValueError(
            f"Vehicle/unit {unit} was not found."
        )

    return vehicle_df


def build_vehicle_health_history(
    unit: int,
    history_points: int = 10,
) -> dict:
    """
    Build current vehicle health and recent health history.

    Engineered features are calculated once for the complete
    trajectory. Each historical prediction uses only the feature
    rows through its corresponding cycle.

    Subsystem health is computed once for the full trajectory,
    with prefix-specific calculations retained before baseline
    completion.
    """

    vehicle_df = load_vehicle_history(unit)

    # --------------------------------------------------------
    # 1. Engineer features once
    # --------------------------------------------------------
    # Feature-equivalence tests verify that the row for a given
    # cycle matches the row generated from that cycle's prefix.
    vehicle_features = build_inference_features(vehicle_df)

    history_points = max(
        1,
        min(history_points, len(vehicle_df)),
    )

    step = max(
        1,
        len(vehicle_df) // history_points,
    )

    selected_indices = list(
        range(0, len(vehicle_df), step)
    )

    last_index = len(vehicle_df) - 1

    # Always include the latest cycle.
    selected_indices = [
        index
        for index in selected_indices
        if index != last_index
    ]

    if history_points == 1:
        selected_indices = [last_index]
    else:
        selected_indices = selected_indices[
            -(history_points - 1):
        ]
        selected_indices.append(last_index)

    # --------------------------------------------------------
    # 2. Calculate subsystem health once
    # --------------------------------------------------------
    full_health = calculate_live_group_health(vehicle_df)
    full_health = add_live_health_statuses(full_health)

    # Align health rows with sorted telemetry rows.
    health_by_index = full_health.reset_index(drop=True)

    # --------------------------------------------------------
    # 3. Build historical predictions
    # --------------------------------------------------------
    history = []

    for index in selected_indices:
        # Telemetry available through this cycle.
        telemetry_window = vehicle_df.iloc[
            : index + 1
        ].copy()

        # Features through this cycle only. The diagnostic report
        # predicts from the final row of this feature prefix.
        feature_window = vehicle_features.iloc[
            : index + 1
        ]

        if index + 1 < LIVE_BASELINE_CYCLES:
            report = build_diagnostic_report(
                telemetry_window,
                include_shap=False,
                precomputed_features_df=feature_window,
            )
        else:
            report = build_diagnostic_report(
                telemetry_window,
                include_shap=False,
                precomputed_health_df=health_by_index.iloc[
                    [index]
                ].reset_index(drop=True),
                precomputed_features_df=feature_window,
            )

        # ----------------------------------------------------
        # 4. Select health scores for this cycle
        # ----------------------------------------------------
        if index + 1 < LIVE_BASELINE_CYCLES:
            # Preserve prefix-specific health behavior before
            # the initial baseline has been completed.
            prefix_health = calculate_live_group_health(
                telemetry_window
            )
            prefix_health = add_live_health_statuses(
                prefix_health
            )
            latest_health = prefix_health.iloc[-1]
        else:
            latest_health = health_by_index.iloc[index]

        subsystem_history = {}

        for name in (
            "temperature",
            "pressure",
            "speed_flow",
            "thermal_bleed",
        ):
            subsystem_history[name] = {
                "health": round(
                    float(latest_health[f"{name}_health"]),
                    2,
                ),
                "status": str(
                    latest_health[f"{name}_status"]
                ),
            }

        history.append(
            {
                "cycle": report["current_cycle"],
                "failure_probability": round(
                    report["failure_probability"],
                    4,
                ),
                "predicted_rul": round(
                    report["predicted_rul"],
                    2,
                ),
                "severity": report["severity"],
                "overall_sensor_health": round(
                    float(
                        latest_health["overall_sensor_health"]
                    ),
                    2,
                ),
                "overall_health_status": str(
                    latest_health["overall_health_status"]
                ),
                "subsystem_health": subsystem_history,
            }
        )

    # --------------------------------------------------------
    # 5. Return the existing response structure
    # --------------------------------------------------------
    current = history[-1]

    return {
        "unit": unit,
        "current_cycle": current["cycle"],
        "failure_probability": current["failure_probability"],
        "predicted_rul": current["predicted_rul"],
        "severity": current["severity"],
        "overall_sensor_health": current[
            "overall_sensor_health"
        ],
        "overall_health_status": current[
            "overall_health_status"
        ],
        "subsystem_health": current["subsystem_health"],
        "history": history,
    }
