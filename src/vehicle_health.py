from pathlib import Path

import pandas as pd

from src.diagnostic_report import build_diagnostic_report
from src.subsystem_health import calculate_live_group_health


PROJECT_ROOT = Path(__file__).resolve().parents[1]

LABELED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fd001_train_labeled.parquet"
)


def load_vehicle_history(unit: int) -> pd.DataFrame:
    """
    Load historical telemetry for one vehicle/unit.
    """

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

    Historical points are generated from progressively larger
    telemetry windows using the existing diagnostic pipeline.
    """

    vehicle_df = load_vehicle_history(unit)

    history_points = max(
        1,
        min(history_points, len(vehicle_df)),
    )

    step = max(
        1,
        len(vehicle_df) // history_points,
    )

    selected_indices = list(
        range(
            0,
            len(vehicle_df),
            step,
        )
    )

    last_index = len(vehicle_df) - 1

    # Always include the latest telemetry observation.
    selected_indices = [
        index
        for index in selected_indices
        if index != last_index
    ]

    selected_indices = selected_indices[
        -(history_points - 1):
    ]

    selected_indices.append(last_index)

    history = []

    for index in selected_indices:

        telemetry_window = vehicle_df.iloc[
            : index + 1
        ].copy()

        report = build_diagnostic_report(
            telemetry_window
        )

        subsystem_history = {
            name: {
                "health": details["health"],
                "status": details["status"],
            }
            for name, details
            in report["subsystem_health"].items()
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
                    report["overall_sensor_health"],
                    2,
                ),
                "overall_health_status": (
                    report["overall_health_status"]
                ),
                "subsystem_health": subsystem_history,
            }
        )

    current = history[-1]

    return {
        "unit": unit,
        "current_cycle": current["cycle"],
        "failure_probability": current[
            "failure_probability"
        ],
        "predicted_rul": current[
            "predicted_rul"
        ],
        "severity": current["severity"],
        "overall_sensor_health": current[
            "overall_sensor_health"
        ],
        "overall_health_status": current[
            "overall_health_status"
        ],
        "subsystem_health": current[
            "subsystem_health"
        ],
        "history": history,
    }