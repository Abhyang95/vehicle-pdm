from pathlib import Path

import pandas as pd


# ============================================================
# VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM
# MAINTENANCE RECOMMENDATION ENGINE
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DIAGNOSTICS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "diagnostics"
)


# ============================================================
# Subsystem recommendation rules
# ============================================================

SUBSYSTEM_RULES = {

    "temperature": {
        "name": "Temperature",
        "inspection": (
            "Inspect temperature-related telemetry "
            "and investigate abnormal thermal trends."
        ),
        "maintenance": (
            "Perform thermal-system diagnostic inspection."
        ),
    },

    "pressure": {
        "name": "Pressure",
        "inspection": (
            "Inspect pressure-related telemetry "
            "and investigate abnormal pressure trends."
        ),
        "maintenance": (
            "Perform pressure-system diagnostic inspection."
        ),
    },

    "speed_flow": {
        "name": "Speed / Flow",
        "inspection": (
            "Inspect speed and flow telemetry "
            "for abnormal variation or degradation."
        ),
        "maintenance": (
            "Perform speed/flow-system diagnostic inspection."
        ),
    },

    "thermal_bleed": {
        "name": "Thermal / Bleed",
        "inspection": (
            "Inspect thermal/bleed-related telemetry "
            "and investigate abnormal trends."
        ),
        "maintenance": (
            "Perform thermal/bleed-system diagnostic inspection."
        ),
    },
}


# ============================================================
# Determine overall priority
# ============================================================

def determine_priority(
    severity,
    subsystem_statuses,
):

    critical_count = sum(
        status == "CRITICAL"
        for status in subsystem_statuses.values()
    )

    degraded_count = sum(
        status == "DEGRADED"
        for status in subsystem_statuses.values()
    )

    if severity == "CRITICAL" or critical_count >= 2:
        return "IMMEDIATE"

    if severity == "HIGH" or critical_count == 1:
        return "URGENT"

    if severity == "MEDIUM" or degraded_count >= 1:
        return "PLANNED"

    return "ROUTINE"


# ============================================================
# Build maintenance recommendation
# ============================================================

def build_recommendation(
    severity,
    predicted_rul,
    subsystem_health,
    subsystem_statuses,
    shap_features,
):

    priority = determine_priority(
        severity,
        subsystem_statuses,
    )

    # --------------------------------------------------------
    # Identify degraded / critical subsystems
    # --------------------------------------------------------

    focus_subsystems = []

    for key, status in subsystem_statuses.items():

        if status in {
            "CRITICAL",
            "DEGRADED",
        }:

            focus_subsystems.append(
                {
                    "subsystem": SUBSYSTEM_RULES[key]["name"],
                    "status": status,
                    "health": subsystem_health[key],
                    "inspection": SUBSYSTEM_RULES[key][
                        "inspection"
                    ],
                    "maintenance": SUBSYSTEM_RULES[key][
                        "maintenance"
                    ],
                }
            )

    # --------------------------------------------------------
    # Overall action
    # --------------------------------------------------------

    if priority == "IMMEDIATE":

        action = (
            "Schedule immediate diagnostic inspection "
            "and assess whether continued operation "
            "is appropriate."
        )

    elif priority == "URGENT":

        action = (
            "Schedule maintenance inspection at the "
            "earliest practical opportunity."
        )

    elif priority == "PLANNED":

        action = (
            "Plan preventive maintenance and continue "
            "enhanced telemetry monitoring."
        )

    else:

        action = (
            "Continue routine monitoring and scheduled "
            "maintenance."
        )

    # --------------------------------------------------------
    # RUL context
    # --------------------------------------------------------

    if predicted_rul <= 7:

        rul_message = (
            "Predicted RUL is below 7 cycles, indicating "
            "very limited remaining useful life."
        )

    elif predicted_rul <= 15:

        rul_message = (
            "Predicted RUL is below 15 cycles, indicating "
            "limited remaining useful life."
        )

    elif predicted_rul <= 30:

        rul_message = (
            "Predicted RUL is below 30 cycles; preventive "
            "maintenance should be planned."
        )

    else:

        rul_message = (
            "Predicted RUL is above 30 cycles; continue "
            "monitoring for emerging degradation."
        )

    # --------------------------------------------------------
    # SHAP focus features
    # --------------------------------------------------------

    shap_focus = []

    for feature in shap_features[:5]:

        shap_focus.append(
            {
                "feature": feature["feature"],
                "shap_value": feature["shap_value"],
                "direction": (
                    "INCREASES_RISK"
                    if feature["shap_value"] > 0
                    else "DECREASES_RISK"
                ),
            }
        )

    # --------------------------------------------------------
    # Build structured result
    # --------------------------------------------------------

    return {
        "priority": priority,
        "action": action,
        "rul_context": rul_message,
        "focus_subsystems": focus_subsystems,
        "shap_focus": shap_focus,
    }


# ============================================================
# Main demonstration
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print(
        "VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM"
    )
    print("MAINTENANCE RECOMMENDATION ENGINE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load integrated diagnostic result
    # --------------------------------------------------------

    report_path = (
        DIAGNOSTICS_DIR
        / "integrated_diagnostic_report.csv"
    )

    report = pd.read_csv(
        report_path
    ).iloc[0]

    # --------------------------------------------------------
    # Load SHAP explanation
    # --------------------------------------------------------

    shap_path = (
        DIAGNOSTICS_DIR
        / "integrated_top_shap_contributors.csv"
    )

    shap_df = pd.read_csv(
        shap_path
    )

    shap_features = (
        shap_df[
            [
                "feature",
                "shap_value",
            ]
        ]
        .to_dict("records")
    )

    # --------------------------------------------------------
    # Subsystem data
    # --------------------------------------------------------

    subsystem_health = {

        "temperature": float(
            report["temperature_health"]
        ),

        "pressure": float(
            report["pressure_health"]
        ),

        "speed_flow": float(
            report["speed_flow_health"]
        ),

        "thermal_bleed": float(
            report["thermal_bleed_health"]
        ),
    }

    subsystem_statuses = {

        "temperature": None,
        "pressure": None,
        "speed_flow": None,
        "thermal_bleed": None,
    }

    # Get exact status for the vehicle/cycle
    subsystem_df = pd.read_csv(
        DIAGNOSTICS_DIR
        / "subsystem_health_results.csv"
    )

    matching = subsystem_df[
        (
            subsystem_df["unit"]
            == int(report["vehicle_id"])
        )
        &
        (
            subsystem_df["cycle"]
            == int(report["cycle"])
        )
    ]

    if matching.empty:

        raise ValueError(
            "Could not locate exact subsystem-health "
            "observation."
        )

    row = matching.iloc[0]

    subsystem_statuses = {

        "temperature": row[
            "temperature_status"
        ],

        "pressure": row[
            "pressure_status"
        ],

        "speed_flow": row[
            "speed_flow_status"
        ],

        "thermal_bleed": row[
            "thermal_bleed_status"
        ],
    }

    # --------------------------------------------------------
    # Generate recommendation
    # --------------------------------------------------------

    recommendation = build_recommendation(
        severity=report["severity"],
        predicted_rul=float(
            report["predicted_rul"]
        ),
        subsystem_health=subsystem_health,
        subsystem_statuses=subsystem_statuses,
        shap_features=shap_features,
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print("MAINTENANCE PLAN")

    print("=" * 70)

    print(
        f"\nVehicle ID : "
        f"{int(report['vehicle_id'])}"
    )

    print(
        f"Cycle      : "
        f"{int(report['cycle'])}"
    )

    print(
        f"Severity   : "
        f"{report['severity']}"
    )

    print(
        f"Predicted RUL : "
        f"{float(report['predicted_rul']):.2f} cycles"
    )

    print(
        f"\nPriority   : "
        f"{recommendation['priority']}"
    )

    print(
        "\nRecommended action:"
    )

    print(
        recommendation["action"]
    )

    print(
        "\nRUL assessment:"
    )

    print(
        recommendation["rul_context"]
    )

    # --------------------------------------------------------
    # Focus subsystems
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print("FOCUS SUBSYSTEMS")

    print("-" * 70)

    if recommendation["focus_subsystems"]:

        for item in recommendation[
            "focus_subsystems"
        ]:

            print(
                f"\n{item['subsystem']}"
            )

            print(
                f"  Health     : "
                f"{item['health']:.2f} / 100"
            )

            print(
                f"  Status     : "
                f"{item['status']}"
            )

            print(
                f"  Inspection : "
                f"{item['inspection']}"
            )

            print(
                f"  Maintenance: "
                f"{item['maintenance']}"
            )

    else:

        print(
            "\nNo degraded or critical subsystem "
            "identified."
        )

    # --------------------------------------------------------
    # SHAP focus
    # --------------------------------------------------------

    print(
        "\n" + "-" * 70
    )

    print("MODEL-IDENTIFIED RISK DRIVERS")

    print("-" * 70)

    for item in recommendation[
        "shap_focus"
    ]:

        print(
            f"{item['feature']:<35} "
            f"{item['shap_value']:>9.4f} "
            f"{item['direction']}"
        )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    maintenance_rows = []

    for item in recommendation[
        "focus_subsystems"
    ]:

        maintenance_rows.append({
            "vehicle_id": int(
                report["vehicle_id"]
            ),
            "cycle": int(
                report["cycle"]
            ),
            "severity": report["severity"],
            "priority": recommendation[
                "priority"
            ],
            "subsystem": item[
                "subsystem"
            ],
            "health": item[
                "health"
            ],
            "status": item[
                "status"
            ],
            "inspection": item[
                "inspection"
            ],
            "maintenance": item[
                "maintenance"
            ],
        })

    if maintenance_rows:

        output_df = pd.DataFrame(
            maintenance_rows
        )

    else:

        output_df = pd.DataFrame([
            {
                "vehicle_id": int(
                    report["vehicle_id"]
                ),
                "cycle": int(
                    report["cycle"]
                ),
                "severity": report["severity"],
                "priority": recommendation[
                    "priority"
                ],
                "subsystem": "None",
                "health": float(
                    report["overall_sensor_health"]
                ),
                "status": report[
                    "overall_health_status"
                ],
                "inspection": (
                    "Continue routine monitoring."
                ),
                "maintenance": (
                    "No immediate intervention indicated."
                ),
            }
        ])

    output_path = (
        DIAGNOSTICS_DIR
        / "maintenance_recommendations.csv"
    )

    output_df.to_csv(
        output_path,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "MAINTENANCE ENGINE COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"\nSaved:"
    )

    print(output_path)