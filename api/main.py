from io import BytesIO
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.diagnostic_report import build_diagnostic_report
from src.inference import predict_from_telemetry
from src.maintenance import build_recommendation
from src.vehicle_health import build_vehicle_health_history
from src.vehicle_health import load_vehicle_history
from src.assistant import answer_vehicle_question
from src.inference import (
    FAILURE_MODEL_PATH,
    RUL_MODEL_PATH,
)


# ============================================================
# Project configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"


# ============================================================
# CSV upload configuration
# ============================================================

REQUIRED_TELEMETRY_COLUMNS = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
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


# ============================================================
# Replay session state
# ============================================================
#
# Uploaded telemetry is stored in memory.
#
# This does NOT modify the original NASA C-MAPSS data.
#
# Future production deployment:
# Redis / PostgreSQL / object storage could replace this.
# ============================================================

REPLAY_SESSIONS = {}


# ============================================================
# FD001 benchmark replay ground truth
# ============================================================

FD001_RUL_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "cmapss"
    / "RUL_FD001.txt"
)

FD001_REPLAY_FILENAME = "fd001_test_replay.csv"


def load_fd001_test_rul():
    """
    Load official NASA C-MAPSS FD001 test-set RUL labels.

    These labels describe the remaining useful life after the
    final observed test cycle for each engine.

    They are used only for the recognized FD001 benchmark
    replay dataset. Arbitrary uploaded CSV files do not receive
    benchmark ground truth.
    """

    if not FD001_RUL_PATH.exists():
        return {}

    values = pd.read_csv(
        FD001_RUL_PATH,
        sep=r"\s+",
        header=None,
    ).iloc[:, 0]

    return {
        unit: float(rul)
        for unit, rul in enumerate(
            values.tolist(),
            start=1,
        )
    }


FD001_TEST_RUL = load_fd001_test_rul()



# ============================================================
# FastAPI application
# ============================================================

app = FastAPI(
    title="Vehicle Predictive Maintenance & Diagnostics Platform",
    description=(
        "ML-powered predictive maintenance API "
        "using NASA C-MAPSS benchmark data."
    ),
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "https://vehicle-pdm-indol.vercel.app",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Request schemas
# ============================================================

class TelemetryPoint(BaseModel):

    unit: int = Field(
        ...,
        ge=1,
        description="Vehicle/unit identifier.",
    )

    cycle: int = Field(
        ...,
        ge=1,
        description="Telemetry cycle.",
    )

    setting_1: float
    setting_2: float
    setting_3: float

    sensor_2: float
    sensor_3: float
    sensor_4: float
    sensor_7: float
    sensor_8: float
    sensor_9: float
    sensor_11: float
    sensor_12: float
    sensor_13: float
    sensor_14: float
    sensor_15: float
    sensor_17: float
    sensor_20: float
    sensor_21: float


class PredictionRequest(BaseModel):

    telemetry: List[TelemetryPoint] = Field(
        ...,
        min_length=1,
        description=(
            "Chronological telemetry history. "
            "At least one observation is required."
        ),
    )


class AssistantRequest(BaseModel):

    unit: int = Field(
        ...,
        ge=1,
        description="Vehicle/unit identifier.",
    )

    question: str = Field(
        ...,
        min_length=1,
        max_length=2000,
        description="Vehicle maintenance question.",
    )

    # --------------------------------------------------------
    # Optional replay context
    # --------------------------------------------------------

    session_id: str | None = Field(
        default=None,
        description=(
            "Uploaded replay session identifier. "
            "When provided, the assistant uses telemetry "
            "from this replay session instead of the built-in "
            "vehicle history."
        ),
    )

    cycle: int | None = Field(
        default=None,
        ge=1,
        description=(
            "Selected telemetry cycle. "
            "For replay mode, the assistant analyzes telemetry "
            "up to this cycle."
        ),
    )


# ============================================================
# Helper functions
# ============================================================

def dataframe_from_request(
    request: PredictionRequest,
) -> pd.DataFrame:
    """
    Convert a validated PredictionRequest into a DataFrame.
    """

    return pd.DataFrame(
        [
            point.model_dump()
            for point in request.telemetry
        ]
    )


def get_risk_status(failure_probability: float) -> str:
    """
    Convert predicted failure probability into an operational risk state.

    The 0.30 decision threshold was selected from the official
    100-engine FD001 evaluation because it improved the balance between
    missed failures and false alarms:

        Precision: 0.8846
        Recall:    0.9200
        F1:        0.9020

    The project prioritizes detecting potential failures because a missed
    failure is more costly than an additional maintenance warning.
    """

    if failure_probability >= 0.30:
        return "HIGH_RISK"

    if failure_probability >= 0.15:
        return "WATCH"

    return "LOW_RISK"


def get_severity(
    failure_probability: float,
    predicted_rul: float,
) -> str:
    """
    Convert model outputs into operational severity.
    """

    if (
        failure_probability >= 0.90
        or predicted_rul <= 7
    ):
        return "CRITICAL"

    if (
        failure_probability >= 0.65
        or predicted_rul <= 15
    ):
        return "HIGH"

    if (
        failure_probability >= 0.35
        or predicted_rul <= 30
    ):
        return "MEDIUM"

    return "LOW"


def build_maintenance_from_report(
    report: dict,
) -> dict:
    """
    Build a maintenance recommendation from an integrated
    diagnostic report.

    The diagnostic report exposes SHAP fields as:
        impact
        absolute_impact

    The maintenance engine expects:
        shap_value
        absolute_shap

    This helper performs that schema adaptation centrally.
    """

    subsystem_health = {
        name: details["health"]
        for name, details
        in report["subsystem_health"].items()
    }

    subsystem_statuses = {
        name: details["status"]
        for name, details
        in report["subsystem_health"].items()
    }

    shap_features = [
        {
            "feature": item["feature"],
            "shap_value": item["impact"],
            "absolute_shap": item["absolute_impact"],
            "direction": item["direction"],
        }
        for item in report["shap_contributors"]
    ]

    return build_recommendation(
        severity=report["severity"],
        predicted_rul=report["predicted_rul"],
        subsystem_health=subsystem_health,
        subsystem_statuses=subsystem_statuses,
        shap_features=shap_features,
    )


def get_replay_context(
    session_id: str,
    unit: int,
    cycle: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load telemetry from an uploaded replay session.

    Returns:
        telemetry dataframe
        metadata describing the active data source
    """

    if session_id not in REPLAY_SESSIONS:
        raise HTTPException(
            status_code=404,
            detail="Replay session was not found.",
        )

    session = REPLAY_SESSIONS[session_id]

    telemetry_df = session["telemetry"]

    unit_df = (
        telemetry_df[
            telemetry_df["unit"] == unit
        ]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    if unit_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unit {unit} was not found "
                "in replay data."
            ),
        )

    min_cycle = int(unit_df["cycle"].min())
    max_cycle = int(unit_df["cycle"].max())

    selected_cycle = max_cycle if cycle is None else cycle

    if (
        selected_cycle < min_cycle
        or selected_cycle > max_cycle
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Requested cycle is outside "
                    "the available replay range."
                ),
                "requested_cycle": selected_cycle,
                "available_cycle_range": {
                    "min": min_cycle,
                    "max": max_cycle,
                },
            },
        )

    active_df = (
        unit_df[
            unit_df["cycle"] <= selected_cycle
        ]
        .copy()
        .reset_index(drop=True)
    )

    if active_df.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "No telemetry is available "
                "for the requested replay cycle."
            ),
        )

    metadata = {
        "mode": "replay",
        "label": "UPLOADED REPLAY",
        "filename": session["filename"],
        "session_id": session_id,
        "unit": unit,
        "current_cycle": selected_cycle,
        "available_cycle_range": {
            "min": min_cycle,
            "max": max_cycle,
        },
        "row_count": int(len(active_df)),
    }

    return active_df, metadata



def build_replay_rul_history(
    telemetry_df: pd.DataFrame,
    unit: int,
    filename: str,
    end_cycle: int | None = None,
) -> tuple[list[dict], bool, str | None]:
    """
    Build cycle-by-cycle RUL predictions for telemetry replay.

    Prediction at cycle t only uses telemetry available from
    cycle 1 through cycle t.

    For recognized FD001 benchmark replay, true RUL is calculated
    using the COMPLETE observed test trajectory:

        true_rul(t) =
            final_observed_cycle - t + official_final_rul

    The true RUL is capped at 125 cycles to match the project's
    RUL-labeling convention.

    Important:
        telemetry_df must contain the complete unit trajectory
        when benchmark ground truth is requested.

    end_cycle controls how far the replay history is exposed.
    """

    normalized_filename = Path(filename).name.lower()

    ground_truth_available = (
        normalized_filename == FD001_REPLAY_FILENAME
        and unit in FD001_TEST_RUL
    )

    ground_truth_source = (
        "NASA C-MAPSS FD001 RUL_FD001.txt"
        if ground_truth_available
        else None
    )

    unit_df = (
        telemetry_df[
            telemetry_df["unit"] == unit
        ]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    if unit_df.empty:
        return (
            [],
            ground_truth_available,
            ground_truth_source,
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Determine final observed cycle from the COMPLETE
    # trajectory, not from the truncated replay prefix.
    # --------------------------------------------------------

    final_observed_cycle = int(
        unit_df["cycle"].max()
    )

    official_final_rul = (
        FD001_TEST_RUL.get(unit)
        if ground_truth_available
        else None
    )

    # --------------------------------------------------------
    # Limit the visible prediction history to the selected
    # replay cycle.
    # --------------------------------------------------------


    if end_cycle is not None:
        history_df = unit_df[
            unit_df["cycle"] <= end_cycle
        ].copy()
    else:
        history_df = unit_df.copy()

    history_df = (
        history_df
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    # Limit historical predictions to at most 20 points.
    # Always include the first and latest available cycles.
    MAX_HISTORY_POINTS = 20

    all_cycles = (
        history_df["cycle"]
        .astype(int)
        .tolist()
    )

    if len(all_cycles) > MAX_HISTORY_POINTS:
        positions = sorted({
            round(
                i * (len(all_cycles) - 1)
                / (MAX_HISTORY_POINTS - 1)
            )
            for i in range(MAX_HISTORY_POINTS)
        })

        selected_cycles = [
            all_cycles[position]
            for position in positions
        ]
    else:
        selected_cycles = all_cycles

    history = []


    for cycle in selected_cycles:

        # ----------------------------------------------------
        # Causal prediction:
        # only telemetry available up to this cycle.
        # ----------------------------------------------------

        truncated_df = (
            unit_df[
                unit_df["cycle"] <= cycle
            ]
            .copy()
            .reset_index(drop=True)
        )


        # Causal prediction, health, and risk:
        # use only telemetry available through this cycle.
        historical_report = build_diagnostic_report(
            truncated_df
        )

        overall_health = float(
            historical_report["overall_sensor_health"]
        )

        failure_risk = float(
            historical_report["failure_probability"]
        )

        predicted_rul = float(
            historical_report["predicted_rul"]
        )


        # ----------------------------------------------------
        # Official FD001 true RUL:
        #
        # final observed cycle
        #        -
        # current cycle
        #        +
        # official final RUL
        # ----------------------------------------------------

        true_rul = None

        if ground_truth_available:
            true_rul = float(
                min(
                    125.0,
                    max(
                        0.0,
                        final_observed_cycle
                        - cycle
                        + official_final_rul,
                    ),
                )
            )


        history.append(
        {
            "cycle": int(cycle),
            "predicted_rul": round(
                predicted_rul,
                2,
            ),
            "true_rul": (
                round(true_rul, 2)
                if true_rul is not None
                else None
            ),
            "overall_health": round(
                overall_health,
                2,
            ),
            "failure_risk": round(
                failure_risk,
                4,
            ),
        }
    )

    return (
        history,
        ground_truth_available,
        ground_truth_source,
    )


def get_default_vehicle_context(
    unit: int,
    cycle: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Load the built-in/demo vehicle telemetry.

    This is intentionally separate from replay mode so the API
    can clearly identify where the diagnostic state originated.
    """

    vehicle_df = load_vehicle_history(unit)

    if vehicle_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No telemetry found for vehicle/unit {unit}."
            ),
        )

    vehicle_df = (
        vehicle_df
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    min_cycle = int(vehicle_df["cycle"].min())
    max_cycle = int(vehicle_df["cycle"].max())

    selected_cycle = max_cycle if cycle is None else cycle

    if (
        selected_cycle < min_cycle
        or selected_cycle > max_cycle
    ):
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Requested cycle is outside "
                    "the available vehicle history."
                ),
                "requested_cycle": selected_cycle,
                "available_cycle_range": {
                    "min": min_cycle,
                    "max": max_cycle,
                },
            },
        )

    active_df = (
        vehicle_df[
            vehicle_df["cycle"] <= selected_cycle
        ]
        .copy()
        .reset_index(drop=True)
    )

    metadata = {
        "mode": "demo",
        "label": "DEMO TELEMETRY",
        "filename": "vehicle_unit_1.csv"
        if unit == 1
        else None,
        "session_id": None,
        "unit": unit,
        "current_cycle": selected_cycle,
        "available_cycle_range": {
            "min": min_cycle,
            "max": max_cycle,
        },
        "row_count": int(len(active_df)),
    }

    return active_df, metadata


def get_active_vehicle_context(
    unit: int,
    session_id: str | None = None,
    cycle: int | None = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Resolve the exact telemetry source used by the assistant.

    Priority:

    1. Uploaded replay session, when session_id is provided.
    2. Built-in/demo vehicle history otherwise.
    """

    if session_id:
        return get_replay_context(
            session_id=session_id,
            unit=unit,
            cycle=cycle,
        )

    return get_default_vehicle_context(
        unit=unit,
        cycle=cycle,
    )


# ============================================================
# Health
# ============================================================


@app.get("/health")
def health():
    failure_model_exists = FAILURE_MODEL_PATH.exists()
    rul_model_exists = RUL_MODEL_PATH.exists()

    return {
        "status": (
            "healthy"
            if failure_model_exists and rul_model_exists
            else "unhealthy"
        ),
        "failure_risk_model": failure_model_exists,
        "rul_model": rul_model_exists,
    }



# ============================================================
# Model information
# ============================================================

@app.get("/model-info")
def model_info():

    return {
        "project": (
            "Vehicle Predictive Maintenance "
            "& Diagnostics Platform"
        ),
        "dataset": "NASA C-MAPSS FD001",
        "failure_risk_model": "XGBoost",
        "rul_model": "XGBoost",
        "failure_threshold_cycles": 30,
        "rul_cap_cycles": 125,
        "feature_count": 143,
        "explainability": "SHAP",
        "diagnostics": (
            "Approximate telemetry-derived "
            "subsystem health"
        ),
        "replay_mode": True,
        "csv_upload": True,
        "default_data_source": "DEMO TELEMETRY",
    }


# ============================================================
# CSV telemetry upload
# ============================================================

@app.post("/upload-csv")
async def upload_csv(
    file: UploadFile = File(...),
):
    """
    Upload vehicle telemetry CSV for replay/inference.

    The uploaded file is validated and stored in an in-memory
    replay session. It does not modify the original NASA
    C-MAPSS benchmark dataset.
    """

    try:

        if not file.filename:
            raise HTTPException(
                status_code=400,
                detail="No file was provided.",
            )

        if not file.filename.lower().endswith(".csv"):
            raise HTTPException(
                status_code=400,
                detail="Only CSV files are supported.",
            )

        contents = await file.read()

        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Uploaded CSV file is empty.",
            )

        telemetry_df = pd.read_csv(
            BytesIO(contents)
        )

        missing_columns = [
            column
            for column in REQUIRED_TELEMETRY_COLUMNS
            if column not in telemetry_df.columns
        ]

        if missing_columns:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "CSV is missing required "
                        "telemetry columns."
                    ),
                    "missing_columns": missing_columns,
                    "required_columns": (
                        REQUIRED_TELEMETRY_COLUMNS
                    ),
                },
            )

        if telemetry_df.empty:
            raise HTTPException(
                status_code=422,
                detail="CSV contains no telemetry rows.",
            )

        telemetry_df = telemetry_df[
            REQUIRED_TELEMETRY_COLUMNS
        ].copy()

        # ----------------------------------------------------
        # Convert required telemetry columns to numeric.
        # ----------------------------------------------------

        for column in REQUIRED_TELEMETRY_COLUMNS:
            telemetry_df[column] = pd.to_numeric(
                telemetry_df[column],
                errors="coerce",
            )

        # ----------------------------------------------------
        # Reject missing, non-numeric, NaN, and infinite values.
        # ----------------------------------------------------

        numeric_values = telemetry_df[
            REQUIRED_TELEMETRY_COLUMNS
        ].to_numpy(dtype=float)

        invalid_mask = (
            telemetry_df.isna().any(axis=1)
            | ~np.isfinite(numeric_values).all(axis=1)
        )

        invalid_rows = telemetry_df[invalid_mask]

        if not invalid_rows.empty:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "CSV contains missing, non-numeric, "
                        "or non-finite telemetry values."
                    ),
                    "invalid_row_count": int(
                        len(invalid_rows)
                    ),
                },
            )

        # ----------------------------------------------------
        # Unit identifiers and cycle numbers must be positive
        # integers.
        # ----------------------------------------------------

        for column, label in (
            ("unit", "Unit identifiers"),
            ("cycle", "Cycle values"),
        ):
            values = telemetry_df[column]

            if (values < 1).any():
                raise HTTPException(
                    status_code=422,
                    detail=f"{label} must be >= 1.",
                )

            if (values % 1 != 0).any():
                raise HTTPException(
                    status_code=422,
                    detail=f"{label} must be integers.",
                )

        # ----------------------------------------------------
        # Sort telemetry chronologically by unit and cycle.
        # ----------------------------------------------------

        telemetry_df = (
            telemetry_df
            .sort_values(
                ["unit", "cycle"]
            )
            .reset_index(drop=True)
        )

        # ----------------------------------------------------
        # Reject duplicate unit/cycle observations.
        # ----------------------------------------------------

        duplicate_mask = telemetry_df.duplicated(
            subset=["unit", "cycle"],
            keep=False,
        )

        duplicate_count = int(
            duplicate_mask.sum()
        )

        if duplicate_count > 0:
            raise HTTPException(
                status_code=422,
                detail={
                    "message": (
                        "CSV contains duplicate "
                        "unit/cycle observations."
                    ),
                    "duplicate_rows": duplicate_count,
                },
            )

        # ----------------------------------------------------
        # Build per-unit metadata.
        # ----------------------------------------------------

        unit_metadata = []

        for unit, group in telemetry_df.groupby(
            "unit"
        ):
            unit_metadata.append(
                {
                    "unit": int(unit),
                    "row_count": int(len(group)),
                    "start_cycle": int(
                        group["cycle"].min()
                    ),
                    "end_cycle": int(
                        group["cycle"].max()
                    ),
                }
            )

        # ----------------------------------------------------
        # Create the in-memory replay session.
        # ----------------------------------------------------

        session_id = (
            f"replay_{file.filename}_"
            f"{len(REPLAY_SESSIONS) + 1}"
        )

        REPLAY_SESSIONS[session_id] = {
            "filename": file.filename,
            "telemetry": telemetry_df.copy(),
        }

        return {
            "status": "accepted",
            "session_id": session_id,
            "filename": file.filename,
            "row_count": int(
                len(telemetry_df)
            ),
            "column_count": int(
                len(telemetry_df.columns)
            ),
            "units": [
                int(unit)
                for unit in telemetry_df[
                    "unit"
                ].unique()
            ],
            "unit_count": int(
                telemetry_df["unit"].nunique()
            ),
            "cycle_range": {
                "min": int(
                    telemetry_df["cycle"].min()
                ),
                "max": int(
                    telemetry_df["cycle"].max()
                ),
            },
            "unit_metadata": unit_metadata,
            "telemetry_columns": (
                REQUIRED_TELEMETRY_COLUMNS
            ),
            "data_source": {
                "mode": "replay",
                "label": "UPLOADED REPLAY",
                "filename": file.filename,
            },
        }

    except HTTPException:
        raise

    except pd.errors.EmptyDataError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                "The uploaded file is not a valid CSV."
            ),
        ) from exc

    except pd.errors.ParserError as exc:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unable to parse CSV: {str(exc)}"
            ),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"CSV upload failed: {str(exc)}"
            ),
        ) from exc



# ============================================================
# Replay session information
# ============================================================

@app.get("/replay/{session_id}")
def replay_session_info(
    session_id: str,
):

    if session_id not in REPLAY_SESSIONS:
        raise HTTPException(
            status_code=404,
            detail="Replay session was not found.",
        )

    session = REPLAY_SESSIONS[session_id]
    telemetry_df = session["telemetry"]

    unit_metadata = []

    for unit, group in telemetry_df.groupby(
        "unit"
    ):

        unit_metadata.append(
            {
                "unit": int(unit),
                "row_count": int(len(group)),
                "start_cycle": int(
                    group["cycle"].min()
                ),
                "end_cycle": int(
                    group["cycle"].max()
                ),
            }
        )

    return {
        "session_id": session_id,
        "filename": session["filename"],
        "row_count": int(len(telemetry_df)),
        "unit_count": int(
            telemetry_df["unit"].nunique()
        ),
        "units": [
            int(unit)
            for unit in telemetry_df[
                "unit"
            ].unique()
        ],
        "unit_metadata": unit_metadata,
        "data_source": {
            "mode": "replay",
            "label": "UPLOADED REPLAY",
            "filename": session["filename"],
        },
    }


# ============================================================
# Replay analysis
# ============================================================


@app.get("/replay/{session_id}/analyze")
def replay_analyze(
    session_id: str,
    unit: int,
    cycle: int,
):

    if session_id not in REPLAY_SESSIONS:
        raise HTTPException(
            status_code=404,
            detail="Replay session was not found.",
        )

    session = REPLAY_SESSIONS[session_id]
    telemetry_df = session["telemetry"]

    unit_df = (
        telemetry_df[
            telemetry_df["unit"] == unit
        ]
        .sort_values("cycle")
        .reset_index(drop=True)
    )

    if unit_df.empty:
        raise HTTPException(
            status_code=404,
            detail=(
                f"Unit {unit} was not found "
                "in replay data."
            ),
        )

    min_cycle = int(
        unit_df["cycle"].min()
    )

    max_cycle = int(
        unit_df["cycle"].max()
    )

    if cycle < min_cycle or cycle > max_cycle:
        raise HTTPException(
            status_code=422,
            detail={
                "message": (
                    "Requested cycle is outside "
                    "the available replay range."
                ),
                "requested_cycle": cycle,
                "available_cycle_range": {
                    "min": min_cycle,
                    "max": max_cycle,
                },
            },
        )

    replay_df = (
        unit_df[
            unit_df["cycle"] <= cycle
        ]
        .copy()
        .reset_index(drop=True)
    )

    if replay_df.empty:
        raise HTTPException(
            status_code=422,
            detail=(
                "No telemetry is available "
                "for the requested cycle."
            ),
        )

    try:

        # ----------------------------------------------------
        # Main diagnostic analysis
        # ----------------------------------------------------

        report = build_diagnostic_report(
            replay_df
        )

        maintenance = build_maintenance_from_report(
            report
        )

        # ----------------------------------------------------
        # Cycle-by-cycle RUL validation history
        #
        # Each prediction only uses telemetry available up
        # to that cycle. No future rows are exposed to the
        # prediction for an earlier cycle.
        # ----------------------------------------------------


        (
            rul_history,
            ground_truth_available,
            ground_truth_source,
        ) = build_replay_rul_history(
            telemetry_df=unit_df,
            unit=unit,
            filename=session["filename"],
            end_cycle=cycle,
        )


        return {
            "status": "success",
            "session_id": session_id,
            "filename": session["filename"],
            "unit": unit,
            "current_cycle": cycle,
            "replay_rows": int(
                len(replay_df)
            ),
            "available_cycle_range": {
                "min": min_cycle,
                "max": max_cycle,
            },

            "data_source": {
                "mode": "replay",
                "label": "UPLOADED REPLAY",
                "filename": session["filename"],
                "session_id": session_id,
            },

            # ------------------------------------------------
            # Replay ground-truth metadata
            # ------------------------------------------------

            "ground_truth_available": (
                ground_truth_available
            ),
            "ground_truth_source": (
                ground_truth_source
            ),

            # ------------------------------------------------
            # Current-cycle prediction
            # ------------------------------------------------

            "prediction": {
                "failure_probability": round(
                    report[
                        "failure_probability"
                    ],
                    4,
                ),
                "predicted_rul": round(
                    report[
                        "predicted_rul"
                    ],
                    2,
                ),
                "risk_status": report[
                    "risk_status"
                ],
                "severity": report[
                    "severity"
                ],
            },

            # ------------------------------------------------
            # RUL validation history
            # ------------------------------------------------

            "rul_history": rul_history,

            # ------------------------------------------------
            # Vehicle health
            # ------------------------------------------------

            "overall_health": {
                "score": round(
                    report[
                        "overall_sensor_health"
                    ],
                    2,
                ),
                "status": report[
                    "overall_health_status"
                ],
            },

            # ------------------------------------------------
            # Diagnostics
            # ------------------------------------------------

            "subsystem_health": report[
                "subsystem_health"
            ],

            "shap_contributors": report[
                "shap_contributors"
            ],

            # ------------------------------------------------
            # Maintenance
            # ------------------------------------------------

            "maintenance": maintenance,

            "feature_count": report[
                "feature_count"
            ],
        }

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Replay analysis failed: {str(exc)}"
            ),
        ) from exc



# ============================================================
# Prediction
# ============================================================

@app.post("/predict")
def predict(
    request: PredictionRequest,
):

    try:

        telemetry_df = dataframe_from_request(
            request
        )

        result = predict_from_telemetry(
            telemetry_df
        )

        failure_probability = (
            result["failure_probability"]
        )

        predicted_rul = result["predicted_rul"]

        risk_status = get_risk_status(
            failure_probability
        )

        severity = get_severity(
            failure_probability,
            predicted_rul,
        )

        return {
            "unit": int(
                telemetry_df["unit"].iloc[-1]
            ),
            "current_cycle": result[
                "current_cycle"
            ],
            "failure_probability": round(
                failure_probability,
                4,
            ),
            "predicted_rul": round(
                predicted_rul,
                2,
            ),
            "risk_status": risk_status,
            "severity": severity,
            "feature_count": result[
                "feature_count"
            ],
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=400,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Prediction failed: "
                f"{str(exc)}"
            ),
        ) from exc


# ============================================================
# Diagnostics
# ============================================================

@app.post("/diagnostics")
def diagnostics(
    request: PredictionRequest,
):

    try:

        telemetry_df = dataframe_from_request(
            request
        )

        report = build_diagnostic_report(
            telemetry_df
        )

        return report

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Diagnostics failed: "
                f"{str(exc)}"
            ),
        ) from exc


# ============================================================
# Maintenance recommendations
# ============================================================

@app.post("/maintenance")
def maintenance(
    request: PredictionRequest,
):

    try:

        telemetry_df = dataframe_from_request(
            request
        )

        report = build_diagnostic_report(
            telemetry_df
        )

        recommendation = (
            build_maintenance_from_report(
                report
            )
        )

        return {
            "unit": report["unit"],
            "current_cycle": report[
                "current_cycle"
            ],
            "failure_probability": report[
                "failure_probability"
            ],
            "predicted_rul": report[
                "predicted_rul"
            ],
            "risk_status": report[
                "risk_status"
            ],
            "severity": report[
                "severity"
            ],
            "overall_sensor_health": report[
                "overall_sensor_health"
            ],
            "overall_health_status": report[
                "overall_health_status"
            ],
            "subsystem_health": report[
                "subsystem_health"
            ],
            "maintenance": recommendation,
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Maintenance recommendation failed: "
                f"{str(exc)}"
            ),
        ) from exc


# ============================================================
# AI Vehicle Assistant
# ============================================================

@app.post("/assistant")
def assistant(
    request: AssistantRequest,
):
    """
    Answer a vehicle-maintenance question using:

    1. Active vehicle telemetry
    2. ML diagnostic report
    3. RAG knowledge
    4. Gemini grounded explanation

    IMPORTANT:

    If session_id is supplied, the assistant uses the uploaded
    replay telemetry.

    Otherwise, it uses the built-in/demo vehicle telemetry.

    The LLM explains the diagnostic state. It does not calculate
    the underlying ML predictions itself.
    """

    try:

        # ----------------------------------------------------
        # Resolve the exact active telemetry source
        # ----------------------------------------------------

        vehicle_df, data_source = (
            get_active_vehicle_context(
                unit=request.unit,
                session_id=request.session_id,
                cycle=request.cycle,
            )
        )

        # ----------------------------------------------------
        # Build current diagnostic state
        # ----------------------------------------------------

        report = build_diagnostic_report(
            vehicle_df
        )

        # ----------------------------------------------------
        # Ask grounded AI assistant
        # ----------------------------------------------------

        result = answer_vehicle_question(
            question=request.question,
            diagnostic_state=report,
            top_k=4,
        )

        # ----------------------------------------------------
        # Return assistant response
        # ----------------------------------------------------

        return {
            "status": "success",
            "unit": int(
                report["unit"]
            ),
            "current_cycle": int(
                report["current_cycle"]
            ),
            "question": result["question"],
            "answer": result["answer"],
            "diagnostic_state": result[
                "diagnostic_state"
            ],
            "data_source": data_source,
            "sources": [
                {
                    "source": item["source"],
                    "chunk_index": item["chunk_index"],
                    "distance": round(
                        float(item["distance"]),
                        4,
                    ),
                }
                for item in result[
                    "retrieved_knowledge"
                ]
            ],
        }

    except HTTPException:
        raise

    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=(
                f"Assistant failed: {str(exc)}"
            ),
        ) from exc


# ============================================================
# Vehicle health and history
# ============================================================

@app.get("/vehicle/{unit}/health")
def vehicle_health(
    unit: int,
    history_points: int = 10,
):

    try:

        result = build_vehicle_health_history(
            unit=unit,
            history_points=history_points,
        )

        return result

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Vehicle health analysis failed: "
                f"{str(exc)}"
            ),
        ) from exc


# ============================================================
# Vehicle telemetry
# ============================================================

@app.get("/vehicle/{unit}/telemetry")
def vehicle_telemetry(
    unit: int,
):

    try:

        vehicle_df = load_vehicle_history(
            unit
        )

        return {
            "unit": unit,
            "telemetry": vehicle_df.to_dict(
                orient="records"
            ),
            "data_source": {
                "mode": "demo",
                "label": "DEMO TELEMETRY",
                "filename": "vehicle_unit_1.csv"
                if unit == 1
                else None,
            },
        }

    except ValueError as exc:

        raise HTTPException(
            status_code=404,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        ) from exc

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=(
                "Telemetry retrieval failed: "
                f"{str(exc)}"
            ),
        ) from exc
