import uuid

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import REPLAY_SESSIONS, app


client = TestClient(app)
DATA_PATH = "data/processed/vehicle_unit_1.csv"


@pytest.fixture
def replay_session():
    df = pd.read_csv(DATA_PATH).sort_values("cycle").copy()

    if df.empty:
        pytest.skip("Replay telemetry dataset is empty")

    unit = int(df["unit"].iloc[0])
    unit_df = df[df["unit"] == unit].copy()
    cycles = sorted(unit_df["cycle"].astype(int).unique())

    if len(cycles) < 2:
        pytest.skip("Replay dataset needs at least two cycles")

    session_id = f"test-replay-{uuid.uuid4().hex}"
    REPLAY_SESSIONS[session_id] = {
        "telemetry": df,
        "filename": "test_uploaded_replay.csv",
    }

    try:
        yield session_id, unit, cycles
    finally:
        REPLAY_SESSIONS.pop(session_id, None)


def test_replay_history_includes_health_and_risk(replay_session):
    session_id, unit, cycles = replay_session
    selected_cycle = cycles[min(len(cycles) - 1, 10)]

    response = client.get(
        f"/replay/{session_id}/analyze",
        params={"unit": unit, "cycle": selected_cycle},
    )

    assert response.status_code == 200, response.text
    data = response.json()
    history = data["rul_history"]

    assert len(history) >= 2
    assert int(history[-1]["cycle"]) == selected_cycle

    for point in history:
        assert {
            "cycle",
            "predicted_rul",
            "overall_health",
            "failure_risk",
        }.issubset(point)

        assert 0 <= point["overall_health"] <= 100
        assert 0 <= point["failure_risk"] <= 1
        assert 0 <= point["predicted_rul"] <= 125

        assert pd.notna(point["overall_health"])
        assert pd.notna(point["failure_risk"])
        assert pd.notna(point["predicted_rul"])

    assert all(
        int(point["cycle"]) <= selected_cycle
        for point in history
    )

def test_replay_history_isolated_from_future_rows(replay_session):
    session_id, unit, cycles = replay_session
    selected_cycle = cycles[min(len(cycles) - 1, 10)]

    full_df = REPLAY_SESSIONS[session_id]["telemetry"]
    truncated_df = full_df[
        full_df["cycle"] <= selected_cycle
    ].copy()

    truncated_session_id = f"test-replay-truncated-{uuid.uuid4().hex}"
    REPLAY_SESSIONS[truncated_session_id] = {
        "telemetry": truncated_df,
        "filename": "test_uploaded_replay.csv",
    }

    try:
        full_response = client.get(
            f"/replay/{session_id}/analyze",
            params={"unit": unit, "cycle": selected_cycle},
        )
        truncated_response = client.get(
            f"/replay/{truncated_session_id}/analyze",
            params={"unit": unit, "cycle": selected_cycle},
        )

        assert full_response.status_code == 200, full_response.text
        assert truncated_response.status_code == 200, truncated_response.text

        full_history = full_response.json()["rul_history"]
        truncated_history = truncated_response.json()["rul_history"]

        assert len(full_history) == len(truncated_history)

        for full_point, truncated_point in zip(
            full_history, truncated_history
        ):
            assert full_point["cycle"] == truncated_point["cycle"]

            assert full_point["predicted_rul"] == pytest.approx(
                truncated_point["predicted_rul"], abs=1e-9
            )
            assert full_point["overall_health"] == pytest.approx(
                truncated_point["overall_health"], abs=1e-9
            )
            assert full_point["failure_risk"] == pytest.approx(
                truncated_point["failure_risk"], abs=1e-9
            )
    finally:
        REPLAY_SESSIONS.pop(truncated_session_id, None)

def test_current_prediction_matches_final_history_point(replay_session):
    session_id, unit, cycles = replay_session
    selected_cycle = cycles[min(len(cycles) - 1, 10)]

    response = client.get(
        f"/replay/{session_id}/analyze",
        params={"unit": unit, "cycle": selected_cycle},
    )

    assert response.status_code == 200, response.text
    data = response.json()

    prediction = data["prediction"]
    history = data["rul_history"]

    assert history, "Replay history should not be empty"
    assert int(history[-1]["cycle"]) == selected_cycle

    assert prediction["predicted_rul"] == pytest.approx(
        history[-1]["predicted_rul"],
        abs=0.01,
    )

    assert prediction["failure_probability"] == pytest.approx(
        history[-1]["failure_risk"],
        abs=0.0001,
    )
