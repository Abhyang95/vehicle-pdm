import pandas as pd
import pytest
from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)
TELEMETRY_PATH = "data/processed/vehicle_unit_1.csv"


@pytest.fixture(scope="module")
def telemetry_payload():
    df = pd.read_csv(TELEMETRY_PATH)

    required = [
        "unit", "cycle",
        "setting_1", "setting_2", "setting_3",
        "sensor_2", "sensor_3", "sensor_4",
        "sensor_7", "sensor_8", "sensor_9",
        "sensor_11", "sensor_12", "sensor_13",
        "sensor_14", "sensor_15", "sensor_17",
        "sensor_20", "sensor_21",
    ]

    missing = sorted(set(required) - set(df.columns))
    assert not missing, f"Demo telemetry missing columns: {missing}"

    df = df[required].sort_values("cycle")
    return {"telemetry": df.to_dict(orient="records")}


def test_predict_cycle_192(telemetry_payload):
    rows = telemetry_payload["telemetry"]
    selected = [row for row in rows if row["cycle"] <= 192]

    if not selected:
        pytest.skip("Demo telemetry has no observations through cycle 192")

    response = client.post("/predict", json={"telemetry": selected})
    assert response.status_code == 200, response.text

    data = response.json()
    assert data["unit"] == int(selected[-1]["unit"])
    assert data["current_cycle"] == int(selected[-1]["cycle"])
    assert 0 <= data["failure_probability"] <= 1
    assert 0 <= data["predicted_rul"] <= 125
    assert data["risk_status"] in {"LOW_RISK", "WATCH", "HIGH_RISK"}
    assert data["severity"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
    assert data["feature_count"] > 0


def test_diagnostics_contract(telemetry_payload):
    response = client.post("/diagnostics", json=telemetry_payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert isinstance(data, dict)
    assert data, "Diagnostics returned an empty report"


def test_maintenance_contract(telemetry_payload):
    response = client.post("/maintenance", json=telemetry_payload)
    assert response.status_code == 200, response.text

    data = response.json()
    assert isinstance(data, dict)

    for key in (
        "unit",
        "current_cycle",
        "failure_probability",
        "predicted_rul",
    ):
        assert key in data, f"Maintenance response missing {key}"

    assert isinstance(data["failure_probability"], (int, float))
    assert 0 <= data["failure_probability"] <= 1
    assert 0 <= data["predicted_rul"] <= 125


def test_predict_rejects_empty_telemetry():
    response = client.post("/predict", json={"telemetry": []})
    assert response.status_code == 422


def test_diagnostics_rejects_empty_telemetry():
    response = client.post("/diagnostics", json={"telemetry": []})
    assert response.status_code == 422


def test_maintenance_rejects_empty_telemetry():
    response = client.post("/maintenance", json={"telemetry": []})
    assert response.status_code == 422
