import pandas as pd
import pytest

from src.inference import (
    build_inference_features,
    predict_from_features,
    predict_from_telemetry,
)
from src.vehicle_health import load_vehicle_history


@pytest.mark.parametrize("unit", [1, 10, 25])
def test_full_trajectory_features_match_prefix_features(unit):
    telemetry = load_vehicle_history(unit)
    full_features = build_inference_features(telemetry)

    indices = sorted(set([
        0,
        len(telemetry) // 2,
        len(telemetry) - 1,
    ]))

    for index in indices:
        prefix = telemetry.iloc[: index + 1].copy()
        prefix_features = build_inference_features(prefix)

        pd.testing.assert_frame_equal(
            full_features.iloc[[index]].reset_index(drop=True),
            prefix_features.iloc[[-1]].reset_index(drop=True),
            check_dtype=False,
            check_exact=False,
            rtol=0,
            atol=1e-12,
        )
        original_prediction = predict_from_telemetry(prefix)

        optimized_prediction = predict_from_features(
            full_features.iloc[: index + 1]
        )

        assert optimized_prediction["unit"] == original_prediction["unit"]
        assert optimized_prediction["current_cycle"] == original_prediction["current_cycle"]

        assert optimized_prediction["failure_probability"] == pytest.approx(
            original_prediction["failure_probability"],
            abs=1e-12,
        )
        assert optimized_prediction["predicted_rul"] == pytest.approx(
            original_prediction["predicted_rul"],
            abs=1e-12,
        )