import pandas as pd
import pytest

from src.inference import compare_replay_consistency


DATA_PATH = "data/processed/fd001_train_labeled.parquet"
ENGINES = [1, 10, 25, 50, 75, 100]


@pytest.fixture(scope="module")
def telemetry():
    df = pd.read_parquet(DATA_PATH)

    required = {"unit", "cycle"}
    assert required.issubset(df.columns), (
        f"Missing required columns: {required - set(df.columns)}"
    )

    return df


@pytest.mark.parametrize("unit", ENGINES)
def test_replay_consistency_early_middle_late(telemetry, unit):
    unit_df = telemetry[telemetry["unit"] == unit].sort_values("cycle")

    assert not unit_df.empty, f"Engine {unit} missing from dataset"

    cycles = unit_df["cycle"].astype(int).tolist()
    selected_cycles = sorted({
        cycles[0],
        cycles[len(cycles) // 2],
        cycles[-1],
    })

    for cycle in selected_cycles:
        result = compare_replay_consistency(
            full_telemetry=telemetry,
            unit=unit,
            cycle=cycle,
        )

        assert result["passed"], (
            f"Replay consistency failed for engine {unit}, cycle {cycle}: "
            f"max difference={result['max_absolute_difference']}, "
            f"tolerance={result['tolerance']}"
        )
        assert (
            result["max_absolute_difference"] <= result["tolerance"]
        )


def test_missing_engine_raises_value_error(telemetry):
    with pytest.raises(ValueError, match="not found"):
        compare_replay_consistency(
            full_telemetry=telemetry,
            unit=-999,
            cycle=1,
        )


def test_missing_cycle_raises_value_error(telemetry):
    with pytest.raises(ValueError, match="No telemetry exists"):
        compare_replay_consistency(
            full_telemetry=telemetry,
            unit=1,
            cycle=-1,
        )
