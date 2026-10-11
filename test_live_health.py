import pandas as pd
from src.subsystem_health import calculate_live_group_health

path = "data/processed/fd001_train_labeled.parquet"
df = pd.read_parquet(path)

columns = [
    "unit", "cycle",
    "sensor_2", "sensor_3", "sensor_4",
    "sensor_7", "sensor_8", "sensor_9", "sensor_11",
    "sensor_12", "sensor_13", "sensor_14", "sensor_15",
    "sensor_17", "sensor_20", "sensor_21",
]
df = df[columns]

print("LIVE HEALTH CONSISTENCY TEST")
print("=" * 65)

for unit in [1, 2, 5, 10]:
    print("\nUNIT", unit)

    for end_cycle in [100, 120, 150, 192]:
        window = df[
            (df["unit"] == unit) & (df["cycle"] <= end_cycle)
        ].copy()

        if window.empty or window["cycle"].max() < 100:
            continue

        health = calculate_live_group_health(window)
        row = health[health["cycle"] == 100]

        if not row.empty:
            score = float(row.iloc[0]["overall_sensor_health"])
            print(
                "  Input ends at cycle {}: health at cycle 100 = {:.2f}".format(
                    end_cycle, score
                )
            )
