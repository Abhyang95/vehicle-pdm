from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    precision_score,
    recall_score,
    f1_score,
)

from feature_engineering import (
    SENSOR_COLUMNS,
    build_features,
    validate_features,
)


ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw" / "cmapss"
MODEL_DIR = ROOT / "models"
DIAGNOSTICS_DIR = ROOT / "data" / "processed" / "diagnostics"

TEST_PATH = RAW_DIR / "test_FD001.txt"
RUL_PATH = RAW_DIR / "RUL_FD001.txt"

RUL_MODEL_PATH = MODEL_DIR / "rul_xgboost.joblib"
CLASSIFIER_MODEL_PATH = MODEL_DIR / "failure_risk_xgboost.joblib"

RUL_FEATURES_PATH = MODEL_DIR / "rul_features.json"
CLASSIFIER_FEATURES_PATH = MODEL_DIR / "failure_risk_features.json"

MAX_RUL = 125
FAILURE_RUL_THRESHOLD = 30

FAILURE_PROBABILITY_THRESHOLD = 0.50

BOOTSTRAP_ITERATIONS = 10_000
BOOTSTRAP_SEED = 42

EXPECTED_FEATURE_COUNT = 143


RAW_COLUMNS = [
    "unit",
    "cycle",
    "setting_1",
    "setting_2",
    "setting_3",
    "sensor_1",
    "sensor_2",
    "sensor_3",
    "sensor_4",
    "sensor_5",
    "sensor_6",
    "sensor_7",
    "sensor_8",
    "sensor_9",
    "sensor_10",
    "sensor_11",
    "sensor_12",
    "sensor_13",
    "sensor_14",
    "sensor_15",
    "sensor_16",
    "sensor_17",
    "sensor_18",
    "sensor_19",
    "sensor_20",
    "sensor_21",
]


def load_official_test():
    """Load official NASA C-MAPSS FD001 test telemetry."""

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            f"Official test file not found: {TEST_PATH}"
        )

    if not RUL_PATH.exists():
        raise FileNotFoundError(
            f"Official RUL file not found: {RUL_PATH}"
        )

    test_df = pd.read_csv(
        TEST_PATH,
        sep=r"\s+",
        header=None,
        names=RAW_COLUMNS,
    )

    true_rul = pd.read_csv(
        RUL_PATH,
        sep=r"\s+",
        header=None,
        names=["RUL"],
    )

    if test_df["unit"].nunique() != 100:
        raise ValueError(
            f"Expected 100 official test engines, "
            f"found {test_df['unit'].nunique()}."
        )

    if len(true_rul) != test_df["unit"].nunique():
        raise ValueError(
            "Official RUL file length does not match "
            "the number of test engines."
        )

    return test_df, true_rul["RUL"].to_numpy(dtype=float)


def load_models():
    """Load trained RUL and failure-risk models."""

    if not RUL_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"RUL model not found: {RUL_MODEL_PATH}"
        )

    if not CLASSIFIER_MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Classifier model not found: {CLASSIFIER_MODEL_PATH}"
        )

    rul_model = joblib.load(RUL_MODEL_PATH)
    classifier_model = joblib.load(CLASSIFIER_MODEL_PATH)

    return rul_model, classifier_model


def validate_feature_schema():
    """Load and validate the saved model feature schemas.

    The training scripts may save feature metadata either as a plain
    list or as a dictionary containing the feature list. This function
    normalizes both formats into one canonical list.
    """

    import json

    def load_feature_metadata(path, model_name):
        with open(path, "r", encoding="utf-8") as file:
            metadata = json.load(file)

        if isinstance(metadata, list):
            features = metadata

        elif isinstance(metadata, dict):
            possible_keys = [
                "features",
                "feature_columns",
                "model_features",
                "columns",
            ]

            features = None

            for key in possible_keys:
                value = metadata.get(key)

                if isinstance(value, list):
                    features = value
                    break

            if features is None:
                raise ValueError(
                    f"{model_name} feature metadata is a dictionary, "
                    f"but no supported feature-list key was found.\n"
                    f"Available keys: {list(metadata.keys())}"
                )

        else:
            raise ValueError(
                f"{model_name} feature metadata must be a list or "
                f"dictionary, found {type(metadata).__name__}."
            )

        if not all(isinstance(feature, str) for feature in features):
            raise ValueError(
                f"{model_name} feature schema contains non-string values."
            )

        return features

    rul_features = load_feature_metadata(
        RUL_FEATURES_PATH,
        "RUL",
    )

    classifier_features = load_feature_metadata(
        CLASSIFIER_FEATURES_PATH,
        "classifier",
    )

    if len(rul_features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} RUL features, "
            f"found {len(rul_features)}."
        )

    if len(classifier_features) != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} classifier features, "
            f"found {len(classifier_features)}."
        )

    rul_set = set(rul_features)
    classifier_set = set(classifier_features)

    missing_from_classifier = sorted(
        rul_set - classifier_set
    )

    extra_in_classifier = sorted(
        classifier_set - rul_set
    )

    if missing_from_classifier or extra_in_classifier:
        raise ValueError(
            "RUL and classifier feature schemas contain different "
            "feature names.\n"
            f"Missing from classifier: {missing_from_classifier}\n"
            f"Extra in classifier: {extra_in_classifier}"
        )

    forbidden = {
        "cycle",
        "cycle_progress",
        "final_cycle",
        "max_cycle",
        "unit_max_cycle",
    }

    leakage = forbidden.intersection(rul_set)

    if leakage:
        raise ValueError(
            f"Forbidden leakage features detected: {sorted(leakage)}"
        )

    print()
    print("Feature metadata:")
    print(f"  RUL schema: {len(rul_features)} features")
    print(f"  Classifier schema: {len(classifier_features)} features")
    print("  Feature-name sets: MATCH")
    print("  Leakage features: NONE")

    # RUL schema is used as the canonical DataFrame ordering.
    return rul_features


def build_official_features(test_df, feature_columns):
    """Build causal features using exactly the training sensor schema."""

    available_columns = set(test_df.columns)

    required = {
        "unit",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
        *SENSOR_COLUMNS,
    }

    missing = required - available_columns

    if missing:
        raise ValueError(
            f"Official test telemetry is missing columns: {sorted(missing)}"
        )

    selected_columns = [
        "unit",
        "cycle",
        "setting_1",
        "setting_2",
        "setting_3",
        *SENSOR_COLUMNS,
    ]

    selected = test_df[selected_columns].copy()

    features = build_features(selected)

    validate_features(features)

    actual_feature_columns = [
        column
        for column in features.columns
        if column not in {
            "unit",
            "cycle",
            "RUL",
            "failure_risk",
        }
    ]

    if actual_feature_columns != feature_columns:
        missing_features = sorted(
            set(feature_columns) - set(actual_feature_columns)
        )
        extra_features = sorted(
            set(actual_feature_columns) - set(feature_columns)
        )

        raise ValueError(
            "Official test feature schema does not match training schema.\n"
            f"Missing: {missing_features}\n"
            f"Extra: {extra_features}"
        )

    return features


def get_final_engine_rows(features):
    """Take the final observed telemetry row for every official test engine."""

    final_rows = (
        features
        .sort_values(["unit", "cycle"])
        .groupby("unit", as_index=False)
        .tail(1)
        .sort_values("unit")
        .reset_index(drop=True)
    )

    if len(final_rows) != 100:
        raise ValueError(
            f"Expected 100 final engine rows, found {len(final_rows)}."
        )

    return final_rows


def calculate_rul_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def calculate_classification_metrics(y_true, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)

    return {
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0,
        ),
        "false_negatives": int(
            ((y_true == 1) & (predictions == 0)).sum()
        ),
        "false_positives": int(
            ((y_true == 0) & (predictions == 1)).sum()
        ),
    }


def bootstrap_engine_metrics(
    evaluation_df,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=BOOTSTRAP_SEED,
):
    """
    Bootstrap the official evaluation by resampling complete engines.

    Each bootstrap sample contains 100 engines sampled with replacement.
    This preserves the engine-level evaluation unit.
    """

    rng = np.random.default_rng(seed)

    engine_ids = evaluation_df["unit"].to_numpy()

    y_true_rul = evaluation_df["true_rul"].to_numpy(dtype=float)
    y_pred_rul = evaluation_df["predicted_rul"].to_numpy(dtype=float)

    y_true_failure = evaluation_df["true_failure"].to_numpy(dtype=int)
    probabilities = evaluation_df["failure_probability"].to_numpy(dtype=float)

    n_engines = len(engine_ids)

    rul_results = np.empty((iterations, 3), dtype=float)
    classification_results = np.empty((iterations, 3), dtype=float)

    valid_classification_iterations = 0

    for iteration in range(iterations):
        sample_indices = rng.integers(
            0,
            n_engines,
            size=n_engines,
        )

        sample_true_rul = y_true_rul[sample_indices]
        sample_pred_rul = y_pred_rul[sample_indices]

        rul_results[iteration, 0] = mean_absolute_error(
            sample_true_rul,
            sample_pred_rul,
        )

        rul_results[iteration, 1] = np.sqrt(
            mean_squared_error(
                sample_true_rul,
                sample_pred_rul,
            )
        )

        rul_results[iteration, 2] = r2_score(
            sample_true_rul,
            sample_pred_rul,
        )

        sample_true_failure = y_true_failure[sample_indices]
        sample_probabilities = probabilities[sample_indices]

        sample_predictions = (
            sample_probabilities >= FAILURE_PROBABILITY_THRESHOLD
        ).astype(int)

        classification_results[
            iteration,
            0,
        ] = precision_score(
            sample_true_failure,
            sample_predictions,
            zero_division=0,
        )

        classification_results[
            iteration,
            1,
        ] = recall_score(
            sample_true_failure,
            sample_predictions,
            zero_division=0,
        )

        classification_results[
            iteration,
            2,
        ] = f1_score(
            sample_true_failure,
            sample_predictions,
            zero_division=0,
        )

        valid_classification_iterations += 1

    return rul_results, classification_results


def percentile_ci(values, lower=2.5, upper=97.5):
    return (
        float(np.percentile(values, lower)),
        float(np.percentile(values, upper)),
    )


def build_rul_range_metrics(evaluation_df):
    """Calculate RUL error by clinically relevant prediction ranges."""

    def assign_range(rul):
        if rul <= 10:
            return "0-10"
        if rul <= 20:
            return "11-20"
        if rul <= 30:
            return "21-30"
        if rul <= 40:
            return "31-40"
        if rul <= 60:
            return "41-60"
        return "61+"

    result = evaluation_df.copy()
    result["rul_range"] = result["true_rul"].apply(assign_range)

    range_order = [
        "0-10",
        "11-20",
        "21-30",
        "31-40",
        "41-60",
        "61+",
    ]

    rows = []

    for rul_range in range_order:
        subset = result[result["rul_range"] == rul_range]

        if subset.empty:
            continue

        rows.append(
            {
                "rul_range": rul_range,
                "engines": len(subset),
                "mae": mean_absolute_error(
                    subset["true_rul"],
                    subset["predicted_rul"],
                ),
                "rmse": np.sqrt(
                    mean_squared_error(
                        subset["true_rul"],
                        subset["predicted_rul"],
                    )
                ),
                "mean_true_rul": subset["true_rul"].mean(),
                "mean_predicted_rul": subset["predicted_rul"].mean(),
                "mean_signed_error": (
                    subset["predicted_rul"]
                    - subset["true_rul"]
                ).mean(),
            }
        )

    return pd.DataFrame(rows)


def build_threshold_analysis(evaluation_df):
    """Evaluate failure-risk performance across probability thresholds."""

    true_failure = evaluation_df["true_failure"].to_numpy(dtype=int)
    probabilities = evaluation_df[
        "failure_probability"
    ].to_numpy(dtype=float)

    thresholds = np.arange(
        0.30,
        0.71,
        0.05,
    )

    rows = []

    for threshold in thresholds:
        predictions = (
            probabilities >= threshold
        ).astype(int)

        rows.append(
            {
                "threshold": round(float(threshold), 2),
                "precision": precision_score(
                    true_failure,
                    predictions,
                    zero_division=0,
                ),
                "recall": recall_score(
                    true_failure,
                    predictions,
                    zero_division=0,
                ),
                "f1": f1_score(
                    true_failure,
                    predictions,
                    zero_division=0,
                ),
                "false_negatives": int(
                    (
                        (true_failure == 1)
                        & (predictions == 0)
                    ).sum()
                ),
                "false_positives": int(
                    (
                        (true_failure == 0)
                        & (predictions == 1)
                    ).sum()
                ),
            }
        )

    return pd.DataFrame(rows)


def main():
    DIAGNOSTICS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 70)
    print("OFFICIAL NASA C-MAPSS FD001 HELD-OUT EVALUATION")
    print("=" * 70)

    test_df, true_rul = load_official_test()

    rul_model, classifier_model = load_models()

    feature_columns = validate_feature_schema()

    print()
    print(f"Official test rows: {len(test_df):,}")
    print(f"Official test engines: {test_df['unit'].nunique()}")
    print(f"Model feature count: {len(feature_columns)}")
    print(f"Training sensors retained: {len(SENSOR_COLUMNS)}")

    print()
    print("Building causal features with the training sensor schema...")

    features = build_official_features(
        test_df,
        feature_columns,
    )

    print("Feature schema check: PASS")

    final_rows = get_final_engine_rows(features)

    X = final_rows[feature_columns]

    predicted_rul = rul_model.predict(X)

    predicted_rul = np.clip(
        predicted_rul,
        0,
        MAX_RUL,
    )

    failure_probability = classifier_model.predict_proba(X)[:, 1]

    observed_cycles = final_rows["cycle"].to_numpy(dtype=int)
    units = final_rows["unit"].to_numpy(dtype=int)

    true_failure = (
        true_rul <= FAILURE_RUL_THRESHOLD
    ).astype(int)

    evaluation_df = pd.DataFrame(
        {
            "unit": units,
            "observed_cycle": observed_cycles,
            "true_rul": true_rul,
            "predicted_rul": predicted_rul,
            "rul_error": predicted_rul - true_rul,
            "absolute_rul_error": np.abs(
                predicted_rul - true_rul
            ),
            "failure_probability": failure_probability,
            "true_failure": true_failure,
        }
    )

    # ---------------------------------------------------------------
    # RUL
    # ---------------------------------------------------------------

    rul_metrics = calculate_rul_metrics(
        true_rul,
        predicted_rul,
    )

    baseline_prediction = float(np.mean(true_rul))

    baseline_predictions = np.full_like(
        true_rul,
        baseline_prediction,
        dtype=float,
    )

    baseline_metrics = calculate_rul_metrics(
        true_rul,
        baseline_predictions,
    )

    print()
    print("=" * 70)
    print("OFFICIAL ENGINE-LEVEL RUL RESULTS")
    print("=" * 70)

    print(f"MAE: {rul_metrics['mae']:.4f}")
    print(f"RMSE: {rul_metrics['rmse']:.4f}")
    print(f"R²: {rul_metrics['r2']:.4f}")

    print()
    print("=" * 70)
    print("MEAN-RUL BASELINE")
    print("=" * 70)

    print(
        f"Baseline prediction: "
        f"{baseline_prediction:.2f}"
    )
    print(
        f"Baseline MAE: "
        f"{baseline_metrics['mae']:.4f}"
    )
    print(
        f"Baseline RMSE: "
        f"{baseline_metrics['rmse']:.4f}"
    )
    print(
        f"Baseline R²: "
        f"{baseline_metrics['r2']:.4f}"
    )

    improvement = (
        1
        - rul_metrics["mae"]
        / baseline_metrics["mae"]
    ) * 100

    print(
        f"MAE improvement vs baseline: "
        f"{improvement:.2f}%"
    )

    # ---------------------------------------------------------------
    # RUL range analysis
    # ---------------------------------------------------------------

    range_metrics = build_rul_range_metrics(
        evaluation_df
    )

    print()
    print("=" * 70)
    print("RUL ERROR BY RANGE")
    print("=" * 70)

    print(
        range_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # ---------------------------------------------------------------
    # Failure-risk evaluation
    # ---------------------------------------------------------------

    classification_metrics = calculate_classification_metrics(
        true_failure,
        failure_probability,
        FAILURE_PROBABILITY_THRESHOLD,
    )

    print()
    print("=" * 70)
    print("OFFICIAL FAILURE-RISK RESULTS")
    print("=" * 70)

    print(
        f"Failure threshold: "
        f"RUL <= {FAILURE_RUL_THRESHOLD}"
    )

    print(
        f"Probability threshold: "
        f"{FAILURE_PROBABILITY_THRESHOLD:.2f}"
    )

    print(
        f"Precision: "
        f"{classification_metrics['precision']:.4f}"
    )

    print(
        f"Recall: "
        f"{classification_metrics['recall']:.4f}"
    )

    print(
        f"F1: "
        f"{classification_metrics['f1']:.4f}"
    )

    print(
        f"False negatives: "
        f"{classification_metrics['false_negatives']}"
    )

    print(
        f"False positives: "
        f"{classification_metrics['false_positives']}"
    )

    # ---------------------------------------------------------------
    # Threshold analysis
    # ---------------------------------------------------------------

    threshold_metrics = build_threshold_analysis(
        evaluation_df
    )

    print()
    print("=" * 70)
    print("FAILURE-RISK THRESHOLD ANALYSIS")
    print("=" * 70)

    print(
        threshold_metrics.to_string(
            index=False,
            float_format=lambda value: f"{value:.4f}",
        )
    )

    # ---------------------------------------------------------------
    # Engine-level bootstrap
    # ---------------------------------------------------------------

    print()
    print("=" * 70)
    print(
        f"ENGINE-LEVEL BOOTSTRAP "
        f"({BOOTSTRAP_ITERATIONS:,} ITERATIONS)"
    )
    print("=" * 70)

    rul_bootstrap, classification_bootstrap = (
        bootstrap_engine_metrics(
            evaluation_df,
        )
    )

    mae_ci = percentile_ci(
        rul_bootstrap[:, 0]
    )

    rmse_ci = percentile_ci(
        rul_bootstrap[:, 1]
    )

    r2_ci = percentile_ci(
        rul_bootstrap[:, 2]
    )

    precision_ci = percentile_ci(
        classification_bootstrap[:, 0]
    )

    recall_ci = percentile_ci(
        classification_bootstrap[:, 1]
    )

    f1_ci = percentile_ci(
        classification_bootstrap[:, 2]
    )

    print()
    print("RUL METRICS")
    print(
        f"MAE:  {rul_metrics['mae']:.4f} "
        f"(95% CI: {mae_ci[0]:.4f} - {mae_ci[1]:.4f})"
    )

    print(
        f"RMSE: {rul_metrics['rmse']:.4f} "
        f"(95% CI: {rmse_ci[0]:.4f} - {rmse_ci[1]:.4f})"
    )

    print(
        f"R²:   {rul_metrics['r2']:.4f} "
        f"(95% CI: {r2_ci[0]:.4f} - {r2_ci[1]:.4f})"
    )

    print()
    print("FAILURE-RISK METRICS")

    print(
        f"Precision: {classification_metrics['precision']:.4f} "
        f"(95% CI: {precision_ci[0]:.4f} - {precision_ci[1]:.4f})"
    )

    print(
        f"Recall:    {classification_metrics['recall']:.4f} "
        f"(95% CI: {recall_ci[0]:.4f} - {recall_ci[1]:.4f})"
    )

    print(
        f"F1:        {classification_metrics['f1']:.4f} "
        f"(95% CI: {f1_ci[0]:.4f} - {f1_ci[1]:.4f})"
    )

    # ---------------------------------------------------------------
    # Save outputs
    # ---------------------------------------------------------------

    predictions_path = (
        DIAGNOSTICS_DIR
        / "official_fd001_test_predictions.csv"
    )

    metrics_path = (
        DIAGNOSTICS_DIR
        / "official_fd001_test_metrics.csv"
    )

    range_path = (
        DIAGNOSTICS_DIR
        / "official_fd001_rul_range_metrics.csv"
    )

    threshold_path = (
        DIAGNOSTICS_DIR
        / "official_fd001_threshold_metrics.csv"
    )

    bootstrap_path = (
        DIAGNOSTICS_DIR
        / "official_fd001_bootstrap_metrics.csv"
    )

    evaluation_df.to_csv(
        predictions_path,
        index=False,
    )

    range_metrics.to_csv(
        range_path,
        index=False,
    )

    threshold_metrics.to_csv(
        threshold_path,
        index=False,
    )

    bootstrap_metrics = pd.DataFrame(
        [
            {
                "metric": "RUL MAE",
                "point_estimate": rul_metrics["mae"],
                "ci_lower": mae_ci[0],
                "ci_upper": mae_ci[1],
            },
            {
                "metric": "RUL RMSE",
                "point_estimate": rul_metrics["rmse"],
                "ci_lower": rmse_ci[0],
                "ci_upper": rmse_ci[1],
            },
            {
                "metric": "RUL R2",
                "point_estimate": rul_metrics["r2"],
                "ci_lower": r2_ci[0],
                "ci_upper": r2_ci[1],
            },
            {
                "metric": "Failure Precision",
                "point_estimate": classification_metrics[
                    "precision"
                ],
                "ci_lower": precision_ci[0],
                "ci_upper": precision_ci[1],
            },
            {
                "metric": "Failure Recall",
                "point_estimate": classification_metrics[
                    "recall"
                ],
                "ci_lower": recall_ci[0],
                "ci_upper": recall_ci[1],
            },
            {
                "metric": "Failure F1",
                "point_estimate": classification_metrics[
                    "f1"
                ],
                "ci_lower": f1_ci[0],
                "ci_upper": f1_ci[1],
            },
        ]
    )

    bootstrap_metrics.to_csv(
        bootstrap_path,
        index=False,
    )

    metrics_summary = pd.DataFrame(
        [
            {
                "evaluation": "official_fd001_engine_level",
                "engines": len(evaluation_df),
                "rul_mae": rul_metrics["mae"],
                "rul_rmse": rul_metrics["rmse"],
                "rul_r2": rul_metrics["r2"],
                "baseline_mae": baseline_metrics["mae"],
                "baseline_rmse": baseline_metrics["rmse"],
                "baseline_r2": baseline_metrics["r2"],
                "mae_improvement_percent": improvement,
                "failure_threshold_rul": FAILURE_RUL_THRESHOLD,
                "failure_probability_threshold": (
                    FAILURE_PROBABILITY_THRESHOLD
                ),
                "failure_precision": classification_metrics[
                    "precision"
                ],
                "failure_recall": classification_metrics[
                    "recall"
                ],
                "failure_f1": classification_metrics[
                    "f1"
                ],
                "false_negatives": classification_metrics[
                    "false_negatives"
                ],
                "false_positives": classification_metrics[
                    "false_positives"
                ],
                "rul_mae_ci_lower": mae_ci[0],
                "rul_mae_ci_upper": mae_ci[1],
                "rul_rmse_ci_lower": rmse_ci[0],
                "rul_rmse_ci_upper": rmse_ci[1],
                "rul_r2_ci_lower": r2_ci[0],
                "rul_r2_ci_upper": r2_ci[1],
                "failure_precision_ci_lower": precision_ci[0],
                "failure_precision_ci_upper": precision_ci[1],
                "failure_recall_ci_lower": recall_ci[0],
                "failure_recall_ci_upper": recall_ci[1],
                "failure_f1_ci_lower": f1_ci[0],
                "failure_f1_ci_upper": f1_ci[1],
            }
        ]
    )

    metrics_summary.to_csv(
        metrics_path,
        index=False,
    )

    print()
    print("=" * 70)
    print("FILES SAVED")
    print("=" * 70)

    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Range metrics: {range_path}")
    print(f"Threshold metrics: {threshold_path}")
    print(f"Bootstrap metrics: {bootstrap_path}")

    print()
    print("OFFICIAL EVALUATION COMPLETE")


if __name__ == "__main__":
    main()