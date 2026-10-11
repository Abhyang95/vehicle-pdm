from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

TEST_FILE = PROCESSED_DIR / "test_scaled.parquet"

CLASSIFIER_FILE = (
    PROCESSED_DIR / "xgboost_classifier.joblib"
)

THRESHOLD_FILE = (
    PROCESSED_DIR / "xgboost_failure_threshold.joblib"
)

RUL_MODEL_FILE = (
    PROCESSED_DIR / "xgboost_rul_regressor.joblib"
)


def load_artifacts():
    test_df = pd.read_parquet(TEST_FILE)

    classifier_artifact = joblib.load(
        CLASSIFIER_FILE
    )

    threshold_artifact = joblib.load(
        THRESHOLD_FILE
    )

    rul_artifact = joblib.load(
        RUL_MODEL_FILE
    )

    classifier = classifier_artifact["model"]
    classifier_features = classifier_artifact[
        "feature_columns"
    ]

    threshold = threshold_artifact[
        "threshold"
    ]

    rul_model = rul_artifact["model"]
    rul_features = rul_artifact[
        "feature_columns"
    ]

    if classifier_features != rul_features:
        raise ValueError(
            "Classifier and RUL feature columns do not match."
        )

    return (
        test_df,
        classifier,
        rul_model,
        classifier_features,
        threshold,
    )


def calculate_severity(
    failure_probability,
    predicted_rul,
):
    """
    Convert model outputs into an interpretable
    maintenance severity category.

    These thresholds are policy rules, not learned
    model parameters.
    """

    if (
        failure_probability >= 0.85
        or predicted_rul <= 10
    ):
        return "CRITICAL"

    if (
        failure_probability >= 0.65
        or predicted_rul <= 30
    ):
        return "HIGH"

    if (
        failure_probability >= 0.35
        or predicted_rul <= 60
    ):
        return "MEDIUM"

    return "LOW"


def generate_diagnostics(
    test_df,
    classifier,
    rul_model,
    feature_columns,
    threshold,
):
    X = test_df[
        feature_columns
    ]

    failure_probabilities = (
        classifier.predict_proba(X)[:, 1]
    )

    failure_predictions = (
        failure_probabilities >= threshold
    ).astype(int)

    rul_predictions = rul_model.predict(X)

    rul_predictions = np.clip(
        rul_predictions,
        0,
        125,
    )

    severity_levels = [
        calculate_severity(
            probability,
            rul,
        )
        for probability, rul
        in zip(
            failure_probabilities,
            rul_predictions,
        )
    ]

    diagnostics_df = pd.DataFrame(
        {
            "unit": test_df["unit"].astype(int),
            "cycle": test_df["cycle"].astype(int),
            "actual_rul": test_df["RUL"],
            "failure_probability": (
                failure_probabilities
            ),
            "failure_prediction": (
                failure_predictions
            ),
            "predicted_rul": (
                rul_predictions
            ),
            "severity": severity_levels,
        }
    )

    return diagnostics_df


def print_summary(
    diagnostics_df,
):
    print(
        "\n" + "=" * 70
    )
    print(
        "DIAGNOSTIC SEVERITY SUMMARY"
    )
    print(
        "=" * 70
    )

    print(
        "\nSeverity distribution:"
    )

    severity_counts = (
        diagnostics_df["severity"]
        .value_counts()
        .reindex(
            [
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW",
            ],
            fill_value=0,
        )
    )

    for severity, count in (
        severity_counts.items()
    ):
        print(
            f"  {severity:<10} : {count:,}"
        )

    print(
        "\nFailure predictions:"
    )

    print(
        diagnostics_df[
            "failure_prediction"
        ].value_counts()
        .sort_index()
        .to_string()
    )

    print(
        "\nAverage predicted RUL by severity:"
    )

    print(
        diagnostics_df.groupby(
            "severity"
        )["predicted_rul"]
        .mean()
        .reindex(
            [
                "CRITICAL",
                "HIGH",
                "MEDIUM",
                "LOW",
            ]
        )
        .round(2)
        .to_string()
    )


if __name__ == "__main__":
    print("=" * 70)
    print(
        "VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM"
    )
    print(
        "SEVERITY ENGINE"
    )
    print("=" * 70)

    (
        test_df,
        classifier,
        rul_model,
        feature_columns,
        threshold,
    ) = load_artifacts()

    print(
        f"\nTest observations : "
        f"{len(test_df):,}"
    )

    print(
        f"Failure threshold : "
        f"{threshold:.2f}"
    )

    diagnostics_df = generate_diagnostics(
        test_df,
        classifier,
        rul_model,
        feature_columns,
        threshold,
    )

    print_summary(
        diagnostics_df
    )

    output_file = (
        PROCESSED_DIR
        / "diagnostics"
        / "severity_results.csv"
    )

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    diagnostics_df.to_csv(
        output_file,
        index=False,
    )

    print(
        f"\nSaved: {output_file}"
    )

    print(
        "\n" + "=" * 70
    )
    print(
        "SEVERITY ENGINE COMPLETE"
    )
    print(
        "=" * 70
    )