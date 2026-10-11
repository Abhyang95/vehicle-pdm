from pathlib import Path

import joblib
import pandas as pd



# ============================================================
# Project configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

MODEL_DIR = (
    PROJECT_ROOT
    / "models"
)

DIAGNOSTICS_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "diagnostics"
)

DIAGNOSTICS_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

RANDOM_SEED = 42


# ============================================================
# Load models
# ============================================================

def load_model(filename):

    path = MODEL_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Model not found:\n{path}"
        )

    return joblib.load(path)

# ============================================================
# Explain a live prediction
# ============================================================

def explain_prediction(
    model,
    X: pd.DataFrame,
    top_k: int = 10,
) -> pd.DataFrame:
    """
    Generate local SHAP explanations for the supplied
    feature matrix.

    Parameters
    ----------
    model:
        Trained XGBoost failure-risk model.

    X:
        DataFrame containing the model's feature columns.

    top_k:
        Number of top contributors to return.

    Returns
    -------
    pd.DataFrame
        Top SHAP contributors for the final observation.
    """

    if X.empty:
        raise ValueError(
            "Cannot explain an empty feature matrix."
        )
    import shap
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    # SHAP/XGBoost compatibility across versions.
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Explain the latest telemetry observation.
    observation = X.iloc[-1]
    observation_shap = shap_values[-1]

    explanation = pd.DataFrame({
        "feature": X.columns,
        "feature_value": observation.values,
        "shap_value": observation_shap,
    })

    explanation["absolute_shap"] = (
        explanation["shap_value"].abs()
    )

    explanation["direction"] = (
        explanation["shap_value"]
        .apply(
            lambda value:
            "INCREASES_RISK"
            if value > 0
            else "DECREASES_RISK"
        )
    )

    explanation = (
        explanation
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return explanation.head(top_k)
# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    print("=" * 70)
    print("VEHICLE PREDICTIVE MAINTENANCE & DIAGNOSTICS PLATFORM")
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Load XGBoost failure-risk model
    # --------------------------------------------------------

    failure_bundle = load_model(
        "xgboost_failure_risk.joblib"
    )

    failure_model = failure_bundle[
        "model"
    ]

    feature_columns = failure_bundle[
        "feature_columns"
    ]

    print(
        f"\nFailure-risk model features: "
        f"{len(feature_columns)}"
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    test_path = (
        SPLIT_DIR
        / "test.parquet"
    )

    test_df = pd.read_parquet(
        test_path
    )

    X_test = test_df[
        feature_columns
    ]

    print(
        f"Test observations: "
        f"{len(X_test):,}"
    )

    # --------------------------------------------------------
    # Create SHAP TreeExplainer
    # --------------------------------------------------------

    print(
        "\nCreating SHAP TreeExplainer..."
    )

    explainer = shap.TreeExplainer(
        failure_model
    )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    print(
        "Calculating SHAP values..."
    )

    shap_values = explainer.shap_values(
        X_test
    )

    # XGBoost binary classifiers may return
    # either an array or a list depending on SHAP version.
    if isinstance(
        shap_values,
        list
    ):
        shap_values = shap_values[1]

    print(
        "SHAP calculation complete."
    )

    # --------------------------------------------------------
    # Global feature importance
    # --------------------------------------------------------

    mean_abs_shap = (
        abs(shap_values)
        .mean(axis=0)
    )

    global_importance = pd.DataFrame({
        "feature": feature_columns,
        "mean_abs_shap": mean_abs_shap,
    })

    global_importance = (
        global_importance
        .sort_values(
            "mean_abs_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    global_path = (
        DIAGNOSTICS_DIR
        / "shap_global_importance.csv"
    )

    global_importance.to_csv(
        global_path,
        index=False,
    )

    print(
        "\n" + "=" * 70
    )

    print(
        "TOP GLOBAL SHAP FEATURES"
    )

    print(
        "=" * 70
    )

    print(
        global_importance.head(20).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Select representative high-risk observation
    # --------------------------------------------------------

    probabilities = (
        failure_model
        .predict_proba(X_test)[:, 1]
    )

    high_risk_indices = (
        probabilities
        .argsort()[::-1]
    )

    representative_index = (
        high_risk_indices[0]
    )

    observation = X_test.iloc[
        representative_index
    ]

    observation_shap = shap_values[
        representative_index
    ]

    local_explanation = pd.DataFrame({
        "feature": feature_columns,
        "feature_value": observation.values,
        "shap_value": observation_shap,
    })

    local_explanation[
        "absolute_shap"
    ] = abs(
        local_explanation[
            "shap_value"
        ]
    )

    local_explanation = (
        local_explanation
        .sort_values(
            "absolute_shap",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    local_path = (
        DIAGNOSTICS_DIR
        / "shap_representative_observation.csv"
    )

    local_explanation.to_csv(
        local_path,
        index=False,
    )

    # --------------------------------------------------------
    # Print representative observation
    # --------------------------------------------------------

    actual_row = test_df.iloc[
        representative_index
    ]

    print(
        "\n" + "=" * 70
    )

    print(
        "REPRESENTATIVE HIGH-RISK OBSERVATION"
    )

    print(
        "=" * 70
    )

    print(
        f"\nUnit: "
        f"{actual_row['unit']}"
    )

    print(
        f"Cycle: "
        f"{actual_row['cycle']:.0f}"
    )

    print(
        f"Actual RUL: "
        f"{actual_row['RUL']:.0f}"
    )

    print(
        f"Failure probability: "
        f"{probabilities[representative_index]:.4f}"
    )

    print(
        "\nTOP LOCAL SHAP CONTRIBUTORS"
    )

    print(
        "----------------------------"
    )

    print(
        local_explanation.head(15).to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )

    print(
        "SHAP ANALYSIS COMPLETE"
    )

    print(
        "=" * 70
    )

    print(
        f"""
Global importance:
{global_path}

Representative explanation:
{local_path}
"""
    )