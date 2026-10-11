```markdown
# Predictive Maintenance Fundamentals

## Purpose

Predictive maintenance uses telemetry, machine-learning models, and diagnostic signals to estimate equipment condition and identify situations that may require inspection or maintenance.

The Vehicle Predictive Maintenance & Diagnostics Platform demonstrates this workflow using NASA C-MAPSS simulated turbofan-engine degradation data represented as vehicle telemetry.

The dataset is simulated and does not represent real-world vehicle or manufacturer telemetry.

---

## Remaining Useful Life

Remaining Useful Life (RUL) estimates the number of operating cycles remaining before the modeled end-of-life condition.

A lower RUL generally indicates greater degradation and a shorter expected operating horizon.

RUL is a model prediction and should not be interpreted as an exact physical lifetime.

Within this project, RUL is capped at 125 cycles for the training target.

---

## Failure Risk

Failure risk is a classification probability produced by the predictive model.

The project uses the following operational threshold:

- RUL <= 30 cycles: failure-risk target

The deployed model produces a probability between 0 and 1.

The probability should be interpreted as a model-generated risk signal rather than a guarantee that a physical failure will or will not occur.

---

## Risk Status

The application converts failure probability into an operational risk status.

### LOW_RISK

Failure probability is below 35%.

This indicates that the current model does not identify a high immediate failure-risk signal.

### WATCH

Failure probability is between 35% and 65%.

The vehicle should receive closer monitoring and diagnostic attention.

### HIGH_RISK

Failure probability is at least 65%.

The vehicle requires elevated diagnostic and maintenance attention.

---

## Severity

The application combines failure probability and predicted RUL when determining severity.

### LOW

Used when failure probability is below 35% and predicted RUL is above 30 cycles.

### MEDIUM

Used when failure probability reaches at least 35% or predicted RUL reaches 30 cycles or below.

### HIGH

Used when failure probability reaches at least 65% or predicted RUL reaches 15 cycles or below.

### CRITICAL

Used when failure probability reaches at least 90% or predicted RUL reaches 7 cycles or below.

Severity is an application-level interpretation of model outputs and does not represent an OEM service classification.

---

## Model Interpretation

The project uses machine-learning models for two main predictions:

1. Failure-risk classification
2. Remaining Useful Life regression

XGBoost is the selected model for both production inference paths because it provided the strongest validation/test performance among the evaluated models.

The models use engineered telemetry features including rolling statistics, trends, slopes, and cycle-related features.

---

## Why RUL and Sensor Health Can Disagree

Predicted RUL, failure probability, and telemetry-derived subsystem health represent different signals.

A vehicle may have:

- low predicted failure probability,
- moderate or high predicted RUL,
- but poor current sensor-health scores.

This is possible because subsystem health is derived from sensor deviation relative to reference behavior, while the predictive models use learned relationships across multiple engineered features.

The signals should therefore be considered together rather than assuming that one metric must always agree with another.

---

## Diagnostic Workflow

The platform follows this workflow:

Telemetry
→ preprocessing
→ feature engineering
→ failure-risk prediction
→ RUL prediction
→ SHAP explanation
→ subsystem health
→ diagnostic report
→ maintenance recommendation

The diagnostic layer combines these signals into an interpretable operational summary.

---

## Replay Mode

Replay mode accepts historical telemetry in CSV format.

The system reconstructs the telemetry history from the beginning of the selected vehicle/unit up to the requested cycle.

This is necessary because rolling features and trend features require chronological history.

Replay mode is intended for demonstration, testing, and historical analysis.

It is not an IoT streaming implementation.

---

## Important Interpretation Rule

Model outputs should support maintenance decision-making rather than replace qualified human inspection.

A prediction indicates what the model believes based on the available telemetry and learned patterns.

It does not prove that a specific physical component has failed.
