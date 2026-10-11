```markdown
# Maintenance Guidelines

## Purpose

Maintenance recommendations combine predicted RUL, model severity, subsystem health, and model explanations.

The recommendation engine provides an operational priority for the dashboard.

It does not replace a manufacturer's maintenance schedule or qualified technician inspection.

---

## Maintenance Priority

### IMMEDIATE

Immediate priority is assigned when:

- severity is CRITICAL, or
- at least two subsystems are CRITICAL

Recommended interpretation:

Perform diagnostic inspection as soon as practical and investigate the signals responsible for the elevated concern.

---

## URGENT

Urgent priority is assigned when:

- severity is HIGH, or
- at least one subsystem is CRITICAL

Recommended interpretation:

Prioritize diagnostic inspection and investigate the affected telemetry patterns.

---

## PLANNED

Planned priority is assigned when:

- severity is MEDIUM, or
- at least one subsystem is DEGRADED

Recommended interpretation:

Schedule inspection or maintenance attention and continue monitoring the vehicle.

---

## ROUTINE

Routine priority is assigned when:

- severity is LOW, and
- there are no degraded or critical subsystem conditions requiring escalation.

Recommended interpretation:

Continue normal monitoring and scheduled maintenance.

---

## RUL Interpretation

### More than 30 cycles

The model does not identify a near-term RUL threshold concern.

Continue monitoring alongside the other diagnostic signals.

### 16–30 cycles

The predicted RUL is entering the project's defined failure-risk region.

Increase monitoring and consider planned diagnostic attention.

### 8–15 cycles

The predicted RUL is low.

Prioritize diagnostic investigation.

### 0–7 cycles

The predicted RUL is critically low.

Treat the prediction as a high-priority diagnostic signal and verify the condition through appropriate inspection.

---

## SHAP Interpretation

SHAP values describe how model features contributed to an individual prediction.

A positive SHAP contribution for the failure-risk model increases the model's predicted risk relative to the model baseline.

A negative contribution decreases predicted risk.

The magnitude represents the relative contribution of that feature to the model output.

SHAP explanations should be presented as model explanations, not causal proof.

For example, a high contribution from a rolling sensor feature means that the model associated that feature with the prediction. It does not prove that the physical sensor or component is defective.

---

## Maintenance Recommendation Principles

The assistant should:

1. State the current model prediction.
2. Identify the strongest supporting diagnostic signals.
3. Identify affected telemetry-derived subsystems.
4. Explain the maintenance priority.
5. Recommend verification or inspection.
6. Avoid claiming certainty about physical component failure.
7. Avoid inventing maintenance procedures not present in the knowledge base.

---

## Human Verification

All recommendations should be treated as decision support.

Qualified personnel should verify the physical condition of the vehicle or equipment before performing maintenance.

The system does not provide safety-critical operating instructions.
