```markdown
# Telemetry Subsystem Knowledge

## Important Limitation

The subsystem groupings in this project are approximate telemetry-derived groups.

They are not official manufacturer component mappings.

The NASA C-MAPSS dataset contains simulated engine sensors rather than real vehicle component telemetry. Therefore, the project uses sensor groups as diagnostic abstractions.

---

## Temperature Group

### Sensors

- sensor_2
- sensor_3
- sensor_4

### Interpretation

The temperature group summarizes behavior from the selected temperature-related telemetry signals.

A lower health score indicates greater deviation from the reference behavior used by the diagnostic system.

Potential interpretation:

- HEALTHY: behavior remains close to the reference range
- WATCH: moderate deviation
- DEGRADED: meaningful deviation
- CRITICAL: substantial deviation

The system should describe this as temperature-related telemetry deterioration rather than claiming that a specific physical temperature component has failed.

---

## Pressure Group

### Sensors

- sensor_7
- sensor_11

### Interpretation

The pressure group summarizes selected pressure-related telemetry signals.

Poor subsystem health indicates that the observed pressure-related telemetry differs substantially from the reference behavior.

The diagnostic assistant should avoid naming a specific physical component unless supporting evidence exists.

---

## Speed and Flow Group

### Sensors

- sensor_8
- sensor_9
- sensor_12
- sensor_13
- sensor_14
- sensor_15

### Interpretation

This group summarizes selected speed- and flow-related telemetry behavior.

Health degradation can indicate that the corresponding telemetry patterns differ from the expected reference behavior.

The result is a diagnostic signal, not proof of a specific mechanical fault.

---

## Thermal and Bleed Group

### Sensors

- sensor_17
- sensor_20
- sensor_21

### Interpretation

This group summarizes selected thermal/bleed-related telemetry.

Low health indicates substantial deviation from the reference behavior.

The assistant should describe this as a telemetry-derived thermal/bleed concern rather than asserting a confirmed physical failure.

---

## Health Score

Subsystem health is represented on a 0–100 scale.

Higher values indicate healthier telemetry relative to the reference.

Thresholds:

- 80–100: HEALTHY
- 60–79.99: WATCH
- 40–59.99: DEGRADED
- below 40: CRITICAL

The overall sensor health is derived from the subsystem-level health signals.

---

## Health Reference

The diagnostic system establishes a reference using early-life telemetry and robust statistics.

Sensor deviation is normalized using a robust median/MAD-based approach.

The resulting normalized deviation is converted into a health score.

Therefore, health should be interpreted as relative telemetry condition rather than an absolute physical measurement.

---

## Interpreting Multiple Subsystems

When several subsystems are degraded or critical simultaneously, the diagnostic system may increase maintenance priority.

Multiple degraded groups can indicate broader telemetry deterioration.

However, subsystem health alone should not be interpreted as confirmation of physical component failure.

The assistant should recommend inspection and verification rather than claiming certainty.
