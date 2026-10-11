
from __future__ import annotations

import json
from typing import Any, Dict


# ============================================================
# Assistant system instructions
# ============================================================

SYSTEM_PROMPT = """
You are the Vehicle Predictive Maintenance Assistant for the
Vehicle Predictive Maintenance & Diagnostics Platform.

Your role is to explain machine-learning predictions and
telemetry-derived diagnostic signals clearly and conservatively.

IMPORTANT RULES:

1. CURRENT DIAGNOSTIC STATE
- Treat the provided CURRENT DIAGNOSTIC STATE as the authoritative
  source for the current vehicle/unit values.
- Do not invent, modify, estimate, or round diagnostic values unless
  clearly stated as an approximation.
- If a value is not provided, say that it is not available.

2. RAG KNOWLEDGE
- Use RETRIEVED KNOWLEDGE to explain project concepts, thresholds,
  maintenance priorities, subsystem interpretation, RUL interpretation,
  and safety limitations.
- Do not treat retrieved knowledge as a replacement for the current
  diagnostic state.

3. MAINTENANCE PRIORITY AND RECOMMENDED ACTION
- When answering maintenance-priority OR recommended-action questions,
  follow the project's deterministic maintenance policy.
- Maintenance priority must be evaluated from highest priority to
  lowest priority:
  1. IMMEDIATE: severity is CRITICAL OR at least two subsystems are CRITICAL.
  2. URGENT: severity is HIGH OR exactly one subsystem is CRITICAL.
  3. PLANNED: severity is MEDIUM OR at least one subsystem is DEGRADED.
  4. ROUTINE: otherwise.
- If multiple conditions appear to apply, always choose the
  highest-priority condition.
- Four CRITICAL subsystems therefore means IMMEDIATE, even if the
  overall severity is LOW.
- Do not choose ROUTINE merely because failure probability is low,
  RUL is above 30 cycles, or overall severity is LOW when the
  IMMEDIATE condition is satisfied.
- For questions asking what action should be taken when diagnostic
  indicators are degraded or critical, first determine the correct
  maintenance priority from the rules above.
- Do not contradict the deterministic maintenance policy by
  recommending ROUTINE when the current diagnostic state satisfies
  the IMMEDIATE condition.
- The current diagnostic state is authoritative for the actual
  subsystem statuses and severity.

4. FAILURE PROBABILITY VS HEALTH
- Distinguish failure probability from telemetry-derived subsystem
  health.
- A CRITICAL overall sensor health status does not automatically mean
  that the model predicts imminent failure.
- A LOW failure probability does not guarantee that the vehicle will
  not fail.
- If these signals disagree, explicitly explain the disagreement.

5. RUL
- RUL means predicted remaining useful life in operating cycles under
  the project's modeled conditions.
- RUL is a model prediction, not an exact physical lifetime.
- Do not claim that the vehicle will definitely fail after the stated
  number of cycles.

6. SUBSYSTEM HEALTH
- Subsystem groups are project-defined abstractions over sensor groups.
- They are telemetry-derived indicators.
- Never claim that a specific physical component has failed solely
  because a subsystem is CRITICAL.

7. SAFETY
- Never provide safety-critical operating instructions.
- Never claim certainty about a physical mechanical failure.
- Recommend qualified inspection or verification when appropriate.
- Do not invent repair procedures, replacement parts, service intervals,
  or manufacturer-specific instructions.

8. DATA LIMITATION
- The project uses NASA C-MAPSS simulated turbofan-engine degradation
  data represented as vehicle telemetry.
- It is not real manufacturer vehicle telemetry.
- Do not claim that the system has been trained on Mercedes, OEM,
  fleet, or real-world vehicle telemetry.

9. ANSWER STYLE
- Answer the user's question directly.
- Use concise explanations.
- When useful, structure the answer with short sections or bullets.
- Mention the relevant current values.
- Explain why the values matter.
- End with an appropriate verification/monitoring recommendation when
  the situation warrants it.

10. UNCERTAINTY
- If the available information is insufficient, explicitly say so.
- Never fill missing information with assumptions.

11. SOURCE PRIORITY
For current values:
CURRENT DIAGNOSTIC STATE > assumptions

For domain interpretation:
RETRIEVED KNOWLEDGE > unsupported assumptions

If the two sources appear inconsistent, explain the inconsistency
rather than silently choosing an interpretation.
"""


# ============================================================
# Prompt builder
# ============================================================

def build_assistant_prompt(
    question: str,
    diagnostic_state: Dict[str, Any],
    retrieved_knowledge: list[Dict[str, Any]],
) -> str:
    """
    Build the user/context prompt supplied to the LLM.

    The system instructions are kept separate from the dynamic
    diagnostic and retrieval context.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    if not isinstance(diagnostic_state, dict):
        raise ValueError("diagnostic_state must be a dictionary")

    if not isinstance(retrieved_knowledge, list):
        raise ValueError(
            "retrieved_knowledge must be a list"
        )

    diagnostic_json = json.dumps(
        diagnostic_state,
        indent=2,
        ensure_ascii=False,
    )

    lines = []

    lines.append("USER QUESTION")
    lines.append("=" * 70)
    lines.append(question.strip())
    lines.append("")

    lines.append("CURRENT DIAGNOSTIC STATE")
    lines.append("=" * 70)
    lines.append(diagnostic_json)
    lines.append("")

    lines.append("RETRIEVED KNOWLEDGE")
    lines.append("=" * 70)

    if not retrieved_knowledge:
        lines.append(
            "No relevant knowledge was retrieved."
        )
    else:
        for index, item in enumerate(
            retrieved_knowledge,
            start=1,
        ):
            source = item.get(
                "source",
                "unknown",
            )

            chunk_index = item.get(
                "chunk_index",
                "unknown",
            )

            content = item.get(
                "content",
                "",
            )

            lines.append(
                f"[Knowledge {index}] "
                f"Source: {source} | "
                f"Chunk: {chunk_index}"
            )

            lines.append(content)
            lines.append("")

    lines.append(
        "ANSWER THE USER QUESTION USING ONLY THE "
        "PROVIDED DIAGNOSTIC STATE AND RETRIEVED KNOWLEDGE."
    )

    return "\n".join(lines)


# ============================================================
# Complete prompt package
# ============================================================

def build_prompt_package(
    question: str,
    diagnostic_state: Dict[str, Any],
    retrieved_knowledge: list[Dict[str, Any]],
) -> Dict[str, str]:
    """
    Return the system prompt and dynamic user prompt separately.

    This structure maps cleanly onto Gemini/OpenAI-style chat APIs.
    """

    user_prompt = build_assistant_prompt(
        question=question,
        diagnostic_state=diagnostic_state,
        retrieved_knowledge=retrieved_knowledge,
    )

    return {
        "system_prompt": SYSTEM_PROMPT.strip(),
        "user_prompt": user_prompt,
    }


# ============================================================
# Standalone test
# ============================================================

if __name__ == "__main__":

    sample_diagnostic_state = {
        "unit": 1,
        "current_cycle": 192,
        "failure_probability": 0.0257,
        "predicted_rul": 66.02,
        "risk_status": "LOW_RISK",
        "severity": "LOW",
        "overall_sensor_health": 14.52,
        "overall_health_status": "CRITICAL",
        "subsystem_health": {
            "temperature": {
                "health": 31.62,
                "status": "CRITICAL",
            },
            "pressure": {
                "health": 0.0,
                "status": "CRITICAL",
            },
            "speed_flow": {
                "health": 7.26,
                "status": "CRITICAL",
            },
            "thermal_bleed": {
                "health": 19.18,
                "status": "CRITICAL",
            },
        },
    }

    sample_knowledge = [
        {
            "source": "subsystem_knowledge.md",
            "chunk_index": 2,
            "distance": 1.05,
            "content": (
                "Subsystem health is represented on a 0–100 scale. "
                "Values below 40 are classified as CRITICAL."
            ),
        },
        {
            "source": "safety_and_limitations.md",
            "chunk_index": 0,
            "distance": 1.08,
            "content": (
                "Subsystem health is telemetry-derived and does not "
                "prove physical component failure."
            ),
        },
    ]

    question = "Why is the vehicle health critical?"

    print("=" * 70)
    print("ASSISTANT PROMPT TEST")
    print("=" * 70)

    package = build_prompt_package(
        question=question,
        diagnostic_state=sample_diagnostic_state,
        retrieved_knowledge=sample_knowledge,
    )

    print("\nSYSTEM PROMPT")
    print("=" * 70)
    print(package["system_prompt"])

    print("\nUSER PROMPT")
    print("=" * 70)
    print(package["user_prompt"])
