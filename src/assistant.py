
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

# -------------------------------------------------------------------
# Make project root importable when running:
# python src\assistant.py
# -------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.assistant_context import build_assistant_context
from src.assistant_prompt import SYSTEM_PROMPT, build_assistant_prompt
from src.llm_client import generate_answer


# -------------------------------------------------------------------
# Default diagnostic state
# Used only for standalone local testing.
# API integration passes the real diagnostic report instead.
# -------------------------------------------------------------------
DEFAULT_DIAGNOSTIC_STATE: dict[str, Any] = {
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


def answer_vehicle_question(
    question: str,
    diagnostic_state: dict[str, Any],
    top_k: int = 4,
) -> dict[str, Any]:
    """
    Answer a vehicle-maintenance question using:

    1. Current diagnostic state
    2. RAG knowledge retrieval
    3. Grounded assistant prompt
    4. Gemini LLM
    """

    if not isinstance(question, str):
        raise ValueError("question must be a string")

    question = question.strip()

    if not question:
        raise ValueError("question must not be empty")

    if not isinstance(diagnostic_state, dict):
        raise ValueError("diagnostic_state must be a dictionary")

    # ---------------------------------------------------------------
    # Step 1: Build assistant context
    # ---------------------------------------------------------------
    context = build_assistant_context(
        question=question,
        diagnostic_state=diagnostic_state,
        top_k=top_k,
    )

    # IMPORTANT:
    # Keep the RAG results exactly as returned by the context builder.
    retrieved_knowledge = context.get("retrieved_knowledge", [])

    if retrieved_knowledge is None:
        retrieved_knowledge = []

    # ---------------------------------------------------------------
    # Step 2: Build grounded prompt
    # ---------------------------------------------------------------
    prompt = build_assistant_prompt(
        question=question,
        diagnostic_state=diagnostic_state,
        retrieved_knowledge=retrieved_knowledge,
    )

    # ---------------------------------------------------------------
    # Step 3: Ask Gemini
    # ---------------------------------------------------------------
    answer = generate_answer(
    system_prompt=SYSTEM_PROMPT,
    user_prompt=prompt,
    temperature=0.2,
    max_output_tokens=500,
)

    # ---------------------------------------------------------------
    # Step 4: Return structured result
    # ---------------------------------------------------------------
    return {
        "question": question,
        "answer": answer,
        "diagnostic_state": diagnostic_state,
        "retrieved_knowledge": retrieved_knowledge,
    }


def print_assistant_result(result: dict[str, Any]) -> None:
    """Pretty-print an assistant response for local testing."""

    print("\n" + "=" * 70)
    print("VEHICLE AI ASSISTANT")
    print("=" * 70)

    print("\nQuestion:")
    print(result["question"])

    print("\nAssistant:")
    print(result["answer"])

    print("\n" + "-" * 70)
    print("RETRIEVED KNOWLEDGE")
    print("-" * 70)

    retrieved = result.get("retrieved_knowledge", [])

    if not retrieved:
        print("\nNo RAG sources returned.")
    else:
        for index, item in enumerate(retrieved, start=1):
            print(f"\n[{index}] {item.get('source', 'unknown')}")
            print(f"Chunk: {item.get('chunk_index', 'unknown')}")
            print(
                f"Distance: "
                f"{float(item.get('distance', 0.0)):.4f}"
            )

            content = item.get(
                "content",
                item.get("text", ""),
            )

            print(content[:500])

    print("\n" + "=" * 70)
    print("ASSISTANT PIPELINE: OK")
    print("=" * 70)


if __name__ == "__main__":
    question = "Why is the vehicle health critical?"

    print("=" * 70)
    print("END-TO-END VEHICLE ASSISTANT TEST")
    print("=" * 70)

    try:
        result = answer_vehicle_question(
            question=question,
            diagnostic_state=DEFAULT_DIAGNOSTIC_STATE,
            top_k=4,
        )

        print_assistant_result(result)

    except Exception as exc:
        print("\nASSISTANT PIPELINE: FAILED")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")
        raise
