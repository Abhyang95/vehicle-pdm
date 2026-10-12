from __future__ import annotations

import sys
import time
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
    4. Gemini answer generation

    Logs the duration of each stage to help diagnose slow requests.
    """

    request_started = time.perf_counter()

    # ---------------------------------------------------------------
    # Validate inputs
    # ---------------------------------------------------------------
    if not isinstance(question, str):
        raise ValueError("question must be a string.")

    question = question.strip()

    if not question:
        raise ValueError("question must not be empty.")

    if not isinstance(diagnostic_state, dict):
        raise ValueError(
            "diagnostic_state must be a dictionary."
        )

    if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
        raise ValueError(
            "top_k must be a positive integer."
        )

    unit = diagnostic_state.get("unit", "unknown")
    cycle = diagnostic_state.get("current_cycle", "unknown")

    print(
        f"[assistant] Started | unit={unit} | cycle={cycle} "
        f"| top_k={top_k} | question_chars={len(question)}",
        flush=True,
    )

    try:
        # -----------------------------------------------------------
        # Step 1: Retrieve RAG knowledge and build context
        # -----------------------------------------------------------
        stage_started = time.perf_counter()

        print(
            "[assistant] RAG retrieval started",
            flush=True,
        )

        context = build_assistant_context(
            question=question,
            diagnostic_state=diagnostic_state,
            top_k=top_k,
        )

        if not isinstance(context, dict):
            raise ValueError(
                "Assistant context builder must return a dictionary."
            )

        retrieved_knowledge = (
            context.get("retrieved_knowledge") or []
        )

        if not isinstance(retrieved_knowledge, list):
            raise ValueError(
                "retrieved_knowledge must be a list."
            )

        print(
            f"[assistant] RAG retrieval completed in "
            f"{time.perf_counter() - stage_started:.2f}s "
            f"| sources={len(retrieved_knowledge)}",
            flush=True,
        )

        # -----------------------------------------------------------
        # Step 2: Build grounded prompt
        # -----------------------------------------------------------
        stage_started = time.perf_counter()

        print(
            "[assistant] Prompt construction started",
            flush=True,
        )

        prompt = build_assistant_prompt(
            question=question,
            diagnostic_state=diagnostic_state,
            retrieved_knowledge=retrieved_knowledge,
        )

        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(
                "Assistant prompt builder returned an empty prompt."
            )

        print(
            f"[assistant] Prompt construction completed in "
            f"{time.perf_counter() - stage_started:.2f}s "
            f"| prompt_chars={len(prompt)}",
            flush=True,
        )

        # -----------------------------------------------------------
        # Step 3: Generate the answer with Gemini
        # -----------------------------------------------------------
        stage_started = time.perf_counter()

        print(
            "[assistant] Gemini generation started",
            flush=True,
        )

        answer = generate_answer(
            system_prompt=SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=0.2,
            max_output_tokens=500,
            max_retries=1,
        )

        if not isinstance(answer, str) or not answer.strip():
            raise RuntimeError(
                "Gemini returned an empty answer."
            )

        print(
            f"[assistant] Gemini generation completed in "
            f"{time.perf_counter() - stage_started:.2f}s",
            flush=True,
        )

        # -----------------------------------------------------------
        # Step 4: Return structured result
        # Keep the existing API response contract intact.
        # -----------------------------------------------------------
        result = {
            "question": question,
            "answer": answer.strip(),
            "diagnostic_state": diagnostic_state,
            "retrieved_knowledge": retrieved_knowledge,
        }

        print(
            f"[assistant] Completed successfully in "
            f"{time.perf_counter() - request_started:.2f}s "
            f"| unit={unit} | cycle={cycle}",
            flush=True,
        )

        return result

    except Exception as exc:
        print(
            f"[assistant] Failed after "
            f"{time.perf_counter() - request_started:.2f}s "
            f"| unit={unit} | cycle={cycle} "
            f"| error={type(exc).__name__}: {exc}",
            flush=True,
        )

        # Preserve the original exception so the API can handle it.
        raise


def print_assistant_result(
    result: dict[str, Any],
) -> None:
    """Pretty-print the assistant answer and retrieved sources."""

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

            try:
                distance = float(item.get("distance", 0.0))
                print(f"Distance: {distance:.4f}")
            except (TypeError, ValueError):
                print("Distance: unavailable")

            content = item.get(
                "content",
                item.get("text", ""),
            )

            if isinstance(content, str) and content:
                print(content[:500])
            else:
                print("No text content available.")

    print("\n" + "=" * 70)
    print("ASSISTANT PIPELINE: OK")
    print("=" * 70)


# -------------------------------------------------------------------
# Standalone end-to-end test
# -------------------------------------------------------------------
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