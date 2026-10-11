
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List


# ============================================================
# Project path configuration
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.rag_retriever import retrieve_knowledge


# ============================================================
# Build structured assistant context
# ============================================================

def build_assistant_context(
    question: str,
    diagnostic_state: Dict[str, Any],
    top_k: int = 4,
) -> Dict[str, Any]:
    """
    Build structured context for the future LLM assistant.

    The context deliberately separates:

    1. Current diagnostic state
    2. Retrieved domain knowledge

    This prevents the LLM from confusing live/model-generated
    diagnostic information with static RAG knowledge.
    """

    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    if not isinstance(diagnostic_state, dict):
        raise ValueError("diagnostic_state must be a dictionary")

    question = question.strip()

    retrieved = retrieve_knowledge(
        question,
        top_k=top_k,
    )

    knowledge_context: List[Dict[str, Any]] = []

    for item in retrieved:
        knowledge_context.append(
            {
                "source": item["source"],
                "chunk_index": item["chunk_index"],
                "distance": item["distance"],
                "content": item["text"],
            }
        )

    return {
        "question": question,
        "diagnostic_state": diagnostic_state,
        "retrieved_knowledge": knowledge_context,
    }


# ============================================================
# Format context for an LLM
# ============================================================

def format_context_for_llm(
    context: Dict[str, Any],
) -> str:
    """
    Convert structured assistant context into deterministic
    text suitable for an LLM prompt.
    """

    question = context["question"]
    diagnostic_state = context["diagnostic_state"]
    retrieved_knowledge = context["retrieved_knowledge"]

    lines: List[str] = []

    lines.append("USER QUESTION")
    lines.append(question)
    lines.append("")

    lines.append("CURRENT DIAGNOSTIC STATE")
    lines.append("--------------------------------")

    for key, value in diagnostic_state.items():
        lines.append(f"{key}: {value}")

    lines.append("")

    lines.append("RETRIEVED KNOWLEDGE")
    lines.append("--------------------------------")

    if not retrieved_knowledge:
        lines.append("No relevant knowledge was retrieved.")
    else:
        for index, item in enumerate(
            retrieved_knowledge,
            start=1,
        ):
            lines.append(
                f"[Knowledge {index}] "
                f"Source: {item['source']} "
                f"Chunk: {item['chunk_index']}"
            )

            lines.append(item["content"])
            lines.append("")

    return "\n".join(lines)


# ============================================================
# Convenience function
# ============================================================

def build_llm_context(
    question: str,
    diagnostic_state: Dict[str, Any],
    top_k: int = 4,
) -> str:
    """
    Build the final text context that will later be passed
    to the LLM.
    """

    context = build_assistant_context(
        question=question,
        diagnostic_state=diagnostic_state,
        top_k=top_k,
    )

    return format_context_for_llm(context)


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

    question = "Why is the vehicle health critical?"

    print("=" * 70)
    print("ASSISTANT CONTEXT TEST")
    print("=" * 70)

    context = build_assistant_context(
        question=question,
        diagnostic_state=sample_diagnostic_state,
        top_k=4,
    )

    print("\nStructured context:")
    print(context)

    print("\n" + "=" * 70)
    print("LLM CONTEXT")
    print("=" * 70)

    print(
        build_llm_context(
            question=question,
            diagnostic_state=sample_diagnostic_state,
            top_k=4,
        )
    )
