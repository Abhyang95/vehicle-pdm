
from __future__ import annotations

import os
import time

from dotenv import load_dotenv
from google import genai
from google.genai import types


# -------------------------------------------------------------------
# Environment
# -------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-3.5-flash-lite",
)

_client: genai.Client | None = None


# -------------------------------------------------------------------
# Gemini client
# -------------------------------------------------------------------
def get_client() -> genai.Client:
    global _client

    if _client is not None:
        return _client

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Add GEMINI_API_KEY to the project .env file."
        )

    _client = genai.Client(
        api_key=GEMINI_API_KEY
    )

    return _client


# -------------------------------------------------------------------
# Retry handling
# -------------------------------------------------------------------
def is_retryable_error(exc: Exception) -> bool:
    """
    Identify temporary API failures that are safe to retry.
    """

    message = str(exc).lower()

    retryable_markers = [
        "503",
        "service_unavailable",
        "currently experiencing high demand",
        "temporarily unavailable",
        "temporarily overloaded",
        "500",
        "internal server error",
        "502",
        "bad gateway",
        "504",
        "gateway timeout",
        "429",
        "rate limit",
        "resource exhausted",
    ]

    return any(
        marker in message
        for marker in retryable_markers
    )


# -------------------------------------------------------------------
# Response extraction
# -------------------------------------------------------------------
def extract_response_text(response: object) -> str:
    """
    Extract text safely from a Gemini generate_content response.

    We first use response.text. If it is empty, inspect the
    candidate/content/parts structure as a fallback.
    """

    # ---------------------------------------------------------------
    # Preferred SDK property
    # ---------------------------------------------------------------
    text = getattr(response, "text", None)

    if isinstance(text, str) and text.strip():
        return text.strip()

    # ---------------------------------------------------------------
    # Fallback: candidates -> content -> parts -> text
    # ---------------------------------------------------------------
    candidates = getattr(
        response,
        "candidates",
        None,
    )

    if candidates:

        for candidate in candidates:

            content = getattr(
                candidate,
                "content",
                None,
            )

            if content is None:
                continue

            parts = getattr(
                content,
                "parts",
                None,
            )

            if not parts:
                continue

            collected: list[str] = []

            for part in parts:

                part_text = getattr(
                    part,
                    "text",
                    None,
                )

                if isinstance(
                    part_text,
                    str,
                ):
                    collected.append(
                        part_text
                    )

            if collected:

                answer = "\n".join(
                    collected
                ).strip()

                if answer:
                    return answer

    raise RuntimeError(
        "Gemini returned a response, but no text "
        "could be extracted from it."
    )


# -------------------------------------------------------------------
# Generate answer
# -------------------------------------------------------------------
def generate_answer(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 700,
    max_retries: int = 3,
) -> str:

    if not isinstance(
        system_prompt,
        str,
    ):
        raise ValueError(
            "system_prompt must be a string"
        )

    if not system_prompt.strip():
        raise ValueError(
            "system_prompt must not be empty"
        )

    if not isinstance(
        user_prompt,
        str,
    ):
        raise ValueError(
            "user_prompt must be a string"
        )

    if not user_prompt.strip():
        raise ValueError(
            "user_prompt must not be empty"
        )

    client = get_client()

    last_error: Exception | None = None

    # ---------------------------------------------------------------
    # Retry loop
    # ---------------------------------------------------------------
    for attempt in range(
        max_retries + 1
    ):

        try:

            if attempt > 0:

                delay = 2 ** attempt

                print(
                    f"\nGemini temporarily unavailable. "
                    f"Retrying in {delay} seconds..."
                )

                time.sleep(delay)

            # -------------------------------------------------------
            # Standard Gemini content generation
            # -------------------------------------------------------
            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt.strip(),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt.strip(),
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )

            # -------------------------------------------------------
            # Extract response
            # -------------------------------------------------------
            return extract_response_text(
                response
            )

        except Exception as exc:

            last_error = exc

            # Do not retry permanent errors.
            if not is_retryable_error(exc):
                raise

            if attempt >= max_retries:
                break

    raise RuntimeError(
        "Gemini service remained unavailable after "
        f"{max_retries + 1} attempts. "
        "Please try the assistant again later."
    ) from last_error


# -------------------------------------------------------------------
# Standalone connectivity test
# -------------------------------------------------------------------
if __name__ == "__main__":

    print("=" * 70)
    print("GEMINI LLM CONNECTIVITY TEST")
    print("=" * 70)

    print(f"\nSDK: google-genai {__import__('google.genai').genai.__version__}")
    print(f"Model: {GEMINI_MODEL}")

    try:

        answer = generate_answer(
            system_prompt=(
                "You are a concise technical assistant. "
                "Answer only the user's question."
            ),
            user_prompt=(
                "Explain RUL in predictive maintenance "
                "in one short sentence."
            ),
            temperature=0.2,
            max_output_tokens=100,
            max_retries=3,
        )

        print("\nGemini response:")
        print(answer)

        print("\n" + "=" * 70)
        print("GEMINI CONNECTION: OK")
        print("=" * 70)

    except Exception as exc:

        print("\nGEMINI CONNECTION: FAILED")
        print(f"Error type: {type(exc).__name__}")
        print(f"Error: {exc}")

        raise
