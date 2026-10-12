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

# Gemini client is initialized lazily on the first request.
_client: genai.Client | None = None


# -------------------------------------------------------------------
# Gemini client
# -------------------------------------------------------------------
def get_client() -> genai.Client:
    """Create and cache the Gemini client."""

    global _client

    if _client is not None:
        return _client

    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. "
            "Configure GEMINI_API_KEY in your local .env file "
            "or in your deployment environment variables."
        )

    # IMPORTANT:
    # This block must be outside the missing-key condition.
    _client = genai.Client(
        api_key=GEMINI_API_KEY,
        http_options=types.HttpOptions(
            timeout=25_000,  # Request timeout in milliseconds.
        ),
    )

    return _client


# -------------------------------------------------------------------
# Retry handling
# -------------------------------------------------------------------
def is_retryable_error(exc: Exception) -> bool:
    """Identify temporary API errors that may succeed on retry."""

    message = str(exc).lower()

    retryable_markers = (
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
    )

    return any(
        marker in message
        for marker in retryable_markers
    )


# -------------------------------------------------------------------
# Response extraction
# -------------------------------------------------------------------
def extract_response_text(response: object) -> str:
    """
    Extract generated text from a Gemini response.

    Prefer response.text, then fall back to candidate content parts.
    """

    # Preferred SDK property.
    response_text = getattr(response, "text", None)

    if isinstance(response_text, str) and response_text.strip():
        return response_text.strip()

    # Fallback: candidates -> content -> parts -> text.
    candidates = getattr(response, "candidates", None)

    if candidates:
        for candidate in candidates:
            content = getattr(candidate, "content", None)

            if content is None:
                continue

            parts = getattr(content, "parts", None)

            if not parts:
                continue

            collected: list[str] = []

            for part in parts:
                part_text = getattr(part, "text", None)

                if isinstance(part_text, str) and part_text.strip():
                    collected.append(part_text.strip())

            if collected:
                answer = "\n".join(collected).strip()

                if answer:
                    return answer

    raise RuntimeError(
        "Gemini returned a response, but no text could be extracted. "
        "The response may have been blocked or contained no text."
    )


# -------------------------------------------------------------------
# Generate answer
# -------------------------------------------------------------------
def generate_answer(
    system_prompt: str,
    user_prompt: str,
    temperature: float = 0.2,
    max_output_tokens: int = 700,
    max_retries: int = 1,
) -> str:
    """
    Generate a Gemini response.

    max_retries=1 means at most two attempts:
    one initial request and one retry for a recognized transient error.
    """

    if not isinstance(system_prompt, str) or not system_prompt.strip():
        raise ValueError(
            "system_prompt must be a non-empty string."
        )

    if not isinstance(user_prompt, str) or not user_prompt.strip():
        raise ValueError(
            "user_prompt must be a non-empty string."
        )

    if not isinstance(max_retries, int) or max_retries < 0:
        raise ValueError(
            "max_retries must be a non-negative integer."
        )

    client = get_client()
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        if attempt > 0:
            # Short backoff before retrying.
            delay = 2 ** attempt

            print(
                f"[gemini] Temporary API failure. "
                f"Retrying in {delay}s "
                f"(attempt {attempt + 1}/{max_retries + 1}).",
                flush=True,
            )

            time.sleep(delay)

        try:
            started = time.perf_counter()

            print(
                f"[gemini] Request started | model={GEMINI_MODEL} "
                f"| attempt={attempt + 1}/{max_retries + 1}",
                flush=True,
            )

            response = client.models.generate_content(
                model=GEMINI_MODEL,
                contents=user_prompt.strip(),
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt.strip(),
                    temperature=temperature,
                    max_output_tokens=max_output_tokens,
                ),
            )

            answer = extract_response_text(response)

            print(
                f"[gemini] Request completed in "
                f"{time.perf_counter() - started:.2f}s",
                flush=True,
            )

            return answer

        except Exception as exc:
            last_error = exc

            print(
                f"[gemini] Request failed | "
                f"attempt={attempt + 1}/{max_retries + 1} "
                f"| error={type(exc).__name__}: {exc}",
                flush=True,
            )

            # Retry only recognized temporary service failures.
            if not is_retryable_error(exc):
                raise

            if attempt >= max_retries:
                break

    raise RuntimeError(
        f"Gemini remained unavailable after {max_retries + 1} "
        "attempt(s). Please try again later."
    ) from last_error


# -------------------------------------------------------------------
# Standalone connectivity test
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 70)
    print("GEMINI LLM CONNECTIVITY TEST")
    print("=" * 70)

    try:
        sdk_version = __import__(
            "google.genai",
            fromlist=["__version__"],
        ).__version__

        print(f"\nSDK: google-genai {sdk_version}")
        print(f"Model: {GEMINI_MODEL}")

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
            max_retries=1,
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