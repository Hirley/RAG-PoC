import os

import anthropic

# Sonnet is the default because a RAG answer is a short summary of context the
# retrieval stage already found -- the reasoning is shallow and the output is
# brief. Set ANTHROPIC_MODEL to reach for a stronger model when the corpus
# demands it; the model is the real cost lever here.
DEFAULT_MODEL = "claude-sonnet-5"

# A ceiling, not a budget: only generated tokens are billed, so this bounds a
# runaway generation rather than the price of a normal one. It is not tight,
# because thinking is on by default and its tokens count against this same
# ceiling -- a value sized for the answer alone would truncate routinely.
DEFAULT_MAX_TOKENS = 4096

FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusalError(RuntimeError):
    """Raised when the model declines to answer the prompt."""


class LLMConfigurationError(RuntimeError):
    """Raised when an ANTHROPIC_* environment variable cannot be used."""


class LLMTruncatedError(RuntimeError):
    """Raised when the answer was cut off by the token ceiling."""


def get_client() -> anthropic.Anthropic:
    # An identity-linked API key is rejected with a 400 unless the request
    # names the workspace it acts in. A standard key needs no header, and
    # sending an empty one would break it, so the key is omitted when unset.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else {}

    return anthropic.Anthropic(default_headers=headers)


def resolve_model(model: str | None = None) -> str:
    """Resolved at call time, not import time: the CLI loads .env inside
    main(), so an import-time lookup would never see it."""
    # Stripped for the same reason as the workspace id: python-dotenv keeps
    # whitespace inside quoted values, and a whitespace-only name is truthy --
    # it would override the default with a blank model rather than fall back.
    return model or os.environ.get("ANTHROPIC_MODEL", "").strip() or DEFAULT_MODEL


def resolve_max_tokens() -> int:
    raw = os.environ.get("ANTHROPIC_MAX_TOKENS", "").strip()
    if not raw:
        return DEFAULT_MAX_TOKENS

    try:
        value = int(raw)
    except ValueError:
        # A bare ValueError would point at int(), not at the .env line that
        # actually needs fixing.
        raise LLMConfigurationError(
            f"ANTHROPIC_MAX_TOKENS must be a positive integer, got {raw!r}."
        ) from None

    if value < 1:
        # The API rejects this too, but only after a billable round trip.
        raise LLMConfigurationError(
            f"ANTHROPIC_MAX_TOKENS must be a positive integer, got {value}."
        )

    return value


def llm(
    prompt: str,
    client: anthropic.Anthropic | None = None,
    model: str | None = None,
) -> str:
    client = client or get_client()
    response = client.beta.messages.create(
        model=resolve_model(model),
        max_tokens=resolve_max_tokens(),
        betas=[FALLBACK_BETA],
        fallbacks="default",
        messages=[{"role": "user", "content": prompt}],
    )

    if response.stop_reason == "refusal":
        details = getattr(response, "stop_details", None)
        raise LLMRefusalError(
            f"The model declined to answer (category: "
            f"{getattr(details, 'category', 'unknown')})."
        )

    if response.stop_reason == "max_tokens":
        # Returning the fragment would hand the caller a half-finished answer
        # indistinguishable from a complete one -- worse than an error in a
        # system whose whole promise is answering from a known corpus.
        raise LLMTruncatedError(
            "The answer hit the token ceiling and was cut off. Raise "
            f"ANTHROPIC_MAX_TOKENS above the current {resolve_max_tokens()} "
            "and ask again."
        )

    # Thinking is enabled by default on current models, so the answer is not
    # necessarily the first content block.
    return "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
