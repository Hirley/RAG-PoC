import os

import anthropic

DEFAULT_MODEL = "claude-opus-5"
MAX_TOKENS = 16000
FALLBACK_BETA = "server-side-fallback-2026-07-01"


class LLMRefusalError(RuntimeError):
    """Raised when the model declines to answer the prompt."""


def get_client() -> anthropic.Anthropic:
    # An identity-linked API key is rejected with a 400 unless the request
    # names the workspace it acts in. A standard key needs no header, and
    # sending an empty one would break it, so the key is omitted when unset.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID")
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else {}

    return anthropic.Anthropic(default_headers=headers)


def llm(
    prompt: str,
    client: anthropic.Anthropic | None = None,
    model: str = DEFAULT_MODEL,
) -> str:
    client = client or get_client()
    response = client.beta.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
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

    # Thinking is enabled by default on current models, so the answer is not
    # necessarily the first content block.
    return "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()
