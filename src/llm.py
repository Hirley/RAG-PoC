"""LLM generation, across providers.

One completion call, three providers. Groq speaks the OpenAI wire format, so
it reuses that SDK against a different base URL rather than pulling in a third
dependency for the same request shape.

Every provider is normalised to the same three outcomes, because the pipeline
above this module cannot make a sensible decision otherwise: an answer, a
refusal, or a truncation. The SDKs disagree on how they signal the last two --
Anthropic uses stop_reason, OpenAI uses finish_reason and a separate refusal
field -- and that difference stops here.
"""

import os
from dataclasses import dataclass
from typing import Any, Callable

import anthropic
import openai

DEFAULT_PROVIDER = "anthropic"

# A ceiling, not a budget: only generated tokens are billed, so this bounds a
# runaway generation rather than the price of a normal one. It is not tight,
# because thinking is on by default on current Claude models and its tokens
# count against this same ceiling -- a value sized for the answer alone would
# truncate routinely.
DEFAULT_MAX_TOKENS = 4096


class LLMRefusalError(RuntimeError):
    """Raised when the model declines to answer the prompt."""


class LLMTruncatedError(RuntimeError):
    """Raised when the answer was cut off by the token ceiling."""


class LLMConfigurationError(RuntimeError):
    """Raised when the environment cannot be turned into a valid request."""


@dataclass(frozen=True)
class Provider:
    """Everything that differs between providers, in one place.

    `name` doubles as the prefix for its own environment variables, so adding
    a provider does not mean remembering to add a lookup anywhere else."""

    name: str
    default_model: str
    api_key_env: str
    build_client: Callable[["Provider"], Any]
    complete: Callable[[Any, str, int, str], str]
    base_url: str | None = None

    @property
    def model_env(self) -> str:
        return f"{self.name.upper()}_MODEL"


def truncated() -> LLMTruncatedError:
    """Returning the fragment would hand the caller a half-finished answer
    indistinguishable from a complete one -- worse than an error in a system
    whose whole promise is answering from a known corpus."""
    return LLMTruncatedError(
        "The answer hit the token ceiling and was cut off. Raise "
        f"LLM_MAX_TOKENS above the current {resolve_max_tokens()} and ask again."
    )


def build_anthropic_client(provider: Provider) -> anthropic.Anthropic:
    # An identity-linked API key is rejected with a 400 unless the request
    # names the workspace it acts in; a standard key needs no such header, so
    # it is omitted rather than sent blank. The value is stripped because
    # python-dotenv leaves whitespace inside quoted values and a shell export
    # strips none -- a padded id comes back as "must be a valid workspace ID",
    # a message that names the value rather than the whitespace.
    workspace_id = os.environ.get("ANTHROPIC_WORKSPACE_ID", "").strip()
    headers = {"anthropic-workspace-id": workspace_id} if workspace_id else {}

    return anthropic.Anthropic(default_headers=headers)


def complete_anthropic(
    client: anthropic.Anthropic, model: str, max_tokens: int, prompt: str
) -> str:
    response = client.beta.messages.create(
        model=model,
        max_tokens=max_tokens,
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
        raise truncated()

    # Thinking is enabled by default on current models, so the answer is not
    # necessarily the first content block.
    return "\n".join(
        block.text for block in response.content if block.type == "text"
    ).strip()


def build_openai_client(provider: Provider) -> openai.OpenAI:
    # base_url is what makes Groq work through this SDK; it is None for OpenAI
    # itself, which leaves the SDK on its own default.
    return openai.OpenAI(
        api_key=os.environ.get(provider.api_key_env),
        base_url=provider.base_url,
    )


def complete_openai(
    client: openai.OpenAI, model: str, max_tokens: int, prompt: str
) -> str:
    response = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=[{"role": "user", "content": prompt}],
    )
    choice = response.choices[0]

    # Groq omits `refusal` entirely, so this reads the attribute defensively
    # rather than assuming the full OpenAI response shape.
    refusal = getattr(choice.message, "refusal", None)
    if refusal:
        raise LLMRefusalError(f"The model declined to answer: {refusal}")

    if choice.finish_reason == "length":
        raise truncated()

    return (choice.message.content or "").strip()


FALLBACK_BETA = "server-side-fallback-2026-07-01"

PROVIDERS: dict[str, Provider] = {
    "anthropic": Provider(
        name="anthropic",
        default_model="claude-sonnet-5",
        api_key_env="ANTHROPIC_API_KEY",
        build_client=build_anthropic_client,
        complete=complete_anthropic,
    ),
    "openai": Provider(
        name="openai",
        default_model="gpt-4o",
        api_key_env="OPENAI_API_KEY",
        build_client=build_openai_client,
        complete=complete_openai,
    ),
    "groq": Provider(
        name="groq",
        default_model="llama-3.3-70b-versatile",
        api_key_env="GROQ_API_KEY",
        build_client=build_openai_client,
        complete=complete_openai,
        base_url="https://api.groq.com/openai/v1",
    ),
}


def resolve_provider(provider: str | None = None) -> Provider:
    """Resolved at call time, not import time: the CLI loads .env inside
    main(), so an import-time lookup would never see it."""
    name = (provider or os.environ.get("LLM_PROVIDER", "")).strip().lower()
    if not name:
        return PROVIDERS[DEFAULT_PROVIDER]

    if name not in PROVIDERS:
        # Listing the valid names turns a typo into a one-step fix rather than
        # a hunt through the docs.
        raise LLMConfigurationError(
            f"Unknown LLM provider {name!r}. Available: "
            f"{', '.join(sorted(PROVIDERS))}."
        )

    return PROVIDERS[name]


def resolve_model(provider: Provider, model: str | None = None) -> str:
    # Stripped for the same reason as the workspace id: python-dotenv keeps
    # whitespace inside quoted values, and a whitespace-only name is truthy --
    # it would override the default with a blank model rather than fall back.
    return model or os.environ.get(provider.model_env, "").strip() or provider.default_model


def resolve_max_tokens() -> int:
    raw = os.environ.get("LLM_MAX_TOKENS", "").strip()
    if not raw:
        return DEFAULT_MAX_TOKENS

    try:
        value = int(raw)
    except ValueError:
        # A bare ValueError would point at int(), not at the .env line that
        # actually needs fixing.
        raise LLMConfigurationError(
            f"LLM_MAX_TOKENS must be a positive integer, got {raw!r}."
        ) from None

    if value < 1:
        # The API rejects this too, but only after a round trip.
        raise LLMConfigurationError(
            f"LLM_MAX_TOKENS must be a positive integer, got {value}."
        )

    return value


def get_client(provider: Provider | None = None) -> Any:
    provider = provider or resolve_provider()
    return provider.build_client(provider)


def llm(
    prompt: str,
    client: Any | None = None,
    model: str | None = None,
    provider: str | None = None,
) -> str:
    resolved = resolve_provider(provider)
    client = client if client is not None else get_client(resolved)

    return resolved.complete(
        client,
        resolve_model(resolved, model),
        resolve_max_tokens(),
        prompt,
    )
