from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.llm import (
    DEFAULT_MAX_TOKENS,
    PROVIDERS,
    LLMConfigurationError,
    LLMRefusalError,
    LLMTruncatedError,
    get_client,
    llm,
)

ANTHROPIC = PROVIDERS["anthropic"]
OPENAI = PROVIDERS["openai"]


@pytest.fixture(autouse=True)
def clean_llm_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """These tests assert on defaults, so a developer's own LLM_PROVIDER or
    model variables must not leak in and make them pass for the wrong reason."""
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LLM_MAX_TOKENS", raising=False)
    for provider in PROVIDERS.values():
        monkeypatch.delenv(provider.model_env, raising=False)


def build_client(response: SimpleNamespace) -> MagicMock:
    """An Anthropic-shaped client. The default provider is Anthropic, so this
    is what `llm` reaches for unless a scenario says otherwise."""
    client = MagicMock()
    client.beta.messages.create.return_value = response
    return client


def build_openai_client(choice: SimpleNamespace) -> MagicMock:
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(choices=[choice])
    return client


def ok_client() -> MagicMock:
    return build_client(
        SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="ok")],
        )
    )


def call_and_capture(client: MagicMock) -> dict:
    llm("Any prompt", client=client)
    return client.beta.messages.create.call_args.kwargs


# --- Anthropic response handling ------------------------------------------


def test_llm_ignores_non_text_blocks() -> None:
    """Thinking is on by default on current models, so the first content block
    is not necessarily the answer."""
    client = build_client(
        SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="thinking", thinking="Let me reason first."),
                SimpleNamespace(type="text", text="The answer is 42."),
            ],
        )
    )

    assert llm("Any prompt", client=client) == "The answer is 42."


def test_llm_joins_multiple_text_blocks() -> None:
    client = build_client(
        SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="text", text="First part."),
                SimpleNamespace(type="text", text="Second part."),
            ],
        )
    )

    assert llm("Any prompt", client=client) == "First part.\nSecond part."


def test_llm_raises_on_refusal() -> None:
    client = build_client(
        SimpleNamespace(
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="cyber", explanation="Declined."),
            content=[],
        )
    )

    with pytest.raises(LLMRefusalError):
        llm("Any prompt", client=client)


def test_llm_sends_the_prompt_as_a_single_user_message() -> None:
    kwargs = call_and_capture(ok_client())

    assert kwargs["model"] == ANTHROPIC.default_model
    assert kwargs["messages"] == [{"role": "user", "content": "Any prompt"}]


def test_a_truncated_answer_is_not_returned_as_if_complete() -> None:
    """stop_reason "max_tokens" means the model was cut off mid-answer. In a
    RAG system a confidently half-finished answer is worse than an error, and
    the caller cannot tell one from the other by looking at the text."""
    client = build_client(
        SimpleNamespace(
            stop_reason="max_tokens",
            content=[SimpleNamespace(type="text", text="Deploys are frozen bet")],
        )
    )

    with pytest.raises(LLMTruncatedError, match="LLM_MAX_TOKENS"):
        llm("Any prompt", client=client)


# --- OpenAI response handling ---------------------------------------------


def test_openai_returns_the_message_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    client = build_openai_client(
        SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="  The answer is 42.  ", refusal=None),
        )
    )

    assert llm("Any prompt", client=client) == "The answer is 42."


def test_openai_raises_on_a_refusal(monkeypatch: pytest.MonkeyPatch) -> None:
    """OpenAI signals a refusal in its own field rather than in finish_reason,
    so the two providers cannot share one check."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    client = build_openai_client(
        SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content=None, refusal="I cannot help with that."),
        )
    )

    with pytest.raises(LLMRefusalError, match="cannot help"):
        llm("Any prompt", client=client)


def test_openai_raises_on_a_truncated_answer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """finish_reason "length" is OpenAI's stop_reason "max_tokens"; both must
    reach the caller as the same error or the pipeline cannot react to it."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    client = build_openai_client(
        SimpleNamespace(
            finish_reason="length",
            message=SimpleNamespace(content="Deploys are frozen bet", refusal=None),
        )
    )

    with pytest.raises(LLMTruncatedError):
        llm("Any prompt", client=client)


def test_openai_tolerates_a_missing_refusal_field(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Groq speaks the OpenAI format but omits `refusal`, so reading it must
    not turn a perfectly good answer into an AttributeError."""
    monkeypatch.setenv("LLM_PROVIDER", "groq")
    client = build_openai_client(
        SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="The answer is 42."),
        )
    )

    assert llm("Any prompt", client=client) == "The answer is 42."


# --- Provider and model resolution ----------------------------------------


def test_model_comes_from_the_provider_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    assert call_and_capture(ok_client())["model"] == "claude-haiku-4-5-20251001"


def test_a_model_variable_belongs_to_one_provider_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """OPENAI_MODEL must not steer an Anthropic call, or switching provider
    would silently carry the previous provider's model name along."""
    monkeypatch.setenv("OPENAI_MODEL", "gpt-4o-mini")

    assert call_and_capture(ok_client())["model"] == ANTHROPIC.default_model


def test_model_falls_back_to_the_provider_default() -> None:
    assert call_and_capture(ok_client())["model"] == ANTHROPIC.default_model


def test_an_explicit_model_argument_wins_over_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The env var configures a deployment; the argument overrides one call."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    client = ok_client()

    llm("Any prompt", client=client, model="claude-opus-5")

    assert client.beta.messages.create.call_args.kwargs["model"] == "claude-opus-5"


def test_model_is_stripped_of_surrounding_whitespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """python-dotenv keeps whitespace inside quoted values, so a padded model
    name would be sent to the API and rejected as unknown."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "  claude-opus-5  ")

    assert call_and_capture(ok_client())["model"] == "claude-opus-5"


def test_a_whitespace_only_model_falls_back_to_the_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Truthy but useless: without stripping this would override the default
    with a blank model name rather than fall back to it."""
    monkeypatch.setenv("ANTHROPIC_MODEL", "   ")

    assert call_and_capture(ok_client())["model"] == ANTHROPIC.default_model


def test_an_unknown_provider_is_reported_with_the_known_ones() -> None:
    with pytest.raises(LLMConfigurationError, match="anthropic, groq, openai"):
        llm("Any prompt", client=ok_client(), provider="gemini")


def test_the_provider_name_is_case_insensitive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LLM_PROVIDER=OpenAI is the same request as LLM_PROVIDER=openai, and
    failing on the capital would be a needless papercut."""
    monkeypatch.setenv("LLM_PROVIDER", "  OpenAI  ")
    client = build_openai_client(
        SimpleNamespace(
            finish_reason="stop",
            message=SimpleNamespace(content="ok", refusal=None),
        )
    )

    assert llm("Any prompt", client=client) == "ok"


# --- Token ceiling ---------------------------------------------------------


def test_max_tokens_comes_from_the_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_TOKENS", "256")

    assert call_and_capture(ok_client())["max_tokens"] == 256


def test_max_tokens_falls_back_to_the_default() -> None:
    assert call_and_capture(ok_client())["max_tokens"] == DEFAULT_MAX_TOKENS


def test_a_non_numeric_max_tokens_is_reported_as_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare ValueError traceback would point at int(), not at the .env line
    that actually needs fixing."""
    monkeypatch.setenv("LLM_MAX_TOKENS", "lots")

    with pytest.raises(LLMConfigurationError, match="LLM_MAX_TOKENS"):
        llm("Any prompt", client=ok_client())


def test_a_non_positive_max_tokens_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_MAX_TOKENS", "0")

    with pytest.raises(LLMConfigurationError, match="LLM_MAX_TOKENS"):
        llm("Any prompt", client=ok_client())


# --- Anthropic workspace header -------------------------------------------


def test_get_client_sends_the_workspace_header_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity-linked API keys are rejected with a 400 unless the request
    names the workspace it acts in."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_123")

    client = get_client()

    assert client.default_headers["anthropic-workspace-id"] == "wrkspc_123"


def test_get_client_omits_the_workspace_header_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A standard API key needs no workspace, so the header is omitted rather
    than sent blank -- an empty value is not a value the API can act on."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)

    client = get_client()

    assert "anthropic-workspace-id" not in client.default_headers


def test_get_client_strips_whitespace_around_the_workspace_id(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """python-dotenv strips whitespace only from unquoted values, and a shell
    export strips none, so a padded id reaches the header and the API rejects
    it with a message that names the value rather than the whitespace."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "  wrkspc_123  ")

    client = get_client()

    assert client.default_headers["anthropic-workspace-id"] == "wrkspc_123"


def test_get_client_treats_a_whitespace_only_workspace_id_as_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "   ")

    client = get_client()

    assert "anthropic-workspace-id" not in client.default_headers


def test_get_client_points_groq_at_its_own_endpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The base URL is the whole reason Groq can reuse the OpenAI SDK."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk-test")
    monkeypatch.setenv("LLM_PROVIDER", "groq")

    client = get_client()

    assert str(client.base_url).startswith(PROVIDERS["groq"].base_url)
