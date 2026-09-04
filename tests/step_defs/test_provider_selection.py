from types import SimpleNamespace
from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

import src.llm as llm_module
from src.llm import PROVIDERS, LLMConfigurationError, LLMTruncatedError, llm

scenarios("provider_selection.feature")

ANSWER = "Deploys are frozen between 9:30 PM and 11 PM."


@pytest.fixture
def context() -> dict[str, Any]:
    return {}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Provider selection is entirely environment-driven, so the developer's
    own variables must not decide which branch these scenarios exercise."""
    for name in ("LLM_PROVIDER", "LLM_MAX_TOKENS"):
        monkeypatch.delenv(name, raising=False)
    for provider in PROVIDERS.values():
        monkeypatch.delenv(f"{provider.name.upper()}_MODEL", raising=False)
        monkeypatch.setenv(provider.api_key_env, "test-key")


@pytest.fixture(autouse=True)
def recording_sdks(
    context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both SDKs are replaced at the point src.llm constructs them, so the
    scenarios observe which wire format was chosen without any network."""
    calls: dict[str, Any] = {"finish_reason": "stop", "stop_reason": "end_turn"}

    class FakeAnthropic:
        def __init__(self, **kwargs: Any) -> None:
            calls["anthropic_kwargs"] = kwargs
            self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

        def _create(self, **kwargs: Any) -> SimpleNamespace:
            calls["sdk"] = "anthropic"
            calls["model"] = kwargs["model"]
            calls["max_tokens"] = kwargs["max_tokens"]
            return SimpleNamespace(
                stop_reason=calls["stop_reason"],
                content=[SimpleNamespace(type="text", text=ANSWER)],
            )

    class FakeOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            calls["openai_kwargs"] = kwargs
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(create=self._create)
            )

        def _create(self, **kwargs: Any) -> SimpleNamespace:
            calls["sdk"] = "openai"
            calls["model"] = kwargs["model"]
            calls["max_tokens"] = kwargs["max_tokens"]
            return SimpleNamespace(
                choices=[
                    SimpleNamespace(
                        finish_reason=calls["finish_reason"],
                        message=SimpleNamespace(content=ANSWER, refusal=None),
                    )
                ]
            )

    monkeypatch.setattr(llm_module.anthropic, "Anthropic", FakeAnthropic)
    monkeypatch.setattr(llm_module.openai, "OpenAI", FakeOpenAI)
    context["calls"] = calls


@given("no provider is configured")
def no_provider_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)


@given(parsers.parse('the environment selects the "{provider}" provider'))
def environment_selects_provider(
    monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    monkeypatch.setenv("LLM_PROVIDER", provider)


@given(parsers.parse('the "{variable}" variable names "{model}"'))
def variable_names_model(
    monkeypatch: pytest.MonkeyPatch, variable: str, model: str
) -> None:
    monkeypatch.setenv(variable, model)


@given("the provider stops generating at the token ceiling")
def provider_truncates(context: dict[str, Any]) -> None:
    context["calls"]["finish_reason"] = "length"


@when(parsers.parse('I ask "{question}" through the pipeline'))
def ask_through_pipeline(context: dict[str, Any], question: str) -> None:
    record_outcome(context, question, provider=None)


@when(parsers.parse('I ask "{question}" through the "{provider}" provider'))
def ask_through_provider(
    context: dict[str, Any], question: str, provider: str
) -> None:
    record_outcome(context, question, provider=provider)


def record_outcome(
    context: dict[str, Any], question: str, provider: str | None
) -> None:
    try:
        context["answer"] = llm(question, provider=provider)
    except Exception as error:  # noqa: BLE001 - the scenarios assert on it
        context["error"] = error


@then("the request should go to Anthropic")
def request_went_to_anthropic(context: dict[str, Any]) -> None:
    assert context["calls"]["sdk"] == "anthropic"


@then("the request should go to OpenAI")
def request_went_to_openai(context: dict[str, Any]) -> None:
    assert context["calls"]["sdk"] == "openai"


@then("the model should be the Anthropic default")
def model_is_anthropic_default(context: dict[str, Any]) -> None:
    assert context["calls"]["model"] == PROVIDERS["anthropic"].default_model


@then("the model should be the OpenAI default")
def model_is_openai_default(context: dict[str, Any]) -> None:
    assert context["calls"]["model"] == PROVIDERS["openai"].default_model


@then(parsers.parse('the model should be "{model}"'))
def model_is(context: dict[str, Any], model: str) -> None:
    assert context["calls"]["model"] == model


@then("the client should point at the Groq endpoint")
def client_points_at_groq(context: dict[str, Any]) -> None:
    assert context["calls"]["openai_kwargs"]["base_url"] == PROVIDERS["groq"].base_url


@then("the run should fail naming the providers that do exist")
def failure_names_known_providers(context: dict[str, Any]) -> None:
    error = context["error"]
    assert isinstance(error, LLMConfigurationError)
    # Listing them turns a typo into a one-step fix instead of a doc hunt.
    for name in PROVIDERS:
        assert name in str(error)


@then("the run should fail rather than return the fragment")
def failure_rather_than_fragment(context: dict[str, Any]) -> None:
    assert isinstance(context["error"], LLMTruncatedError)
    assert "answer" not in context
