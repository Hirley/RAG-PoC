from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from src.llm import llm
from src.prompt import NO_CONTEXT_ANSWER, build_prompt

scenarios("answer_generation.feature")


@pytest.fixture
def context() -> dict[str, Any]:
    return {}


def fake_response(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        stop_reason="end_turn",
        content=[SimpleNamespace(type="text", text=text)],
    )


def call_llm(context: dict[str, Any], model_reply: str) -> None:
    client = MagicMock()
    client.beta.messages.create.return_value = fake_response(model_reply)
    context["client"] = client
    context["answer"] = llm(context["prompt"], client=client)


def prompt_sent_to_api(context: dict[str, Any]) -> str:
    call = context["client"].beta.messages.create.call_args
    return call.kwargs["messages"][0]["content"]


@given(parsers.parse('the formatted prompt contains the information "{information}"'))
def prompt_with_information(context: dict[str, Any], information: str) -> None:
    context["prompt"] = build_prompt(
        "When does the main server shut down?",
        [{"title": "Server maintenance window", "content": information}],
    )
    assert information in context["prompt"]


@given("the search did not return any relevant document")
def no_relevant_documents(context: dict[str, Any]) -> None:
    context["prompt"] = build_prompt("Who won the 1998 World Cup?", [])


@when("I send this prompt to the LLM via API")
def send_prompt_via_api(context: dict[str, Any]) -> None:
    call_llm(context, "The main server shuts down at 10 PM.")


@when("the prompt is sent to the LLM")
def send_prompt(context: dict[str, Any]) -> None:
    call_llm(context, NO_CONTEXT_ANSWER)


@then(parsers.parse("the returned response should mention the {moment} time"))
def response_mentions_time(context: dict[str, Any], moment: str) -> None:
    assert moment in context["answer"]
    # The prompt carrying that fact is what actually reached the API, so the
    # answer is grounded in the context rather than in the model's own priors.
    assert moment in prompt_sent_to_api(context)


@then(parsers.parse('the system should instruct the model to respond "{expected}"'))
def prompt_instructs_fallback(context: dict[str, Any], expected: str) -> None:
    assert expected == NO_CONTEXT_ANSWER
    assert expected in prompt_sent_to_api(context)
