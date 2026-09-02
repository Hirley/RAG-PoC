from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from src.prompt import PROMPT_TEMPLATE, build_prompt

scenarios("prompt_formatting.feature")


@pytest.fixture
def context() -> dict[str, Any]:
    return {}


# Plain string step: the braces below are literal template tags, so this must
# not go through parsers.parse, which would read them as placeholders.
@given("the prompt template requires the {context} and {question} tags")
def template_requires_tags(context: dict[str, Any]) -> None:
    assert "{context}" in PROMPT_TEMPLATE
    assert "{question}" in PROMPT_TEMPLATE


@given(parsers.parse("the search returned {count:d} text snippets"))
def search_returned_snippets(context: dict[str, Any], count: int) -> None:
    context["snippets"] = [
        {"title": f"Snippet {i}", "content": f"Text snippet number {i} about RAG."}
        for i in range(count)
    ]


@when(
    parsers.parse(
        'I call the prompt-building function with the question "{question}"'
    )
)
def call_build_prompt(context: dict[str, Any], question: str) -> None:
    context["question"] = question
    context["prompt"] = build_prompt(question, context["snippets"])


@then(
    parsers.parse(
        "the resulting string should contain the {count:d} concatenated text snippets"
    )
)
def prompt_contains_snippets(context: dict[str, Any], count: int) -> None:
    assert len(context["snippets"]) == count
    for snippet in context["snippets"]:
        assert snippet["content"] in context["prompt"]


@then("the resulting string should contain the original question")
def prompt_contains_question(context: dict[str, Any]) -> None:
    assert context["question"] in context["prompt"]
