from typing import Any

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

import src.rag as rag_module
from src.prompt import NO_CONTEXT_ANSWER, build_prompt
from src.rag import rag

scenarios("rag_orchestration.feature")

LLM_ANSWER = "RAG combines retrieval with generation."


@pytest.fixture
def context() -> dict[str, Any]:
    return {}


def install_stages(
    context: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    documents: list[dict],
) -> None:
    """Replace the I/O stages with stubs, but keep the real build_prompt behind
    a spy so the assertions run against the prompt the system actually builds."""
    calls: dict[str, Any] = {"documents": documents}

    def fake_search(query: str) -> list[dict]:
        calls["search_query"] = query
        return documents

    def spying_build_prompt(query: str, search_results: list[dict]) -> str:
        calls["prompt_query"] = query
        calls["prompt_results"] = search_results
        calls["prompt"] = build_prompt(query, search_results)
        return calls["prompt"]

    def fake_llm(prompt: str, provider: str | None = None) -> str:
        calls["llm_prompt"] = prompt
        return LLM_ANSWER

    monkeypatch.setattr(rag_module, "search", fake_search)
    monkeypatch.setattr(rag_module, "build_prompt", spying_build_prompt)
    monkeypatch.setattr(rag_module, "llm", fake_llm)
    context["calls"] = calls


@given(parsers.parse("the search stage returns {count:d} relevant documents"))
def search_returns_documents(
    context: dict[str, Any], monkeypatch: pytest.MonkeyPatch, count: int
) -> None:
    documents = [
        {"title": f"Document {i}", "content": f"Relevant content {i} about RAG."}
        for i in range(count)
    ]
    install_stages(context, monkeypatch, documents)


@given("the search stage returns no documents")
def search_returns_nothing(
    context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    install_stages(context, monkeypatch, [])


@when(parsers.parse('I call the rag function with the question "{question}"'))
def call_rag(context: dict[str, Any], question: str) -> None:
    context["question"] = question
    context["answer"] = rag(question)


@then("the search stage should be called with the original question")
def search_called_with_question(context: dict[str, Any]) -> None:
    assert context["calls"]["search_query"] == context["question"]


@then("the prompt stage should receive the search results")
def prompt_received_results(context: dict[str, Any]) -> None:
    assert context["calls"]["prompt_query"] == context["question"]
    assert context["calls"]["prompt_results"] == context["calls"]["documents"]


@then("the LLM stage should receive the built prompt")
def llm_received_prompt(context: dict[str, Any]) -> None:
    assert context["calls"]["llm_prompt"] == context["calls"]["prompt"]


@then("the returned answer should be the LLM response")
def answer_is_llm_response(context: dict[str, Any]) -> None:
    assert context["answer"] == LLM_ANSWER


@then("the prompt stage should still be called with an empty result list")
def prompt_called_with_empty_list(context: dict[str, Any]) -> None:
    assert context["calls"]["prompt_results"] == []


@then("the LLM stage should receive a prompt carrying the fallback instruction")
def prompt_carries_fallback_instruction(context: dict[str, Any]) -> None:
    assert NO_CONTEXT_ANSWER in context["calls"]["llm_prompt"]
