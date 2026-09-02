import pytest

import src.rag as rag_module
from src.rag import rag


def test_rag_calls_the_stages_in_the_mandated_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The project guidelines fix the pipeline order: search, then prompt, then
    LLM. Asserting each stage's arguments alone would not catch a reordering."""
    order: list[str] = []

    def fake_search(query: str) -> list[dict]:
        order.append("search")
        return [{"title": "Doc", "content": "Content."}]

    def fake_build_prompt(query: str, search_results: list[dict]) -> str:
        order.append("build_prompt")
        return "prompt"

    def fake_llm(prompt: str) -> str:
        order.append("llm")
        return "answer"

    monkeypatch.setattr(rag_module, "search", fake_search)
    monkeypatch.setattr(rag_module, "build_prompt", fake_build_prompt)
    monkeypatch.setattr(rag_module, "llm", fake_llm)

    assert rag("What is RAG?") == "answer"
    assert order == ["search", "build_prompt", "llm"]


def test_rag_returns_the_llm_answer_unmodified(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rag_module, "search", lambda query: [])
    monkeypatch.setattr(
        rag_module, "build_prompt", lambda query, search_results: "prompt"
    )
    monkeypatch.setattr(rag_module, "llm", lambda prompt: "  Spaced answer.  ")

    assert rag("Any question") == "  Spaced answer.  "
