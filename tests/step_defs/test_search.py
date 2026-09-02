from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from src.search import search

scenarios("search.feature")


@pytest.fixture
def context() -> dict[str, Any]:
    return {}


@given(parsers.parse('the database contains a document about "{topic}"'))
def database_contains_document(context: dict[str, Any], topic: str) -> None:
    client = MagicMock()
    client.search.return_value = {
        "hits": {
            "hits": [
                {"_score": 9.8, "_source": {"title": topic, "content": f"Details about {topic}."}},
                {"_score": 4.2, "_source": {"title": "Other topic", "content": "Unrelated content."}},
            ]
        }
    }
    context["client"] = client


@when(parsers.parse('I search for the question "{question}"'))
def perform_search(context: dict[str, Any], question: str) -> None:
    context["results"] = search(question, client=context["client"])


@then(parsers.parse("the system should return a list containing at most {max_count:d} documents"))
def check_max_results(context: dict[str, Any], max_count: int) -> None:
    assert len(context["results"]) <= max_count


@then(parsers.parse('the document about "{topic}" should be the first result (highest score)'))
def check_first_result(context: dict[str, Any], topic: str) -> None:
    assert context["results"][0]["title"] == topic
