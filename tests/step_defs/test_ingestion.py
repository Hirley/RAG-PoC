from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from src.search import index_documents

scenarios("ingestion.feature")


@pytest.fixture
def context() -> dict[str, Any]:
    return {}


@given(parsers.parse("I have a list of {count:d} valid documents in JSON format"))
def documents_available(context: dict[str, Any], count: int) -> None:
    context["documents"] = [
        {"title": f"Document {i}", "content": f"Content of document {i}."}
        for i in range(count)
    ]


@given(parsers.parse('the index "{index_name}" is ready for use'))
def index_is_ready(context: dict[str, Any], index_name: str) -> None:
    client = MagicMock()
    client.bulk.return_value = {"errors": False}
    client.count.return_value = {"count": len(context["documents"])}
    context["client"] = client
    context["index_name"] = index_name


@when("I run the batch indexing function")
def run_batch_indexing(context: dict[str, Any]) -> None:
    context["indexed_count"] = index_documents(
        context["documents"],
        client=context["client"],
        index_name=context["index_name"],
    )


@then(parsers.parse("all {count:d} documents should be inserted into ElasticSearch"))
def documents_were_sent(context: dict[str, Any], count: int) -> None:
    operations = context["client"].bulk.call_args.kwargs["operations"]
    actions = [op for op in operations if "index" in op]
    payloads = [op for op in operations if "index" not in op]

    assert len(actions) == count
    assert payloads == context["documents"]
    assert all(
        action["index"] == {"_index": context["index_name"]} for action in actions
    )


@then("the database should confirm the exact record count")
def database_confirms_count(context: dict[str, Any]) -> None:
    context["client"].count.assert_called_once_with(index=context["index_name"])
    assert context["indexed_count"] == len(context["documents"])
