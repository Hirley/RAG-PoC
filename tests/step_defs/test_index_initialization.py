from typing import Any
from unittest.mock import MagicMock

import pytest
from pytest_bdd import given, parsers, scenarios, then, when

from src.search import ensure_index

scenarios("index_initialization.feature")

INDEX_NAME = "rag_docs"


@pytest.fixture
def context() -> dict[str, Any]:
    return {}


@given("ElasticSearch is running")
def elasticsearch_running(context: dict[str, Any]) -> None:
    context["client"] = MagicMock()


@given(parsers.parse('the index "{index_name}" does not exist'))
def index_does_not_exist(context: dict[str, Any], index_name: str) -> None:
    context["client"].indices.exists.return_value = False


@given(parsers.parse('the index "{index_name}" already exists'))
def index_already_exists(context: dict[str, Any], index_name: str) -> None:
    context["client"].indices.exists.return_value = True


@when("the initialization module is executed")
def run_initialization(context: dict[str, Any]) -> None:
    try:
        ensure_index(context["client"], INDEX_NAME)
    except Exception as exc:
        context["exception"] = exc
    else:
        context["exception"] = None


@then(parsers.parse('the system should successfully create the index "{index_name}"'))
def index_created(context: dict[str, Any], index_name: str) -> None:
    context["client"].indices.create.assert_called_once_with(index=index_name)


@then("no exception should be raised")
def no_exception(context: dict[str, Any]) -> None:
    assert context["exception"] is None


@then("the system should detect that the index already exists")
def index_detected(context: dict[str, Any]) -> None:
    context["client"].indices.exists.assert_called_once_with(index=INDEX_NAME)


@then("it should not attempt to recreate it or delete existing data")
def no_recreate(context: dict[str, Any]) -> None:
    context["client"].indices.create.assert_not_called()
    context["client"].indices.delete.assert_not_called()
