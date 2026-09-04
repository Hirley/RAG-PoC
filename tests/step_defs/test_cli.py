import json
from pathlib import Path
from typing import Any

import elasticsearch
import pytest
from pytest_bdd import given, parsers, scenarios, then, when

import src.cli as cli_module
from src.cli import main

scenarios("cli.feature")

ANSWER = "Deploys happen at 10 PM."
DOCUMENTS = [
    {"title": "Deployment schedule", "content": "The main server shuts down at 10 PM."},
    {"title": "Onboarding", "content": "New engineers get the wiki on day one."},
    {"title": "Support", "content": "The on-call rotation changes every Monday."},
]


@pytest.fixture
def context() -> dict[str, Any]:
    return {}


@pytest.fixture(autouse=True)
def isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI reads its configuration from the environment, and load_dotenv
    would pull the developer's real .env into the test run. Both are pinned
    here so the scenarios describe the state they actually set up."""
    monkeypatch.setattr(cli_module, "load_dotenv", lambda *a, **k: False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ELASTICSEARCH_URL", "http://localhost:9200")


def run_cli(context: dict[str, Any], capsys: pytest.CaptureFixture[str], argv: list[str]) -> None:
    context["exit_code"] = main(argv)
    captured = capsys.readouterr()
    context["stdout"] = captured.out
    context["stderr"] = captured.err


@given("the API key is configured")
def api_key_present(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")


@given("the API key is missing")
def api_key_absent(
    context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    stub_pipeline(context, monkeypatch)


@given(parsers.parse("a JSON file holding {count:d} documents"))
def json_file_with_documents(
    context: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    count: int,
) -> None:
    path = tmp_path / "documents.json"
    path.write_text(json.dumps(DOCUMENTS[:count]), encoding="utf-8")
    context["path"] = path

    indexed: dict[str, Any] = {}

    def fake_index_documents(documents, client=None, index_name=""):
        indexed["documents"] = documents
        indexed["index_name"] = index_name
        return len(documents)

    monkeypatch.setattr(cli_module, "get_client", lambda: object())
    monkeypatch.setattr(cli_module, "ensure_index", lambda client, index_name: None)
    monkeypatch.setattr(cli_module, "index_documents", fake_index_documents)
    context["indexed"] = indexed


@given(parsers.parse('the search stage returns a document titled "{title}"'))
def search_returns_document(
    context: dict[str, Any], monkeypatch: pytest.MonkeyPatch, title: str
) -> None:
    stub_pipeline(context, monkeypatch)
    monkeypatch.setattr(
        cli_module,
        "search",
        lambda query, client=None, index_name="", size=0: [
            {"title": title, "content": "The main server shuts down at 10 PM."}
        ],
    )


@given(parsers.parse('the RAG pipeline answers "{answer}"'))
def pipeline_answers(
    context: dict[str, Any], monkeypatch: pytest.MonkeyPatch, answer: str
) -> None:
    stub_pipeline(context, monkeypatch, answer=answer)


@given("ElasticSearch is unreachable")
def elasticsearch_unreachable(
    context: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def failing_rag(query: str) -> str:
        raise elasticsearch.ConnectionError("Connection refused")

    monkeypatch.setattr(cli_module, "rag", failing_rag)


def stub_pipeline(
    context: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    answer: str = ANSWER,
) -> None:
    """Replace the stages that reach the network, recording what they were
    called with so the scenarios can assert the CLI wired them correctly."""
    calls: dict[str, Any] = {}

    def fake_rag(query: str) -> str:
        calls["rag_query"] = query
        return answer

    monkeypatch.setattr(cli_module, "rag", fake_rag)
    context["calls"] = calls


@when(parsers.parse('I run the CLI with "{arguments}"'))
def run_with_arguments(
    context: dict[str, Any], capsys: pytest.CaptureFixture[str], arguments: str
) -> None:
    run_cli(context, capsys, arguments.split())


@when(parsers.parse('I run the CLI with "{command}" against that file'))
def run_against_file(
    context: dict[str, Any], capsys: pytest.CaptureFixture[str], command: str
) -> None:
    run_cli(context, capsys, [command, str(context["path"])])


@then(parsers.parse("the exit code should be {code:d}"))
def exit_code_is(context: dict[str, Any], code: int) -> None:
    assert context["exit_code"] == code, context["stderr"]


@then("the documents should have been sent to the index")
def documents_were_indexed(context: dict[str, Any]) -> None:
    assert context["indexed"]["documents"] == DOCUMENTS[:3]


@then(parsers.parse("the output should report {count:d} indexed documents"))
def output_reports_count(context: dict[str, Any], count: int) -> None:
    assert str(count) in context["stdout"]


@then(parsers.parse('the output should contain "{text}"'))
def output_contains(context: dict[str, Any], text: str) -> None:
    assert text in context["stdout"]


@then("the LLM should not have been called")
def llm_not_called(context: dict[str, Any]) -> None:
    # rag() is the CLI's only path to the LLM, so an unused pipeline stub is
    # what "no API call was made" looks like from here.
    assert "rag_query" not in context["calls"]


@then(parsers.parse('the pipeline should have received the question "{question}"'))
def pipeline_received_question(context: dict[str, Any], question: str) -> None:
    assert context["calls"]["rag_query"] == question


@then("the pipeline should not have been called")
def pipeline_not_called(context: dict[str, Any]) -> None:
    assert "rag_query" not in context["calls"]


@then(parsers.parse('the error output should mention "{text}"'))
def error_mentions(context: dict[str, Any], text: str) -> None:
    assert text in context["stderr"]


@then("the error output should mention the ElasticSearch URL")
def error_mentions_elasticsearch_url(context: dict[str, Any]) -> None:
    assert "http://localhost:9200" in context["stderr"]
