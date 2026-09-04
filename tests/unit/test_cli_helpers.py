import json
from pathlib import Path

import pytest

from src.cli import DocumentFileError, format_hits, load_documents, main

VALID = [
    {"title": "Deployment schedule", "content": "The server shuts down at 10 PM."},
]


def write(tmp_path: Path, payload: str) -> Path:
    path = tmp_path / "documents.json"
    path.write_text(payload, encoding="utf-8")
    return path


def test_load_documents_reads_a_list_of_documents(tmp_path: Path) -> None:
    path = write(tmp_path, json.dumps(VALID))

    assert load_documents(path) == VALID


def test_load_documents_rejects_a_top_level_object(tmp_path: Path) -> None:
    path = write(tmp_path, json.dumps({"title": "x", "content": "y"}))

    with pytest.raises(DocumentFileError, match="list of documents"):
        load_documents(path)


def test_load_documents_rejects_malformed_json(tmp_path: Path) -> None:
    path = write(tmp_path, "{not json")

    with pytest.raises(DocumentFileError, match="not valid JSON"):
        load_documents(path)


def test_load_documents_names_the_document_missing_a_field(tmp_path: Path) -> None:
    """Reporting the position matters: a 500-document file gives no other clue
    about which entry is broken."""
    path = write(tmp_path, json.dumps([VALID[0], {"title": "no content here"}]))

    with pytest.raises(DocumentFileError, match=r"document 2\b.*'content'"):
        load_documents(path)


def test_load_documents_rejects_an_empty_file(tmp_path: Path) -> None:
    path = write(tmp_path, json.dumps([]))

    with pytest.raises(DocumentFileError, match="no documents"):
        load_documents(path)


def test_format_hits_numbers_each_result(capsys: pytest.CaptureFixture[str]) -> None:
    output = format_hits(
        [
            {"title": "First", "content": "Content one."},
            {"title": "Second", "content": "Content two."},
        ]
    )

    assert "1. First" in output
    assert "2. Second" in output
    assert "Content one." in output


def test_format_hits_reports_an_empty_result_set() -> None:
    assert "No documents" in format_hits([])


def test_main_without_a_subcommand_fails_and_prints_usage(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """argparse would otherwise exit 0 with an empty run, which looks like the
    command succeeded."""
    exit_code = main([])

    assert exit_code == 2
    assert "usage:" in capsys.readouterr().err
