"""Integration tests against a real ElasticSearch instance.

Every other test in this suite mocks the ElasticSearch client, so none of them
would notice a client/server version incompatibility. These do: they exercise
the real wire protocol and are the only place where the pinned client major is
actually verified against the running server.

Skipped automatically when no server is reachable, so the default `uv run
pytest` stays runnable without Docker.
"""

import os
import uuid
from collections.abc import Iterator

import pytest
from elasticsearch import Elasticsearch

from src.search import ensure_index, index_documents, search

pytestmark = pytest.mark.integration

ELASTICSEARCH_URL = os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")

DOCUMENTS = [
    {
        "title": "RAG Limitations",
        "content": (
            "RAG is a poor fit when the answer requires aggregating across the "
            "whole corpus, or when the knowledge base is not the source of truth."
        ),
    },
    {
        "title": "Deployment schedule",
        "content": "The main server shuts down at 10 PM for maintenance.",
    },
    {
        "title": "Onboarding",
        "content": "New engineers get access to the internal wiki on day one.",
    },
]


@pytest.fixture(scope="module")
def client() -> Iterator[Elasticsearch]:
    es = Elasticsearch(ELASTICSEARCH_URL, request_timeout=5)
    try:
        reachable = es.ping()
    except Exception:
        reachable = False

    if not reachable:
        pytest.skip(f"No ElasticSearch reachable at {ELASTICSEARCH_URL}")

    yield es
    es.close()


@pytest.fixture
def index_name(client: Elasticsearch) -> Iterator[str]:
    name = f"rag_docs_test_{uuid.uuid4().hex[:8]}"
    yield name
    client.indices.delete(index=name, ignore_unavailable=True)


def test_client_and_server_majors_are_compatible(client: Elasticsearch) -> None:
    """The mocked tests cannot catch this: a client talking to a server of a
    different major version fails at the transport layer, not in our code."""
    import elasticsearch

    server_major = int(client.info()["version"]["number"].split(".")[0])
    client_major = elasticsearch.__version__[0]

    assert client_major == server_major, (
        f"Python client major {client_major} does not match server major "
        f"{server_major}. Align pyproject.toml with the docker-compose image."
    )


def test_ensure_index_creates_then_is_idempotent(
    client: Elasticsearch, index_name: str
) -> None:
    assert not client.indices.exists(index=index_name)

    ensure_index(client, index_name)
    assert client.indices.exists(index=index_name)

    # Second run must not raise, recreate, or wipe the index.
    client.index(index=index_name, document=DOCUMENTS[0], refresh=True)
    ensure_index(client, index_name)
    assert client.count(index=index_name)["count"] == 1


def test_index_documents_persists_every_document(
    client: Elasticsearch, index_name: str
) -> None:
    ensure_index(client, index_name)

    indexed = index_documents(DOCUMENTS, client=client, index_name=index_name)

    assert indexed == len(DOCUMENTS)


def test_search_ranks_the_matching_document_first(
    client: Elasticsearch, index_name: str
) -> None:
    ensure_index(client, index_name)
    index_documents(DOCUMENTS, client=client, index_name=index_name)

    results = search(
        "When should I not use RAG?", client=client, index_name=index_name
    )

    assert results, "Expected at least one hit from a real ElasticSearch query"
    assert len(results) <= 5
    assert results[0]["title"] == "RAG Limitations"
