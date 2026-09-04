import os

from elasticsearch import Elasticsearch

DEFAULT_INDEX = "rag_docs"
DEFAULT_SIZE = 5


def resolve_index(index_name: str | None = None) -> str:
    """Resolve the index at call time rather than import time: the CLI loads
    .env inside main(), so an import-time lookup would miss
    ELASTICSEARCH_INDEX and silently write to a different index than it reads."""
    return index_name or os.environ.get("ELASTICSEARCH_INDEX", DEFAULT_INDEX)


def ensure_index(client: Elasticsearch, index_name: str) -> None:
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name)


def index_documents(
    documents: list[dict],
    client: Elasticsearch | None = None,
    index_name: str | None = None,
) -> int:
    client = client or get_client()
    index_name = resolve_index(index_name)
    operations: list[dict] = []
    for document in documents:
        operations.append({"index": {"_index": index_name}})
        operations.append(document)

    client.bulk(operations=operations, refresh=True)
    return client.count(index=index_name)["count"]


def get_client() -> Elasticsearch:
    return Elasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))


def search(
    query: str,
    client: Elasticsearch | None = None,
    index_name: str | None = None,
    size: int = DEFAULT_SIZE,
) -> list[dict]:
    client = client or get_client()
    index_name = resolve_index(index_name)
    response = client.search(
        index=index_name,
        query={"multi_match": {"query": query, "fields": ["title", "content"]}},
        size=size,
    )
    return [hit["_source"] for hit in response["hits"]["hits"]]
