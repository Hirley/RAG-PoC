import os

from elasticsearch import Elasticsearch

DEFAULT_INDEX = "rag_docs"
DEFAULT_SIZE = 5


def ensure_index(client: Elasticsearch, index_name: str) -> None:
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name)


def get_client() -> Elasticsearch:
    return Elasticsearch(os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200"))


def search(
    query: str,
    client: Elasticsearch | None = None,
    index_name: str = DEFAULT_INDEX,
    size: int = DEFAULT_SIZE,
) -> list[dict]:
    client = client or get_client()
    response = client.search(
        index=index_name,
        query={"multi_match": {"query": query, "fields": ["title", "content"]}},
        size=size,
    )
    return [hit["_source"] for hit in response["hits"]["hits"]]
