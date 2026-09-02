from elasticsearch import Elasticsearch


def ensure_index(client: Elasticsearch, index_name: str) -> None:
    if not client.indices.exists(index=index_name):
        client.indices.create(index=index_name)
