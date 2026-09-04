"""Command-line entrypoint for exercising the RAG pipeline end to end.

Three subcommands, mirroring the three things you need to do to evaluate a RAG
system by hand:

    rag ingest documents.json    load a corpus into the index
    rag search "<question>"      inspect what retrieval returns, no LLM call
    rag ask "<question>"         run the full search -> prompt -> llm pipeline

`search` exists so retrieval can be debugged on its own: when an answer is
wrong it is usually retrieval that is wrong, and checking that should not cost
an API call.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import anthropic
import elasticsearch
from dotenv import load_dotenv

from src.llm import LLMConfigurationError, LLMRefusalError, LLMTruncatedError
from src.rag import rag
from src.search import (
    DEFAULT_SIZE,
    ensure_index,
    get_client,
    index_documents,
    resolve_index,
    search,
)

PROGRAM = "rag"

EXIT_OK = 0
EXIT_ERROR = 1
EXIT_USAGE = 2


class DocumentFileError(RuntimeError):
    """Raised when the corpus file cannot be read as a list of documents."""


def load_documents(path: Path) -> list[dict]:
    """Read and validate a JSON corpus: a list of objects carrying `title` and
    `content`, the two fields src.search queries against."""
    try:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise DocumentFileError(f"File not found: {path}") from error
    except json.JSONDecodeError as error:
        raise DocumentFileError(f"{path} is not valid JSON: {error}") from error

    if not isinstance(payload, list):
        raise DocumentFileError(
            f"{path} must contain a list of documents, got {type(payload).__name__}."
        )

    if not payload:
        raise DocumentFileError(f"{path} contains no documents.")

    for position, document in enumerate(payload, start=1):
        if not isinstance(document, dict):
            raise DocumentFileError(
                f"{path}: document {position} is a {type(document).__name__}, "
                "expected an object."
            )
        # Reporting the position keeps a large corpus debuggable; without it a
        # single malformed entry is invisible in a file of hundreds.
        for field in ("title", "content"):
            if field not in document:
                raise DocumentFileError(
                    f"{path}: document {position} is missing '{field}'."
                )

    return payload


def format_hits(hits: list[dict]) -> str:
    if not hits:
        return "No documents matched the query."

    return "\n\n".join(
        f"{position}. {hit.get('title', '(untitled)')}\n   {hit.get('content', '')}"
        for position, hit in enumerate(hits, start=1)
    )


def elasticsearch_url() -> str:
    return os.environ.get("ELASTICSEARCH_URL", "http://localhost:9200")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=PROGRAM,
        description="Drive the RAG pipeline from the terminal.",
    )
    subcommands = parser.add_subparsers(dest="command", metavar="COMMAND")

    ingest = subcommands.add_parser(
        "ingest", help="Index a JSON corpus into ElasticSearch."
    )
    ingest.add_argument("path", type=Path, help="JSON file holding a list of documents.")
    ingest.add_argument("--index", default=None, help="Index name to write to.")

    retrieve = subcommands.add_parser(
        "search", help="Show what retrieval returns, without calling the LLM."
    )
    retrieve.add_argument("query", nargs="+", help="The question to retrieve for.")
    retrieve.add_argument("--index", default=None, help="Index name to read from.")
    retrieve.add_argument(
        "--size", type=int, default=DEFAULT_SIZE, help="How many documents to return."
    )

    ask = subcommands.add_parser(
        "ask", help="Run the full pipeline and print the generated answer."
    )
    ask.add_argument("query", nargs="+", help="The question to answer.")

    return parser


def run_ingest(args: argparse.Namespace) -> int:
    documents = load_documents(args.path)
    index_name = args.index or resolve_index()

    client = get_client()
    ensure_index(client, index_name)
    total = index_documents(documents, client=client, index_name=index_name)

    print(f"Indexed {len(documents)} documents into '{index_name}'.")
    print(f"The index now holds {total} documents.")
    return EXIT_OK


def run_search(args: argparse.Namespace) -> int:
    query = " ".join(args.query)
    index_name = args.index or resolve_index()

    hits = search(query, client=get_client(), index_name=index_name, size=args.size)
    print(format_hits(hits))
    return EXIT_OK


def run_ask(args: argparse.Namespace) -> int:
    # Checked up front so a missing key fails immediately with an actionable
    # message, rather than as an SDK traceback after the retrieval round trip.
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and fill it "
            "in, or export the variable in your shell.",
            file=sys.stderr,
        )
        return EXIT_ERROR

    query = " ".join(args.query)
    print(rag(query))
    return EXIT_OK


HANDLERS = {"ingest": run_ingest, "search": run_search, "ask": run_ask}


def main(argv: list[str] | None = None) -> int:
    load_dotenv()

    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.command:
        # argparse would otherwise return successfully having done nothing,
        # which in a pipeline reads as a command that worked.
        parser.print_usage(sys.stderr)
        print(f"{PROGRAM}: a command is required.", file=sys.stderr)
        return EXIT_USAGE

    try:
        return HANDLERS[args.command](args)
    except DocumentFileError as error:
        print(f"{PROGRAM}: {error}", file=sys.stderr)
    except elasticsearch.ApiError as error:
        print(f"{PROGRAM}: ElasticSearch rejected the request: {error}", file=sys.stderr)
    except elasticsearch.TransportError:
        # TransportError's str() is a bare "Connection error", so the URL that
        # actually failed has to be reported here.
        print(
            f"{PROGRAM}: could not reach ElasticSearch at {elasticsearch_url()}. "
            "Is it running? Try: docker-compose up -d elasticsearch",
            file=sys.stderr,
        )
    except LLMConfigurationError as error:
        print(f"{PROGRAM}: {error} Check your .env.", file=sys.stderr)
    except LLMTruncatedError as error:
        print(f"{PROGRAM}: {error}", file=sys.stderr)
    except LLMRefusalError as error:
        print(f"{PROGRAM}: {error}", file=sys.stderr)
    except anthropic.APIError as error:
        print(f"{PROGRAM}: the Anthropic API call failed: {error}", file=sys.stderr)

    return EXIT_ERROR


if __name__ == "__main__":
    sys.exit(main())
