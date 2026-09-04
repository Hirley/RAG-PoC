# RAG-PoC

A Retrieval-Augmented Generation (RAG) system built in pure Python — no
LangChain, no LlamaIndex. Dependencies are managed with [`uv`](https://github.com/astral-sh/uv),
document retrieval currently runs on ElasticSearch, and generation goes through
Anthropic, OpenAI or Groq — selectable at runtime.

## Architecture

The pipeline is three independent, testable stages, orchestrated by a single
`rag(query)` function:

```
query --> search(query) --> build_prompt(query, results) --> llm(prompt) --> answer
          src/search.py       src/prompt.py                    src/llm.py
```

1. **Search** (`src/search.py`) — retrieves the top-N relevant documents from
   the index for a given query.
2. **Prompt** (`src/prompt.py`) — formats the retrieved context and the
   question into the LLM's prompt template.
3. **LLM** (`src/llm.py`) — sends the prompt to the selected provider and
   returns the generated answer.

**Roadmap:** the ElasticSearch backend is planned to be replaced by
PostgreSQL + `pgvector` in production, to simplify disaster recovery
(physical/logical replication, dumps) and to get ACID guarantees on
ingestion.

## Requirements

- Python >= 3.12
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- Docker (for local ElasticSearch)

## Setup

```bash
# Install dependencies and create the virtual environment
uv sync

# Configure environment variables
cp .env.example .env
# then fill in the key for the provider you intend to use

# Start ElasticSearch locally
docker-compose up -d elasticsearch
```

> **Identity-linked API keys** must also set `ANTHROPIC_WORKSPACE_ID`. The API
> rejects them with a `400 invalid_request_error` — *"anthropic-workspace-id is
> required"* — unless the request names the workspace it acts in. The id is in
> the Anthropic Console under **Settings → Workspaces**, and appears in the URL
> as `wrkspc_...`. A standard Console API key needs no such header, so leave the
> variable blank for one.

> If `uv` or Docker fail TLS/certificate validation behind a corporate
> proxy or antivirus that intercepts HTTPS, see `uv.toml`
> (`system-certs = true`) — it's already configured to use the OS
> certificate store instead of `uv`'s bundled one.

## Using the CLI

`src/cli.py` is the entrypoint that drives the pipeline from the terminal. It
is registered as the `rag` console script by `uv sync`, and also runs as
`python -m src.cli`.

```bash
# 1. Load a corpus (a JSON file: a list of {"title", "content"} objects)
uv run rag ingest data/sample_documents.json

# 2. Inspect retrieval on its own — no LLM call, so no API cost
uv run rag search "When are deploys frozen?"

# 3. Run the full pipeline and get a generated answer
uv run rag ask "When are deploys frozen?"
```

`search` is deliberately separate from `ask`: when an answer comes back wrong,
retrieval is usually what is wrong, and diagnosing that should not cost an API
call. `--size` controls how many documents come back, and `--index` overrides
the target index for a single run.

The index is read from `ELASTICSEARCH_INDEX` (default `rag_docs`), so `ingest`
and `ask` always agree on where the corpus lives:

```bash
ELASTICSEARCH_INDEX=my_corpus uv run rag ingest my_documents.json
ELASTICSEARCH_INDEX=my_corpus uv run rag ask "What does it say?"
```

Exit codes are `0` on success, `1` on a runtime failure (unreachable
ElasticSearch, a missing provider key, malformed corpus file) and `2` on a
usage error, so the commands compose in a shell script.

### Choosing a provider

Three providers are available: `anthropic` (default), `openai` and `groq`.
`LLM_PROVIDER` sets the deployment's default and `--provider` overrides it for
one question, so the same question can be put to each of them without editing
anything:

```bash
uv run rag ask --provider openai "When are deploys frozen?"
uv run rag ask --provider groq   "When are deploys frozen?"
```

Groq is not a third integration. It speaks the OpenAI wire format, so it reuses
that SDK against a different base URL rather than adding a dependency for the
same request shape.

Only the selected provider's key is required — `ANTHROPIC_API_KEY`,
`OPENAI_API_KEY` or `GROQ_API_KEY`, each the name that provider's own SDK
already expects. A missing one is reported before retrieval runs, naming the
variable the chosen provider actually needs.

The SDKs disagree on how they report a refusal and a truncation — Anthropic
uses `stop_reason`, OpenAI uses `finish_reason` plus a separate `refusal`
field — and that difference stops inside `src/llm.py`. Every provider reaches
the pipeline as one of the same three outcomes: an answer, `LLMRefusalError`,
or `LLMTruncatedError`.

### Choosing the model

Each provider carries its own default and its own override variable, so
switching provider does not silently carry the previous provider's model name
along:

| Provider | Variable | Default |
| --- | --- | --- |
| `anthropic` | `ANTHROPIC_MODEL` | `claude-sonnet-5` |
| `openai` | `OPENAI_MODEL` | `gpt-4o` |
| `groq` | `GROQ_MODEL` | `llama-3.3-70b-versatile` |

```bash
ANTHROPIC_MODEL=claude-opus-5 uv run rag ask "Something genuinely hard"
```

These defaults are a starting point, not a guarantee that your account has
access to that particular model — set the variable if it does not.

`LLM_MAX_TOKENS` (default `4096`) bounds the generated answer on every
provider. It is a **ceiling, not a budget**: only tokens actually generated are
billed, so raising it costs nothing by itself. It exists to stop a runaway
generation.

Thinking is enabled by default on current models and its tokens count against
the same ceiling, so a value sized for the answer alone would truncate
routinely. An answer that does hit the ceiling raises rather than returning
what it got — a half-finished answer is indistinguishable from a complete one
by looking at the text, which in a RAG system is worse than an error.

## Running the tests

This project follows BDD/TDD: every feature starts as a Gherkin spec in
`tests/features/`, bound to step definitions in `tests/step_defs/` via
`pytest-bdd`, with `pytest` covering isolated logic in `tests/unit/`.

```bash
# Full suite
uv run pytest

# Verbose output
uv run pytest -v
```

### Integration tests

Everything under `tests/step_defs/` and `tests/unit/` mocks the ElasticSearch
client, so none of it would catch a client/server incompatibility.
`tests/integration/` talks to a real server and is the only place where the
pinned client is actually verified against the running cluster.

These tests **skip automatically** when no server is reachable, so the plain
`uv run pytest` above works without Docker. To run them for real:

```bash
docker-compose up -d elasticsearch
uv run pytest -m integration -v
```

Client and server must stay on the same major version — the pin in
`pyproject.toml` (`elasticsearch>=8.15.0,<9.0.0`) and the image tag in
`docker-compose.yml` are two halves of one decision, and
`test_client_and_server_majors_are_compatible` fails loudly if they drift.

## Running in Docker

```bash
# Build the app image and run the full suite against a live ElasticSearch
docker-compose up --build

# Just the database, for local development
docker-compose up -d elasticsearch
```

The `app` service waits for the `elasticsearch` healthcheck before starting,
and its default command runs the test suite.

## Project structure

```
.
├── tests/
│   ├── features/          # Gherkin (.feature) specs
│   ├── step_defs/         # pytest-bdd step definitions
│   ├── unit/              # Unit tests
│   └── integration/       # Tests against a real ElasticSearch
├── src/
│   ├── search.py          # ElasticSearch integration (future: PostgreSQL)
│   ├── prompt.py          # Context/question prompt formatting
│   ├── llm.py             # LLM API calls (Anthropic / OpenAI / Groq)
│   ├── rag.py             # Orchestrates search -> prompt -> llm
│   └── cli.py             # Terminal entrypoint (ingest / search / ask)
├── data/
│   └── sample_documents.json  # Small corpus for trying the CLI out
├── Dockerfile              # Application image
├── docker-compose.yml      # ElasticSearch + app
├── .env.example            # Environment variable template
└── pyproject.toml          # uv/pytest configuration
```

## Roadmap board

Progress is tracked as GitHub Issues against a BDD backlog, organized on the
[RAG PoC Roadmap](https://github.com/users/Hirley/projects/6) project board.

## License

Distributed under the [MIT License](LICENSE).
