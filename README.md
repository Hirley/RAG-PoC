# RAG-PoC

A Retrieval-Augmented Generation (RAG) system built in pure Python — no
LangChain, no LlamaIndex. Dependencies are managed with [`uv`](https://github.com/astral-sh/uv),
document retrieval currently runs on ElasticSearch, and generation goes
through an LLM API (Anthropic).

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
3. **LLM** (`src/llm.py`) — sends the prompt to the model and returns the
   generated answer.

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
# then fill in ANTHROPIC_API_KEY

# Start ElasticSearch locally
docker-compose up -d elasticsearch
```

> If `uv` or Docker fail TLS/certificate validation behind a corporate
> proxy or antivirus that intercepts HTTPS, see `uv.toml`
> (`system-certs = true`) — it's already configured to use the OS
> certificate store instead of `uv`'s bundled one.

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

## Project structure

```
.
├── tests/
│   ├── features/          # Gherkin (.feature) specs
│   ├── step_defs/         # pytest-bdd step definitions
│   └── unit/              # Unit tests
├── src/
│   ├── search.py          # ElasticSearch integration (future: PostgreSQL)
│   ├── prompt.py          # Context/question prompt formatting
│   └── llm.py             # LLM API calls (Claude/OpenAI/Groq)
├── docker-compose.yml      # Local ElasticSearch
├── .env.example            # Environment variable template
└── pyproject.toml          # uv/pytest configuration
```

## Roadmap board

Progress is tracked as GitHub Issues against a BDD backlog, organized on the
[RAG PoC Roadmap](https://github.com/users/Hirley/projects/6) project board.

## License

Distributed under the [MIT License](LICENSE).
