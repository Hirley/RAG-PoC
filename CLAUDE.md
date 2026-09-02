# CLAUDE.md: Diretrizes de Desenvolvimento e Contexto do Projeto

## 1. Visão Geral do Projeto
Este projeto é um sistema de Retrieval-Augmented Generation (RAG) desenvolvido puramente em Python (sem frameworks como LangChain ou LlamaIndex). Ele utiliza o `uv` para gestão de pacotes, ElasticSearch como banco de dados vetorial/texto no ambiente atual e integrações com APIs de LLM.

## 2. Fluxo de Trabalho de Testes (BDD / TDD)
O desenvolvimento deve seguir estritamente o ciclo de Behavior-Driven Development (BDD) e Test-Driven Development (TDD). Nenhuma feature deve ser implementada sem que seu comportamento tenha sido especificado e seus testes falhem primeiro (Red -> Green -> Refactor).

### Ferramentas Adotadas
*   **Gherkin:** Para especificação de comportamentos em linguagem natural (`.feature`).
*   **Pytest / Pytest-BDD:** Para automação dos testes, step definitions e testes unitários clássicos.

### Ciclo de Implementação BDD/TDD
1.  **Especificação (BDD):** Crie ou atualize os arquivos `.feature` na pasta `tests/features/` descrevendo os cenários de uso.
2.  **Step Definitions (BDD):** Implemente os passos correspondentes em `tests/step_defs/` usando `pytest-bdd`.
3.  **Testes Unitários (TDD):** Para lógicas internas complexas (ex: cálculo de pesos, formatação de prompt), escreva testes unitários focados com `pytest` na pasta `tests/unit/`.
4.  **Execução (Fase Red):** Rode os testes e confirme que falham devido à ausência da implementação.
5.  **Implementação (Fase Green):** Escreva o código mínimo necessário em Python para fazer os testes passarem.
6.  **Refatoração:** Melhore a qualidade do código garantindo que a suíte de testes continue passando.

### Comandos de Teste
*   Rodar toda a suíte de testes (BDD e Unitários): `uv run pytest`
*   Rodar apenas cenários BDD: `uv run pytest tests/step_defs/`
*   Rodar testes com saída verbosa: `uv run pytest -v`

## 3. Infraestrutura e Docker (Estado Atual)
A infraestrutura local atual depende do ElasticSearch e da conteinerização da aplicação. Utilizamos o Docker Compose para orquestrar os serviços.

### Comandos Docker
*   **Subir ambiente de desenvolvimento (ElasticSearch em background):**
    `docker-compose up -d elasticsearch`
*   **Subir todo o ambiente (App + Banco):**
    `docker-compose up --build`
*   **Derrubar ambiente e limpar volumes:**
    `docker-compose down -v`
*   **Acessar os logs do ElasticSearch:**
    `docker logs -f elasticsearch_rag`

*Nota:* Certifique-se de que o container do ElasticSearch está saudável (porta 9200) antes de rodar os testes de integração do módulo de busca.

## 4. Padrões de Código e Estilo
*   **Gerenciamento de Pacotes:** Utilize exclusivamente o `uv` (`uv pip install <pacote>`, `uv pip sync`).
*   **Tipagem:** Use Type Hints rigorosamente em todas as funções (`def search(query: str) -> list[dict]:`).
*   **Formatação:** Siga a PEP 8. Recomenda-se o uso do `Ruff` ou `Black` para formatação automática.
*   **Isolamento de Ambiente:** Nunca hardcode variáveis de ambiente. Utilize o pacote `python-dotenv` e exija a presença de um arquivo `.env` para chaves de API (ex: `ANTHROPIC_API_KEY`).

## 5. Estrutura de Diretórios Esperada
```text
/
├── tests/
│   ├── features/          # Arquivos .feature (Gherkin)
│   ├── step_defs/         # Step definitions do pytest-bdd
│   └── unit/              # Testes unitários padrão
├── src/                   # Código fonte da aplicação
│   ├── search.py          # Integração com banco vetorial
│   ├── prompt.py          # Lógica de formatação de contexto
│   └── llm.py             # Chamadas de API (Claude/OpenAI/Groq)
├── docker-compose.yml     # Orquestração do banco e app
├── .env.example           # Template de variáveis
└── pyproject.toml         # Configurações do uv/pytest