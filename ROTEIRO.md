# ROTEIRO.md: Construção de Sistema RAG do Zero (Sem Frameworks)

## 1. Visão Geral e Arquitetura
O objetivo é construir um sistema de Retrieval-Augmented Generation (RAG) puramente em Python, sem o uso de frameworks abstratos (como LangChain ou LlamaIndex). 
A arquitetura é dividida em três pilares fundamentais:
1. **Busca (Search):** Recuperação de contexto relevante em um banco de dados vetorial/texto.
2. **Prompt:** Formatação da pergunta do usuário combinada com o contexto recuperado.
3. **LLM (Geração):** Envio do prompt estruturado para um modelo de linguagem gerar a resposta final.

## 2. Pré-requisitos e Infraestrutura
*   **Linguagem:** Python 3.x
*   **Gerenciador de Pacotes:** `uv` (para alta performance).
*   **Banco de Dados (Atual):** ElasticSearch (rodando localmente via Docker).
*   **IDE recomendada:** VSCode (com as extensões Python e Jupyter).
*   **Provedor de LLM:** Chave de API da Anthropic, OpenAI ou Groq.

## 3. Setup do Ambiente
Crie a pasta do projeto, abra-a no VSCode e inicialize o ambiente utilizando o `uv`:
```bash
uv pip install python-dotenv minsearch elasticsearch anthropic pytest pytest-bdd