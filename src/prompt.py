NO_CONTEXT_ANSWER = (
    "I don't have enough information in the knowledge base to answer this question."
)

PROMPT_TEMPLATE = """
You are a corporate knowledge assistant. Answer the QUESTION using only the
facts provided in the CONTEXT below.

If the CONTEXT does not contain enough information to answer, reply exactly:
"{no_context_answer}"

CONTEXT:
{context}

QUESTION:
{question}
""".strip()


def build_prompt(query: str, search_results: list[dict]) -> str:
    context = "\n\n".join(
        f"{document.get('title', '')}\n{document.get('content', '')}".strip()
        for document in search_results
    )
    return PROMPT_TEMPLATE.format(
        no_context_answer=NO_CONTEXT_ANSWER,
        context=context,
        question=query,
    )
