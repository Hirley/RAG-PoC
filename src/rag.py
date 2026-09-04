from src.llm import llm
from src.prompt import build_prompt
from src.search import search


def rag(query: str, provider: str | None = None) -> str:
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt, provider=provider)
    return answer
