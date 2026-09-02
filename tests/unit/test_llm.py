from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.llm import DEFAULT_MODEL, LLMRefusalError, llm


def build_client(response: SimpleNamespace) -> MagicMock:
    client = MagicMock()
    client.beta.messages.create.return_value = response
    return client


def test_llm_ignores_non_text_blocks() -> None:
    """Thinking is on by default on current models, so the first content block
    is not necessarily the answer."""
    client = build_client(
        SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="thinking", thinking="Let me reason first."),
                SimpleNamespace(type="text", text="The answer is 42."),
            ],
        )
    )

    assert llm("Any prompt", client=client) == "The answer is 42."


def test_llm_joins_multiple_text_blocks() -> None:
    client = build_client(
        SimpleNamespace(
            stop_reason="end_turn",
            content=[
                SimpleNamespace(type="text", text="First part."),
                SimpleNamespace(type="text", text="Second part."),
            ],
        )
    )

    assert llm("Any prompt", client=client) == "First part.\nSecond part."


def test_llm_raises_on_refusal() -> None:
    client = build_client(
        SimpleNamespace(
            stop_reason="refusal",
            stop_details=SimpleNamespace(category="cyber", explanation="Declined."),
            content=[],
        )
    )

    with pytest.raises(LLMRefusalError):
        llm("Any prompt", client=client)


def test_llm_sends_the_prompt_as_a_single_user_message() -> None:
    client = build_client(
        SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text="ok")],
        )
    )

    llm("What is RAG?", client=client)

    kwargs = client.beta.messages.create.call_args.kwargs
    assert kwargs["model"] == DEFAULT_MODEL
    assert kwargs["messages"] == [{"role": "user", "content": "What is RAG?"}]
