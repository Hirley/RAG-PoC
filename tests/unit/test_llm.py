from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from src.llm import DEFAULT_MODEL, LLMRefusalError, get_client, llm


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


def test_get_client_sends_the_workspace_header_when_configured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Identity-linked API keys are rejected with a 400 unless the request
    names the workspace it acts in."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_123")

    client = get_client()

    assert client.default_headers["anthropic-workspace-id"] == "wrkspc_123"


def test_get_client_omits_the_workspace_header_when_unset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A standard API key is rejected when the header is present but empty, so
    it must be absent rather than blank."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID", raising=False)

    client = get_client()

    assert "anthropic-workspace-id" not in client.default_headers
