"""Unit tests for the agentic loop (`run_agent`).

The LLM and the tools are replaced with fakes built from `langchain_core`
message objects, so the loop runs fully offline.
"""
from langchain_core.messages import AIMessage, ToolMessage

import app as agent_app
from app import run_agent


def _tool_call(name="detect_objects", call_id="call_1"):
    return {"name": name, "args": {}, "id": call_id, "type": "tool_call"}


class FakeLLM:
    """Returns queued AIMessages in order, recording the messages it received."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return self._responses.pop(0)


class RepeatLLM:
    """Always returns the same AIMessage (used to force max_iterations)."""

    def __init__(self, message):
        self._message = message

    def invoke(self, messages):
        return self._message


class FakeTool:
    def __init__(self, content):
        self.content = content

    def invoke(self, tool_call):
        return ToolMessage(content=self.content, tool_call_id=tool_call["id"])


def _install_tool(monkeypatch, content='{"detection_count": 2, "labels": ["person", "person"]}'):
    monkeypatch.setattr(agent_app, "TOOLS", {"detect_objects": FakeTool(content)})


def test_returns_final_answer_without_tools(monkeypatch):
    monkeypatch.setattr(agent_app, "MAX_INPUT_TOKENS", None)
    monkeypatch.setattr(
        agent_app,
        "llm_with_tools",
        FakeLLM([AIMessage(content="Hello!", usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13})]),
    )

    result = run_agent([])

    assert result.response == "Hello!"
    assert result.iterations == 1
    assert result.tools_called == []
    assert result.tokens_used.input == 10
    assert result.tokens_used.total == 13


def test_executes_tool_then_returns_answer(monkeypatch):
    monkeypatch.setattr(agent_app, "MAX_INPUT_TOKENS", None)
    _install_tool(monkeypatch)
    monkeypatch.setattr(
        agent_app,
        "llm_with_tools",
        FakeLLM(
            [
                AIMessage(content="", tool_calls=[_tool_call()]),
                AIMessage(content="There are 2 people."),
            ]
        ),
    )

    result = run_agent([])

    assert result.response == "There are 2 people."
    assert result.iterations == 2
    assert result.tools_called == ["detect_objects"]


def test_max_iterations_guard(monkeypatch):
    monkeypatch.setattr(agent_app, "MAX_INPUT_TOKENS", None)
    _install_tool(monkeypatch)
    # Model that never stops calling tools.
    monkeypatch.setattr(
        agent_app,
        "llm_with_tools",
        RepeatLLM(AIMessage(content="", tool_calls=[_tool_call()])),
    )

    result = run_agent([], max_iterations=2)

    assert result.iterations == 2
    assert "couldn't complete" in result.response.lower()
    assert result.tools_called == ["detect_objects", "detect_objects"]


def test_unknown_tool_is_handled(monkeypatch):
    monkeypatch.setattr(agent_app, "MAX_INPUT_TOKENS", None)
    monkeypatch.setattr(agent_app, "TOOLS", {})  # no tools registered
    monkeypatch.setattr(
        agent_app,
        "llm_with_tools",
        FakeLLM(
            [
                AIMessage(content="", tool_calls=[_tool_call(name="ghost")]),
                AIMessage(content="Recovered."),
            ]
        ),
    )

    result = run_agent([])

    assert result.response == "Recovered."
    assert result.tools_called == ["ghost"]


def test_context_limit_flag(monkeypatch):
    monkeypatch.setattr(agent_app, "MAX_INPUT_TOKENS", 100)
    monkeypatch.setattr(
        agent_app,
        "llm_with_tools",
        FakeLLM(
            [
                AIMessage(
                    content="done",
                    usage_metadata={"input_tokens": 95, "output_tokens": 1, "total_tokens": 96},
                )
            ]
        ),
    )

    result = run_agent([])

    assert result.context_limit_exceeded is True
