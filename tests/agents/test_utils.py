import asyncio
import json
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import AsyncMock, Mock

import openai
import pytest

from corral.agents.utils import (
    RETRY_EXCEPTIONS,
    LiteLLMMessage,
    LLMResponse,
    before_sleep_loguru,
    format_examples,
    llm_call,
    save_agent_messages,
    serialize_messages,
)

from .conftest import MockFunction, MockToolCall


# Shared mock classes for LiteLLM testing
class MockLiteLLMMessage:
    """Mock message class for LiteLLM responses."""

    def __init__(self):
        self.role = "assistant"
        self.content = "This is a response"
        self.tool_call_id = None
        self.name = None
        self.reasoning_content = None


class MockLiteLLMChoice:
    """Mock choice class for LiteLLM responses."""

    def __init__(self):
        self.message = MockLiteLLMMessage()
        self.logprobs = None


class MockLiteLLMResponse:
    """Mock response class for LiteLLM."""

    def __init__(self, include_usage=False):
        self.choices = [MockLiteLLMChoice()]
        self.id = "mock_response_id"
        # Always have usage attribute, but set to None if not included
        self.usage = MockLiteLLMUsage() if include_usage else None


class MockLiteLLMUsage:
    """Mock usage class for LiteLLM responses."""

    def __init__(self):
        self.prompt_tokens = 10
        self.completion_tokens = 20
        self.total_tokens = 30


class MockAsyncStream:
    """Reusable async iterator returned by a streaming LiteLLM request."""

    def __init__(self, chunks):
        self.chunks = chunks

    def __aiter__(self):
        return self._iterate()

    async def _iterate(self):
        for chunk in self.chunks:
            yield chunk


class MockLiteLLM:
    """Mock LiteLLM class."""

    def __init__(self):
        self.acompletion = AsyncMock()
        self.stream_chunk_builder = Mock()


def setup_mock_litellm(monkeypatch, return_usage=False):
    """Helper to set up LiteLLM mocking."""
    mock_litellm = MockLiteLLM()
    mock_response = MockLiteLLMResponse(include_usage=return_usage)
    mock_litellm.acompletion.return_value = MockAsyncStream(["chunk"])
    mock_litellm.stream_chunk_builder.return_value = mock_response
    monkeypatch.setattr("corral.agents.utils.litellm", mock_litellm)
    return mock_litellm, mock_response


@pytest.fixture()
def anyio_backend():
    return "asyncio"


def setup_mock_logger(monkeypatch):
    """Helper to set up logger mocking."""
    mock_logger = MockFunction()
    mock_logger.warning = MockFunction()
    monkeypatch.setattr("corral.agents.utils.logger", mock_logger)
    return mock_logger


class MockRetryState:
    """Mock retry state for testing retry functionality."""

    def __init__(self, attempt_number=2, sleep_time=30.0, exception=None):
        self.attempt_number = attempt_number
        self.next_action = type("NextAction", (), {"sleep": sleep_time})()
        self.outcome = type("Outcome", (), {"exception": lambda self: exception})()


class MockSerializableMessage:
    """Mock message class for serialization testing."""

    def __init__(
        self,
        role="assistant",
        content="I'll help you",
        tool_calls=None,
        tool_call_id=None,
        name=None,
        msg_id=None,
    ):
        self.role = role
        self.content = content
        self.tool_calls = tool_calls or []
        self.tool_call_id = tool_call_id
        self.name = name
        self.id = msg_id


def test_retry_exceptions_are_openai_exceptions():
    """Test that RETRY_EXCEPTIONS contains valid OpenAI exception types."""
    # Test that all exceptions in RETRY_EXCEPTIONS are OpenAI exceptions
    for exception_type in RETRY_EXCEPTIONS:
        assert hasattr(openai, exception_type.__name__)
        assert issubclass(exception_type, Exception)

    # Test that the tuple is not empty
    assert len(RETRY_EXCEPTIONS) > 0


def test_before_sleep_loguru_logs_retry_info(monkeypatch):
    """Test that before_sleep_loguru logs retry information."""
    mock_logger = MockFunction()
    mock_logger.warning = MockFunction()
    monkeypatch.setattr("corral.agents.utils.logger", mock_logger)

    mock_retry_state = MockRetryState(exception=ValueError("boom"))

    before_sleep_loguru(mock_retry_state)

    mock_logger.warning.assert_called_once_with(
        "LLM call retry 2/3 after ValueError: boom; waiting 30s"
    )


def test_litellm_message_structure():
    """Test that LiteLLMMessage has expected structure."""
    # Test with minimal required fields
    msg1: LiteLLMMessage = {"role": "user", "content": "Hello"}
    assert msg1["role"] == "user"
    assert msg1["content"] == "Hello"

    # Test with optional fields
    msg2: LiteLLMMessage = {
        "role": "assistant",
        "content": "Hi there",
        "tool_call_id": "call_123",
        "name": "assistant",
        "id": "msg_123",
    }
    assert msg2["tool_call_id"] == "call_123"
    assert msg2["name"] == "assistant"
    assert msg2["id"] == "msg_123"


@pytest.mark.anyio()
async def test_llm_call_basic(monkeypatch):
    """Test basic llm_call without tools."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    result = await llm_call(model="gpt-3.5-turbo", messages=messages, temperature=0.7)

    # Check that result is LLMResponse wrapper
    assert isinstance(result, LLMResponse)
    assert result.message == mock_response.choices[0].message
    assert result.content == "This is a response"
    assert result.id == "mock_response_id"  # ID should always be included
    mock_litellm.acompletion.assert_awaited_once_with(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        api_base=None,
        stream=True,
    )
    mock_litellm.stream_chunk_builder.assert_called_once_with(
        ["chunk"], messages=messages
    )


@pytest.mark.anyio()
async def test_llm_call_with_tools(monkeypatch):
    """Test llm_call with tools."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    result = await llm_call(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7, tools=tools
    )

    # Check that result is LLMResponse wrapper
    assert isinstance(result, LLMResponse)
    assert result.message == mock_response.choices[0].message
    assert result.id == "mock_response_id"
    mock_litellm.acompletion.assert_awaited_once_with(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        tools=tools,
        tool_choice="auto",
        api_base=None,
        stream=True,
    )


@pytest.mark.anyio()
async def test_llm_call_anthropic_model(monkeypatch):
    """Test llm_call with anthropic model adds max_tokens."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    result = await llm_call(
        model="anthropic/claude-3-sonnet", messages=messages, temperature=0.7
    )

    # Check that result is LLMResponse wrapper
    assert isinstance(result, LLMResponse)
    assert result.message == mock_response.choices[0].message
    assert result.id == "mock_response_id"
    mock_litellm.acompletion.assert_awaited_once_with(
        model="anthropic/claude-3-sonnet",
        messages=messages,
        temperature=0.7,
        max_tokens=8192,
        api_base=None,
        stream=True,
    )


@pytest.mark.anyio()
async def test_llm_call_reasoning_effort_forces_temperature_one(monkeypatch):
    """Reasoning effort must override temperature to 1 (Anthropic thinking rule)."""
    mock_litellm, _ = setup_mock_litellm(monkeypatch)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    await llm_call(
        model="anthropic/claude-3-sonnet",
        messages=messages,
        temperature=0.0,
        reasoning_effort="medium",
    )

    # Anthropic rejects any temperature other than 1 when thinking is enabled,
    # so the requested 0.0 is overridden to 1 while reasoning_effort passes through.
    mock_litellm.acompletion.assert_awaited_once_with(
        model="anthropic/claude-3-sonnet",
        messages=messages,
        temperature=1,
        max_tokens=8192,
        api_base=None,
        reasoning_effort="medium",
        stream=True,
    )


@pytest.mark.anyio()
async def test_llm_call_with_usage_info(monkeypatch):
    """Test llm_call with return_usage=True."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch, return_usage=True)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    result = await llm_call(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7, return_usage=True
    )

    # Check that result is LLMResponse wrapper with usage metadata
    assert isinstance(result, LLMResponse)
    assert result.message == mock_response.choices[0].message
    assert result.id == "mock_response_id"
    assert result.usage == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


@pytest.mark.anyio()
async def test_llm_call_with_logprobs(monkeypatch):
    """Test llm_call with logprobs=True in kwargs."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch)
    # Set logprobs on the mock response
    mock_response.choices[0].logprobs = {"token": "test", "logprob": -0.5}

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    result = await llm_call(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        logprobs=True,
        top_logprobs=20,
    )

    # Check that result includes logprobs when requested
    assert isinstance(result, LLMResponse)
    assert result.logprobs == {"token": "test", "logprob": -0.5}
    assert result.id == "mock_response_id"


@pytest.mark.anyio()
async def test_llm_call_without_logprobs(monkeypatch):
    """Test llm_call without logprobs (default behavior)."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    result = await llm_call(model="gpt-3.5-turbo", messages=messages, temperature=0.7)

    # Check that result does NOT include logprobs when not requested
    assert isinstance(result, LLMResponse)
    assert result.logprobs is None
    assert result.id == "mock_response_id"


@pytest.mark.anyio()
async def test_llm_call_exception_handling(monkeypatch):
    """Test llm_call exception handling."""
    mock_litellm = MockLiteLLM()
    mock_litellm.acompletion.side_effect = Exception("API Error")
    monkeypatch.setattr("corral.agents.utils.litellm", mock_litellm)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    with pytest.raises(Exception) as exc_info:
        await llm_call(model="gpt-3.5-turbo", messages=messages, temperature=0.7)

    assert str(exc_info.value) == "API Error"


@pytest.mark.anyio()
async def test_llm_calls_can_overlap(monkeypatch):
    """Independent provider requests must not be serialized by the helper."""
    active = 0
    max_active = 0
    response = MockLiteLLMResponse()

    async def acompletion(**_kwargs):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0)
        active -= 1
        return MockAsyncStream([response])

    mock_litellm = MockLiteLLM()
    mock_litellm.acompletion.side_effect = acompletion
    mock_litellm.stream_chunk_builder.side_effect = lambda chunks, **_kwargs: chunks[0]
    monkeypatch.setattr("corral.agents.utils.litellm", mock_litellm)
    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    await asyncio.gather(
        llm_call(model="model-a", messages=messages, temperature=0.0),
        llm_call(model="model-b", messages=messages, temperature=0.0),
    )

    assert max_active == 2


@pytest.mark.anyio()
async def test_llm_call_rejects_an_empty_stream(monkeypatch):
    """An empty stream must fail explicitly instead of causing an index error."""
    mock_litellm = MockLiteLLM()
    mock_litellm.acompletion.return_value = MockAsyncStream([])
    mock_litellm.stream_chunk_builder.return_value = None
    monkeypatch.setattr("corral.agents.utils.litellm", mock_litellm)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])
    with pytest.raises(ValueError, match="no response chunks"):
        await llm_call(model="model", messages=messages, temperature=0.0)


@pytest.mark.anyio()
async def test_llm_call_cannot_be_downgraded_to_non_streaming(monkeypatch):
    """The shared helper keeps streaming enabled for every caller."""
    mock_litellm, _ = setup_mock_litellm(monkeypatch)
    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    await llm_call(model="model", messages=messages, temperature=0.0, stream=False)

    assert mock_litellm.acompletion.await_args.kwargs["stream"] is True


def test_format_examples_none():
    """Test format_examples with None input."""
    result = format_examples(None)
    assert result == ""


def test_format_examples_empty_list():
    """Test format_examples with empty list."""
    result = format_examples([])
    assert (
        result
        == "To help you in understanding this task, the next 0 examples are provided:\n\n"
    )


def test_format_examples_single_example():
    """Test format_examples with single example."""
    examples = ["This is an example."]
    result = format_examples(examples)
    expected = "To help you in understanding this task, the next 1 examples are provided:\n\nThis is an example."
    assert result == expected


def test_format_examples_multiple_examples():
    """Test format_examples with multiple examples."""
    examples = ["Example 1", "Example 2", "Example 3"]
    result = format_examples(examples)
    expected = "To help you in understanding this task, the next 3 examples are provided:\n\nExample 1\n\nExample 2\n\nExample 3"
    assert result == expected


def test_serialize_messages_dict_messages():
    """Test serializing dictionary messages."""
    messages = cast(
        "list[LiteLLMMessage]",
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
    )

    result = serialize_messages(messages)

    assert result == messages


def test_serialize_messages_with_tool_calls():
    """Test serializing messages with tool calls."""
    mock_tool_call = MockToolCall("call_123", "test_function", {"arg": "value"})
    mock_message = MockSerializableMessage(tool_calls=[mock_tool_call])

    result = serialize_messages([mock_message])

    assert len(result) == 1
    message = result[0]
    assert message["role"] == "assistant"
    assert message["content"] == "I'll help you"
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["id"] == "call_123"
    assert message["tool_calls"][0]["function"]["name"] == "test_function"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"arg": "value"}'


def test_serialize_messages_with_id():
    """Test serializing messages with id field."""
    mock_message = MockSerializableMessage(
        role="assistant", content="Hi there", msg_id="msg_12345"
    )

    result = serialize_messages([mock_message])

    assert len(result) == 1
    message = result[0]
    assert message["role"] == "assistant"
    assert message["content"] == "Hi there"
    assert message["id"] == "msg_12345"


def test_save_agent_messages_basic():
    """Test basic save_agent_messages functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        messages = cast(
            "list[LiteLLMMessage]",
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )

        result_path = save_agent_messages(
            messages=messages,
            task_id="test_task",
            agent_name="test_agent",
            model="test_model",
            output_dir=temp_dir,
        )

        assert Path(result_path).exists()
        assert Path(result_path).parent == Path(temp_dir)

        # Check file content
        with open(result_path) as f:
            data = json.load(f)

        assert data["task_id"] == "test_task"
        assert data["agent"] == "test_agent"
        assert "timestamp" in data
        assert data["messages"] == messages


def test_save_agent_messages_with_tools(tmp_path):
    """Test save_agent_messages with tools."""
    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])
    tools = [{"name": "test_tool", "description": "A test tool"}]

    result_path = save_agent_messages(
        messages=messages,
        task_id="test_task",
        agent_name="test_agent",
        model="test_model",
        output_dir=str(tmp_path),
        tools=tools,
    )

    with open(result_path) as f:
        data = json.load(f)

    assert data["tools"] == tools


def test_save_agent_messages_keeps_trace_metadata_outside_messages(tmp_path):
    """Graph annotations must never become provider message keys."""
    messages = cast(
        "list[LiteLLMMessage]",
        [
            {"role": "user", "content": "Hello"},
            {
                "role": "assistant",
                "content": "Result",
                "name": "evaluate_node_0001",
            },
        ],
    )
    original_messages = [message.copy() for message in messages]
    trace_metadata = {
        "schema": "corral.ai_scientist.graph",
        "nodes": [{"id": "node_0001", "label": "recognizable node"}],
        "edges": [],
    }

    result_path = save_agent_messages(
        messages=messages,
        task_id="test_task",
        agent_name="test_agent",
        model="test_model",
        output_dir=str(tmp_path),
        trace_metadata=trace_metadata,
    )

    with open(result_path) as f:
        data = json.load(f)

    assert data["messages"] == original_messages
    assert data["trace_metadata"] == trace_metadata
    assert messages == original_messages


def test_save_agent_messages_creates_directory():
    """Test that save_agent_messages creates output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "new_logs"

        messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

        result_path = save_agent_messages(
            messages=messages,
            task_id="test_task",
            agent_name="test_agent",
            model="test_model",
            output_dir=str(output_dir),
        )

        assert output_dir.exists()
        assert Path(result_path).exists()


def test_save_agent_messages_filename_format(monkeypatch):
    """Test that save_agent_messages generates correct filename format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

        class MockDatetime:
            @staticmethod
            def now(tz=None):
                class MockNow:
                    def strftime(self, fmt):
                        return "20240101_120000"

                return MockNow()

        mock_datetime = MockDatetime()
        monkeypatch.setattr("corral.agents.utils.datetime", mock_datetime)

        result_path = save_agent_messages(
            messages=messages,
            task_id="test_task",
            agent_name="test_agent",
            model="test_model",
            output_dir=temp_dir,
        )

        expected_filename = "test_task_20240101_120000.json"
        assert Path(result_path).name == expected_filename


def test_message_serialization_and_saving():
    """Test message serialization and saving working together."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create mock messages with various types
        mock_message1 = MockSerializableMessage(
            role="user", content="Hello", tool_calls=None
        )
        mock_message2 = {"role": "assistant", "content": "Hi there"}

        messages = [mock_message1, mock_message2]

        # Save messages
        result_path = save_agent_messages(
            messages=messages,
            task_id="integration_test",
            agent_name="test_agent",
            model="test_model",
            output_dir=temp_dir,
        )

        # Load and verify
        with open(result_path) as f:
            data = json.load(f)

        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello"
        assert data["messages"][1] == mock_message2
