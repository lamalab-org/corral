import json
import tempfile
from pathlib import Path
from typing import cast

import openai
import pytest

from corral.agents.utils import (
    RETRY_EXCEPTIONS,
    TYPE_MAPPING,
    LiteLLMMessage,
    LLMResponse,
    _parse_argument_string_to_dict,
    before_sleep_loguru,
    convert_dict_arg,
    convert_to_openai_tool_format,
    format_examples,
    llm_call,
    parse_string_argument,
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


class MockLiteLLM:
    """Mock LiteLLM class."""

    def __init__(self):
        self.completion = MockFunction()


def setup_mock_litellm(monkeypatch, return_usage=False):
    """Helper to set up LiteLLM mocking."""
    mock_litellm = MockLiteLLM()
    mock_response = MockLiteLLMResponse(include_usage=return_usage)
    mock_litellm.completion.return_value = mock_response
    monkeypatch.setattr("corral.agents.utils.litellm", mock_litellm)
    return mock_litellm, mock_response


def setup_mock_logger(monkeypatch):
    """Helper to set up logger mocking."""
    mock_logger = MockFunction()
    mock_logger.warning = MockFunction()
    monkeypatch.setattr("corral.agents.utils.logger", mock_logger)
    return mock_logger


class MockRetryState:
    """Mock retry state for testing retry functionality."""

    def __init__(self, attempt_number=2, sleep_time=30.0):
        self.attempt_number = attempt_number
        self.next_action = type("NextAction", (), {"sleep": sleep_time})()


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


def test_type_mapping_functionality():
    """Test that TYPE_MAPPING correctly maps Python types to JSON types."""
    # Test that common Python types are correctly mapped
    test_cases = [
        ("str", "string"),
        ("bool", "boolean"),
        ("int", "integer"),
        ("float", "number"),
        ("list[str]", "array"),
        ("none", "null"),
        ("dict", "object"),
    ]

    for python_type, expected_json_type in test_cases:
        assert TYPE_MAPPING.get(python_type) == expected_json_type

    # Test that unknown types return None (graceful fallback)
    assert TYPE_MAPPING.get("unknown_type") is None


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
    mock_logger.info = MockFunction()
    monkeypatch.setattr("corral.agents.utils.logger", mock_logger)

    mock_retry_state = MockRetryState()

    before_sleep_loguru(mock_retry_state)

    mock_logger.info.assert_called_once_with("Retrying: 2, wait: 30.0 seconds")


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


def test_llm_call_basic(monkeypatch):
    """Test basic llm_call without tools."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    result = llm_call(model="gpt-3.5-turbo", messages=messages, temperature=0.7)

    # Check that result is LLMResponse wrapper
    assert isinstance(result, LLMResponse)
    assert result.message == mock_response.choices[0].message
    assert result.content == "This is a response"
    assert result.id == "mock_response_id"  # ID should always be included
    mock_litellm.completion.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        api_base=None,
    )


def test_llm_call_with_tools(monkeypatch):
    """Test llm_call with tools."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    result = llm_call(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7, tools=tools
    )

    # Check that result is LLMResponse wrapper
    assert isinstance(result, LLMResponse)
    assert result.message == mock_response.choices[0].message
    assert result.id == "mock_response_id"
    mock_litellm.completion.assert_called_once_with(
        model="gpt-3.5-turbo",
        messages=messages,
        temperature=0.7,
        tools=tools,
        tool_choice="auto",
        api_base=None,
    )


def test_llm_call_anthropic_model(monkeypatch):
    """Test llm_call with anthropic model adds max_tokens."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    result = llm_call(
        model="anthropic/claude-3-sonnet", messages=messages, temperature=0.7
    )

    # Check that result is LLMResponse wrapper
    assert isinstance(result, LLMResponse)
    assert result.message == mock_response.choices[0].message
    assert result.id == "mock_response_id"
    mock_litellm.completion.assert_called_once_with(
        model="anthropic/claude-3-sonnet",
        messages=messages,
        temperature=0.7,
        max_tokens=8192,
        api_base=None,
    )


def test_llm_call_with_usage_info(monkeypatch):
    """Test llm_call with return_usage=True."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch, return_usage=True)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    result = llm_call(
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


def test_llm_call_with_logprobs(monkeypatch):
    """Test llm_call with logprobs=True in kwargs."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch)
    # Set logprobs on the mock response
    mock_response.choices[0].logprobs = {"token": "test", "logprob": -0.5}

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    result = llm_call(
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


def test_llm_call_without_logprobs(monkeypatch):
    """Test llm_call without logprobs (default behavior)."""
    mock_litellm, mock_response = setup_mock_litellm(monkeypatch)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    result = llm_call(model="gpt-3.5-turbo", messages=messages, temperature=0.7)

    # Check that result does NOT include logprobs when not requested
    assert isinstance(result, LLMResponse)
    assert result.logprobs is None
    assert result.id == "mock_response_id"


def test_llm_call_exception_handling(monkeypatch):
    """Test llm_call exception handling."""
    mock_litellm = MockLiteLLM()
    mock_litellm.completion.side_effect = Exception("API Error")
    monkeypatch.setattr("corral.agents.utils.litellm", mock_litellm)

    messages = cast("list[LiteLLMMessage]", [{"role": "user", "content": "Hello"}])

    with pytest.raises(Exception) as exc_info:
        llm_call(model="gpt-3.5-turbo", messages=messages, temperature=0.7)

    assert str(exc_info.value) == "API Error"


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


def test_convert_dict_arg_basic_string():
    """Test convert_dict_arg with basic string argument."""
    arg = {"name": "test_arg", "type": "str", "description": "A test argument"}
    result = convert_dict_arg(arg)
    expected = {"description": "A test argument", "type": "string"}
    assert result == expected


def test_convert_dict_arg_list_str():
    """Test convert_dict_arg with list[str] type."""
    arg = {"name": "items", "type": "list[str]", "description": "A list of items"}
    result = convert_dict_arg(arg)
    expected = {
        "description": 'A list of items - Provide as an array of strings, e.g., ["item1", "item2"]',
        "type": "array",
        "items": {"type": "string"},
    }
    assert result == expected


def test_convert_dict_arg_with_choices():
    """Test convert_dict_arg with choices."""
    arg = {
        "name": "operation",
        "type": "str",
        "description": "Math operation",
        "choices": ["add", "subtract", "multiply"],
    }
    result = convert_dict_arg(arg)
    expected = {
        "description": "Math operation",
        "type": "string",
        "enum": ["add", "subtract", "multiply"],
    }
    assert result == expected


def test_convert_dict_arg_with_default():
    """Test convert_dict_arg with default value."""
    arg = {
        "name": "count",
        "type": "int",
        "description": "Number of items",
        "default": 5,
    }
    result = convert_dict_arg(arg)
    expected = {"description": "Number of items", "type": "integer", "default": 5}
    assert result == expected


def test_convert_dict_arg_unknown_type(monkeypatch):
    """Test convert_dict_arg with unknown type."""
    mock_logger = setup_mock_logger(monkeypatch)

    arg = {
        "name": "unknown",
        "type": "unknown_type",
        "description": "Unknown type argument",
    }
    result = convert_dict_arg(arg)
    expected = {"description": "Unknown type argument", "type": "string"}
    assert result == expected
    assert mock_logger.warning.call_count == 1


def test_convert_dict_arg_not_dict():
    """Test convert_dict_arg with non-dict input."""
    with pytest.raises(TypeError) as exc_info:
        convert_dict_arg(cast("dict", "not a dict"))

    assert "Expected argument specification to be a dictionary" in str(exc_info.value)


def test_convert_dict_arg_missing_type():
    """Test convert_dict_arg with missing type."""
    arg = {"name": "test", "description": "Test argument"}
    with pytest.raises(ValueError) as exc_info:
        convert_dict_arg(arg)

    assert "Argument 'type' is missing" in str(exc_info.value)


def test_parse_argument_string_required():
    """Test parsing required argument string."""
    arg_string = "name (str, required): The name of the item"
    result = _parse_argument_string_to_dict(arg_string)
    expected = {
        "name": "name",
        "type": "str",
        "required": True,
        "description": "The name of the item",
    }
    assert result == expected


def test_parse_argument_string_optional():
    """Test parsing optional argument string."""
    arg_string = "count (int, optional): Number of items"
    result = _parse_argument_string_to_dict(arg_string)
    expected = {
        "name": "count",
        "type": "int",
        "required": False,
        "description": "Number of items",
    }
    assert result == expected


def test_parse_argument_string_with_default():
    """Test parsing argument string with default value."""
    arg_string = "timeout (float, optional, default: 30.0): Timeout in seconds"
    result = _parse_argument_string_to_dict(arg_string)
    expected = {
        "name": "timeout",
        "type": "float",
        "required": False,
        "description": "Timeout in seconds",
        "default": "30.0",
    }
    assert result == expected


def test_parse_argument_string_invalid_format(monkeypatch):
    """Test parsing invalid argument string format."""
    mock_logger = setup_mock_logger(monkeypatch)

    arg_string = "invalid format"
    result = _parse_argument_string_to_dict(arg_string)
    assert result is None
    assert mock_logger.warning.call_count == 1


def test_convert_to_openai_tool_format_basic():
    """Test basic tool conversion."""
    tools_dict = {
        "tools": [
            {
                "name": "calculator",
                "description": "Perform basic math operations",
                "arguments": [
                    {
                        "name": "operation",
                        "type": "str",
                        "description": "Math operation",
                        "required": True,
                    },
                    {
                        "name": "x",
                        "type": "float",
                        "description": "First number",
                        "required": True,
                    },
                ],
            }
        ]
    }

    result = convert_to_openai_tool_format(tools_dict)

    assert len(result) == 1
    tool = result[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "calculator"
    assert tool["function"]["description"] == "Perform basic math operations"
    assert "operation" in tool["function"]["parameters"]["properties"]
    assert "x" in tool["function"]["parameters"]["properties"]
    assert tool["function"]["parameters"]["required"] == ["operation", "x"]


def test_convert_to_openai_tool_format_string_arguments():
    """Test tool conversion with string arguments."""
    tools_dict = {
        "tools": [
            {
                "name": "test_tool",
                "description": "A test tool",
                "arguments": [
                    "name (str, required): The name",
                    "count (int, optional): Number of items",
                ],
            }
        ]
    }

    result = convert_to_openai_tool_format(tools_dict)

    assert len(result) == 1
    tool = result[0]
    assert tool["function"]["name"] == "test_tool"
    assert "name" in tool["function"]["parameters"]["properties"]
    assert "count" in tool["function"]["parameters"]["properties"]
    assert tool["function"]["parameters"]["required"] == ["name"]


def test_convert_to_openai_tool_format_no_tools(monkeypatch):
    """Test tool conversion with no tools."""
    mock_logger = setup_mock_logger(monkeypatch)

    tools_dict = {}

    result = convert_to_openai_tool_format(tools_dict)
    assert result == []
    assert mock_logger.warning.call_count == 1


def test_convert_to_openai_tool_format_malformed_tool(monkeypatch):
    """Test tool conversion with malformed tool."""
    mock_logger = setup_mock_logger(monkeypatch)

    tools_dict = {
        "tools": [{"name": "incomplete_tool", "description": "Missing arguments"}]
    }

    result = convert_to_openai_tool_format(tools_dict)
    assert result == []
    assert mock_logger.warning.call_count == 1


def test_convert_to_openai_tool_format_dict_arguments():
    """Test tool conversion with dict/object arguments."""
    tools_dict = {
        "tools": [
            {
                "name": "config_processor",
                "description": "Process configuration data",
                "arguments": [
                    {
                        "name": "config",
                        "type": "dict",
                        "description": "Configuration dictionary",
                        "required": True,
                    },
                    {
                        "name": "metadata",
                        "type": "dict",
                        "description": "Optional metadata",
                        "required": False,
                    },
                ],
            }
        ]
    }

    result = convert_to_openai_tool_format(tools_dict)

    assert len(result) == 1
    tool = result[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "config_processor"
    assert tool["function"]["description"] == "Process configuration data"

    # Check dict type conversion
    config_prop = tool["function"]["parameters"]["properties"]["config"]
    assert config_prop["type"] == "object"
    assert config_prop["description"] == "Configuration dictionary"

    metadata_prop = tool["function"]["parameters"]["properties"]["metadata"]
    assert metadata_prop["type"] == "object"
    assert metadata_prop["description"] == "Optional metadata"

    # Check required fields
    assert tool["function"]["parameters"]["required"] == ["config"]


def test_convert_to_openai_tool_format_mixed_argument_types():
    """Test tool conversion with multiple argument types in one tool."""
    tools_dict = {
        "tools": [
            {
                "name": "complex_processor",
                "description": "Process data with various types",
                "arguments": [
                    {
                        "name": "name",
                        "type": "str",
                        "description": "Process name",
                        "required": True,
                    },
                    {
                        "name": "count",
                        "type": "int",
                        "description": "Number of items",
                        "required": True,
                    },
                    {
                        "name": "threshold",
                        "type": "float",
                        "description": "Processing threshold",
                        "required": False,
                    },
                    {
                        "name": "enabled",
                        "type": "bool",
                        "description": "Whether processing is enabled",
                        "required": False,
                    },
                    {
                        "name": "tags",
                        "type": "list[str]",
                        "description": "Processing tags",
                        "required": False,
                    },
                    {
                        "name": "options",
                        "type": "dict",
                        "description": "Processing options",
                        "required": False,
                    },
                ],
            }
        ]
    }

    result = convert_to_openai_tool_format(tools_dict)

    assert len(result) == 1
    tool = result[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "complex_processor"
    assert tool["function"]["description"] == "Process data with various types"

    properties = tool["function"]["parameters"]["properties"]

    # Check string type
    assert properties["name"]["type"] == "string"
    assert properties["name"]["description"] == "Process name"

    # Check integer type
    assert properties["count"]["type"] == "integer"
    assert properties["count"]["description"] == "Number of items"

    # Check float type
    assert properties["threshold"]["type"] == "number"
    assert properties["threshold"]["description"] == "Processing threshold"

    # Check boolean type
    assert properties["enabled"]["type"] == "boolean"
    assert properties["enabled"]["description"] == "Whether processing is enabled"

    # Check array type
    assert properties["tags"]["type"] == "array"
    assert properties["tags"]["items"]["type"] == "string"
    assert "Provide as an array of strings" in properties["tags"]["description"]

    # Check object type
    assert properties["options"]["type"] == "object"
    assert properties["options"]["description"] == "Processing options"

    # Check required fields (only name and count are required)
    assert tool["function"]["parameters"]["required"] == ["name", "count"]


def test_convert_to_openai_tool_format_array_arguments():
    """Test tool conversion with array/list arguments."""
    tools_dict = {
        "tools": [
            {
                "name": "list_processor",
                "description": "Process a list of items",
                "arguments": [
                    {
                        "name": "items",
                        "type": "list[str]",
                        "description": "List of items to process",
                        "required": True,
                    },
                    {
                        "name": "categories",
                        "type": "list[str]",
                        "description": "Optional categories",
                        "required": False,
                    },
                ],
            }
        ]
    }

    result = convert_to_openai_tool_format(tools_dict)

    assert len(result) == 1
    tool = result[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "list_processor"
    assert tool["function"]["description"] == "Process a list of items"

    # Check array type conversion
    items_prop = tool["function"]["parameters"]["properties"]["items"]
    assert items_prop["type"] == "array"
    assert items_prop["items"]["type"] == "string"
    assert "Provide as an array of strings" in items_prop["description"]

    categories_prop = tool["function"]["parameters"]["properties"]["categories"]
    assert categories_prop["type"] == "array"
    assert categories_prop["items"]["type"] == "string"

    # Check required fields
    assert tool["function"]["parameters"]["required"] == ["items"]


def test_parse_string_argument_required():
    """Test parsing required string argument."""
    arg_string = "path (str, required): Path to the directory"
    result = parse_string_argument(arg_string)
    expected = {
        "name": "path",
        "type": "str",
        "description": "Path to the directory",
        "required": True,
    }
    assert result == expected


def test_parse_string_argument_optional():
    """Test parsing optional string argument."""
    arg_string = "timeout (int, optional): Timeout value"
    result = parse_string_argument(arg_string)
    expected = {
        "name": "timeout",
        "type": "int",
        "description": "Timeout value",
        "required": False,
    }
    assert result == expected


def test_parse_string_argument_invalid_format():
    """Test parsing invalid string argument format."""
    arg_string = "invalid format"
    result = parse_string_argument(arg_string)
    assert result is None


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


def test_tool_conversion_pipeline():
    """Test the complete tool conversion pipeline."""
    tools_dict = {
        "tools": [
            {
                "name": "calculator",
                "description": "Perform math operations",
                "arguments": [
                    "operation (str, required): The operation to perform",
                    "x (float, required): First number",
                    "y (float, optional, default: 1.0): Second number",
                ],
            }
        ]
    }

    # Convert to OpenAI format
    openai_tools = convert_to_openai_tool_format(tools_dict)

    # Verify the conversion worked correctly
    assert len(openai_tools) == 1
    tool = openai_tools[0]
    assert tool["type"] == "function"
    assert tool["function"]["name"] == "calculator"

    # Check parameters
    params = tool["function"]["parameters"]
    assert "operation" in params["properties"]
    assert "x" in params["properties"]
    assert "y" in params["properties"]
    assert params["required"] == ["operation", "x"]
    assert params["properties"]["y"]["default"] == "1.0"


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
