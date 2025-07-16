import json
import tempfile
from pathlib import Path
from typing import cast
from unittest.mock import Mock, patch

import pytest

from corral.agents.utils import (
    RETRY_EXCEPTIONS,
    TYPE_MAPPING,
    LiteLLMMessage,
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


def test_type_mapping_completeness():
    """Test that TYPE_MAPPING contains expected type mappings."""
    expected_mappings = {
        "str": "string",
        "bool": "boolean",
        "int": "integer",
        "float": "number",
        "list[str]": "array",
        "none": "null",
        "dict": "object",
    }
    assert expected_mappings == TYPE_MAPPING


def test_retry_exceptions_tuple():
    """Test that RETRY_EXCEPTIONS contains expected exception types."""
    try:
        import openai

        expected_exceptions = (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.APIError,
            openai.APIStatusError,
            openai.InternalServerError,
        )
        assert expected_exceptions == RETRY_EXCEPTIONS
    except ImportError:
        # Skip test if openai not available
        pass


@patch("corral.agents.utils.logger")
def test_before_sleep_loguru_logs_retry_info(mock_logger):
    """Test that before_sleep_loguru logs retry information."""
    mock_retry_state = Mock()
    mock_retry_state.attempt_number = 2
    mock_retry_state.next_action.sleep = 30.0

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
    }
    assert msg2["tool_call_id"] == "call_123"
    assert msg2["name"] == "assistant"


@patch("corral.agents.utils.litellm")
def test_llm_call_basic(mock_litellm):
    """Test basic llm_call without tools."""
    mock_response = Mock()
    mock_message = Mock()
    mock_response.choices = [Mock(message=mock_message)]
    mock_litellm.completion.return_value = mock_response

    messages = cast(list[LiteLLMMessage], [{"role": "user", "content": "Hello"}])

    result = llm_call(model="gpt-3.5-turbo", messages=messages, temperature=0.7)

    assert result == mock_message
    mock_litellm.completion.assert_called_once()
    call_args = mock_litellm.completion.call_args[1]
    assert call_args["model"] == "gpt-3.5-turbo"
    assert call_args["messages"] == messages
    assert call_args["temperature"] == 0.7
    assert call_args["api_base"] is None


@patch("corral.agents.utils.litellm")
def test_llm_call_with_tools(mock_litellm):
    """Test llm_call with tools."""
    mock_response = Mock()
    mock_message = Mock()
    mock_response.choices = [Mock(message=mock_message)]
    mock_litellm.completion.return_value = mock_response

    messages = cast(list[LiteLLMMessage], [{"role": "user", "content": "Hello"}])
    tools = [{"type": "function", "function": {"name": "test_tool"}}]

    result = llm_call(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7, tools=tools
    )

    assert result == mock_message
    call_args = mock_litellm.completion.call_args[1]
    assert call_args["tools"] == tools
    assert call_args["tool_choice"] == "auto"


@patch("corral.agents.utils.litellm")
def test_llm_call_anthropic_model(mock_litellm):
    """Test llm_call with anthropic model adds max_tokens."""
    mock_response = Mock()
    mock_message = Mock()
    mock_response.choices = [Mock(message=mock_message)]
    mock_litellm.completion.return_value = mock_response

    messages = cast(list[LiteLLMMessage], [{"role": "user", "content": "Hello"}])

    result = llm_call(
        model="anthropic/claude-3-sonnet", messages=messages, temperature=0.7
    )

    assert result == mock_message
    call_args = mock_litellm.completion.call_args[1]
    assert call_args["max_tokens"] == 8192


@patch("corral.agents.utils.litellm")
def test_llm_call_with_usage_info(mock_litellm):
    """Test llm_call with return_usage=True."""
    mock_response = Mock()
    mock_message = Mock()
    mock_usage = Mock()
    mock_usage.prompt_tokens = 10
    mock_usage.completion_tokens = 20
    mock_usage.total_tokens = 30
    mock_response.choices = [Mock(message=mock_message)]
    mock_response.usage = mock_usage
    mock_litellm.completion.return_value = mock_response

    messages = cast(list[LiteLLMMessage], [{"role": "user", "content": "Hello"}])

    result = llm_call(
        model="gpt-3.5-turbo", messages=messages, temperature=0.7, return_usage=True
    )

    assert isinstance(result, tuple)
    message, usage_info = result
    assert message == mock_message
    assert usage_info == {
        "prompt_tokens": 10,
        "completion_tokens": 20,
        "total_tokens": 30,
    }


@patch("corral.agents.utils.litellm")
def test_llm_call_exception_handling(mock_litellm):
    """Test llm_call exception handling."""
    mock_litellm.completion.side_effect = Exception("API Error")

    messages = cast(list[LiteLLMMessage], [{"role": "user", "content": "Hello"}])

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


@patch("corral.agents.utils.logger")
def test_convert_dict_arg_unknown_type(mock_logger):
    """Test convert_dict_arg with unknown type."""
    arg = {
        "name": "unknown",
        "type": "unknown_type",
        "description": "Unknown type argument",
    }
    result = convert_dict_arg(arg)
    expected = {"description": "Unknown type argument", "type": "string"}
    assert result == expected
    mock_logger.warning.assert_called_once()


def test_convert_dict_arg_not_dict():
    """Test convert_dict_arg with non-dict input."""
    with pytest.raises(TypeError) as exc_info:
        convert_dict_arg(cast(dict, "not a dict"))

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


@patch("corral.agents.utils.logger")
def test_parse_argument_string_invalid_format(mock_logger):
    """Test parsing invalid argument string format."""
    arg_string = "invalid format"
    result = _parse_argument_string_to_dict(arg_string)
    assert result is None
    mock_logger.warning.assert_called_once()


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


@patch("corral.agents.utils.logger")
def test_convert_to_openai_tool_format_no_tools(mock_logger):
    """Test tool conversion with no tools."""
    tools_dict = {}

    result = convert_to_openai_tool_format(tools_dict)
    assert result == []
    mock_logger.warning.assert_called_once()


@patch("corral.agents.utils.logger")
def test_convert_to_openai_tool_format_malformed_tool(mock_logger):
    """Test tool conversion with malformed tool."""
    tools_dict = {
        "tools": [{"name": "incomplete_tool", "description": "Missing arguments"}]
    }

    result = convert_to_openai_tool_format(tools_dict)
    assert result == []
    mock_logger.warning.assert_called_once()


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
        list[LiteLLMMessage],
        [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"},
        ],
    )

    result = serialize_messages(messages)

    assert result == messages


def test_serialize_messages_with_tool_calls():
    """Test serializing messages with tool calls."""
    mock_tool_call = Mock()
    mock_tool_call.id = "call_123"
    mock_tool_call.function.name = "test_function"
    mock_tool_call.function.arguments = '{"arg": "value"}'

    mock_message = Mock()
    mock_message.role = "assistant"
    mock_message.content = "I'll help you"
    mock_message.tool_calls = [mock_tool_call]
    mock_message.tool_call_id = None
    mock_message.name = None

    result = serialize_messages([mock_message])

    assert len(result) == 1
    message = result[0]
    assert message["role"] == "assistant"
    assert message["content"] == "I'll help you"
    assert len(message["tool_calls"]) == 1
    assert message["tool_calls"][0]["id"] == "call_123"
    assert message["tool_calls"][0]["function"]["name"] == "test_function"
    assert message["tool_calls"][0]["function"]["arguments"] == '{"arg": "value"}'


def test_save_agent_messages_basic():
    """Test basic save_agent_messages functionality."""
    with tempfile.TemporaryDirectory() as temp_dir:
        messages = cast(
            list[LiteLLMMessage],
            [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
        )

        result_path = save_agent_messages(
            messages=messages,
            task_id="test_task",
            agent_name="test_agent",
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


def test_save_agent_messages_with_tools():
    """Test save_agent_messages with tools."""
    with tempfile.TemporaryDirectory() as temp_dir:
        messages = cast(list[LiteLLMMessage], [{"role": "user", "content": "Hello"}])
        tools = [{"name": "test_tool", "description": "A test tool"}]

        result_path = save_agent_messages(
            messages=messages,
            task_id="test_task",
            agent_name="test_agent",
            output_dir=temp_dir,
            tools=tools,
        )

        with open(result_path) as f:
            data = json.load(f)

        assert data["tools"] == tools


def test_save_agent_messages_creates_directory():
    """Test that save_agent_messages creates output directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "new_logs"

        messages = cast(list[LiteLLMMessage], [{"role": "user", "content": "Hello"}])

        result_path = save_agent_messages(
            messages=messages,
            task_id="test_task",
            agent_name="test_agent",
            output_dir=str(output_dir),
        )

        assert output_dir.exists()
        assert Path(result_path).exists()


@patch("corral.agents.utils.datetime")
def test_save_agent_messages_filename_format(mock_datetime):
    """Test that save_agent_messages generates correct filename format."""
    with tempfile.TemporaryDirectory() as temp_dir:
        messages = cast(list[LiteLLMMessage], [{"role": "user", "content": "Hello"}])

        mock_datetime.now.return_value.strftime.return_value = "20240101_120000"

        result_path = save_agent_messages(
            messages=messages,
            task_id="test_task",
            agent_name="test_agent",
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
        mock_message1 = Mock()
        mock_message1.role = "user"
        mock_message1.content = "Hello"
        mock_message1.tool_call_id = None
        mock_message1.name = None
        mock_message1.tool_calls = None

        mock_message2 = {"role": "assistant", "content": "Hi there"}

        messages = [mock_message1, mock_message2]

        # Save messages
        result_path = save_agent_messages(
            messages=messages,
            task_id="integration_test",
            agent_name="test_agent",
            output_dir=temp_dir,
        )

        # Load and verify
        with open(result_path) as f:
            data = json.load(f)

        assert len(data["messages"]) == 2
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Hello"
        assert data["messages"][1] == mock_message2
