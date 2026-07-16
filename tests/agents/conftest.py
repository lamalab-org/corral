"""Shared test fixtures and mock classes for agent tests."""

import json
from typing import Any
from urllib.parse import quote, urlencode

import pytest

from corral.types import ToolResponse


class MockPrompt:
    """Mock prompt class that implements the required fill method."""

    def __init__(self, content: str):
        self.content = content

    def fill(self, replacements: dict[str, Any]) -> str:
        """Fill the prompt with replacements, same behavior as StringPrompt."""
        result = self.content
        for key, value in replacements.items():
            # Fill all keys including framework keys
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result


class MockBenchmarkInterface:
    """Mock BenchmarkInterface for testing."""

    def __init__(self):
        self.base_url = "http://test-server:8000"
        self.current_verbosity = "brief"
        self.task_guide = "Test task guide"
        self.task_prompt = "Test task prompt"
        self.available_tools = {
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "test_tool",
                        "description": "A test tool",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Test query",
                                }
                            },
                            "required": ["query"],
                        },
                    },
                }
            ]
        }
        self.tool_responses = []
        self.tool_calls = []
        self.call_counts = {}
        self.get_last_score = None  # Can be overridden in tests

    def get_task_guide(self, task_id: str) -> str:
        self._record_call("get_task_guide", task_id)
        return self.task_guide

    def get_task_prompt(self, task_id: str) -> str:
        self._record_call("get_task_prompt", task_id)
        return self.task_prompt

    def get_available_tools_for_task(
        self, task_id: str, verbosity: str | None = None
    ) -> dict:
        self._record_call("get_available_tools_for_task", task_id)
        return self.available_tools

    def get_mcp_tool_schema(self, task_id: str, verbosity: str | None = None) -> dict:
        self._record_call("get_mcp_tool_schema", task_id)
        return {"tools": [], "mcp_schema_sha256": "deadbeef"}

    def mcp_url(self, task_id: str, verbosity: str | None = None) -> str:
        # Mirror CorralRouter.mcp_url so agents get the task-scoped MCP URL.
        self._record_call("mcp_url", task_id)
        verbosity = verbosity or self.current_verbosity or "brief"
        base_url = self.base_url.rstrip("/")
        encoded_task_id = quote(str(task_id), safe="")
        query = urlencode({"verbosity": verbosity})
        return f"{base_url}/tasks/{encoded_task_id}/mcp/?{query}"

    def execute_tool(
        self, task_id: str, tool_name: str, arguments: dict
    ) -> ToolResponse:
        self._record_call("execute_tool", task_id, tool_name, arguments)
        self.tool_calls.append(
            {"task_id": task_id, "tool_name": tool_name, "arguments": arguments}
        )
        if self.tool_responses:
            return self.tool_responses.pop(0)
        return ToolResponse(success=True, result="Tool execution result", error=None)

    def _record_call(self, method_name: str, *args, **kwargs):
        """Record method calls for verification."""
        if method_name not in self.call_counts:
            self.call_counts[method_name] = 0
        self.call_counts[method_name] += 1

    def assert_called_once_with(self, method_name: str, *expected_args):
        """Assert that a method was called once with expected arguments."""
        assert (
            self.call_counts.get(method_name, 0) == 1
        ), f"{method_name} was not called exactly once"

    def assert_not_called(self, method_name: str):
        """Assert that a method was not called."""
        assert (
            self.call_counts.get(method_name, 0) == 0
        ), f"{method_name} was called when it shouldn't have been"


class MockMessage:
    """Mock message class that mimics litellm's Message object."""

    def __init__(self, content: str | None = None, tool_calls: list | None = None):
        self.content = content
        self.tool_calls = tool_calls or []
        self.role = "assistant"
        self.id = None  # Can be set later


class MockLLMResponse:
    """Mock LLM response for testing, matching the LLMResponse interface."""

    def __init__(
        self,
        content: str | None = None,
        tool_calls: list | None = None,
        usage: dict | None = None,
        response_id: str | None = None,
        logprobs: Any | None = None,
    ):
        # Create underlying message object
        self.message = MockMessage(content=content, tool_calls=tool_calls or [])
        self._usage = usage
        self._id = response_id
        self._logprobs = logprobs

    @property
    def content(self) -> str | None:
        """Get message content"""
        return self.message.content

    @property
    def tool_calls(self) -> list:
        """Get tool calls from the message"""
        return self.message.tool_calls

    @property
    def usage(self) -> dict | None:
        """Get token usage from metadata"""
        return self._usage

    @property
    def id(self) -> str | None:
        """Get message ID from metadata"""
        return self._id

    @property
    def logprobs(self) -> Any:
        """Get logprobs from metadata"""
        return self._logprobs


class MockToolCall:
    """Mock tool call for testing."""

    def __init__(self, call_id: str, name: str, arguments: dict):
        self.id = call_id
        self.function = type(
            "MockFunction", (), {"name": name, "arguments": json.dumps(arguments)}
        )()


@pytest.fixture()
def mock_prompt_store():
    """Mock PromptStore for testing."""

    class MockPromptStore:
        def get(self, prompt_name: str):
            # Return specific prompts for known IDs
            if "system_prompt" in prompt_name:
                return MockPrompt("You are a helpful assistant.")
            elif "extractor_prompt" in prompt_name:
                return MockPrompt("Extract the answer from: {{answer}}")
            elif "user_prompt" in prompt_name:
                return MockPrompt("Task: {{task_guide}}")
            else:
                return MockPrompt("Test prompt: {{task_guide}}")

    return MockPromptStore()


@pytest.fixture(autouse=True)
def mock_promptstore_module(monkeypatch):
    """Automatically mock the promptstore module for all tests."""

    class MockPromptStoreClass:
        """Mock PromptStore class that mimics promptstore.PromptStore."""

        def __init__(self, *args, **kwargs):
            # Ignore initialization arguments
            pass

        def get(self, prompt_name: str):
            """Return mock prompts based on prompt_name."""
            if "system_prompt" in prompt_name:
                return MockPrompt("You are a helpful assistant.")
            elif "extractor_prompt" in prompt_name:
                return MockPrompt("Extract the answer from: {{answer}}")
            elif "user_prompt" in prompt_name:
                return MockPrompt("Task: {{task_guide}}")
            else:
                return MockPrompt("Test prompt: {{task_guide}}")

    # Mock the PromptStore class in the promptstore module
    monkeypatch.setattr("promptstore.PromptStore", MockPromptStoreClass)
    # Also mock it where it's imported in the agents module
    monkeypatch.setattr("corral.agents.base_agent.PromptStore", MockPromptStoreClass)


@pytest.fixture()
def mock_interface():
    """Mock BenchmarkInterface for testing."""
    return MockBenchmarkInterface()


@pytest.fixture()
def mock_tool_response():
    """Mock ToolResponse for testing."""
    return ToolResponse(success=True, result="Tool execution result", error=None)


@pytest.fixture()
def mock_tool_response_with_error():
    """Mock ToolResponse with error for testing."""
    return ToolResponse(success=False, result=None, error="Tool execution failed")


class Call:
    """Represents a function call for testing."""

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def __eq__(self, other):
        if isinstance(other, tuple) and len(other) == 2:
            args, kwargs = other
            return self.args == args and self.kwargs == kwargs
        return False

    def __hash__(self):
        return hash((self.args, tuple(sorted(self.kwargs.items()))))

    def __repr__(self):
        return f"Call({self.args!r}, {self.kwargs!r})"


def call(*args, **kwargs):
    """Helper function to create Call objects."""
    return Call(*args, **kwargs)


class MockFunction:
    """A simple mock function for testing without unittest.mock."""

    def __init__(self, return_value=None, side_effect=None):
        self.return_value = return_value
        self.side_effect = side_effect
        self.call_count = 0
        self.call_args_list = []

    def __call__(self, *args, **kwargs):
        self.call_count += 1
        self.call_args_list.append((args, kwargs))

        if self.side_effect is not None:
            if isinstance(self.side_effect, Exception):
                raise self.side_effect
            if callable(self.side_effect):
                return self.side_effect(*args, **kwargs)
            if isinstance(self.side_effect, list):
                if self.call_count <= len(self.side_effect):
                    result = self.side_effect[self.call_count - 1]
                    if isinstance(result, Exception):
                        raise result
                    return result
                raise IndexError("side_effect list exhausted")

        return self.return_value

    def assert_called_once_with(self, *args, **kwargs):
        """Assert the function was called once with specific arguments."""
        assert self.call_count == 1, f"Expected 1 call, got {self.call_count}"
        if args or kwargs:
            assert self.call_args_list[0] == (
                args,
                kwargs,
            ), f"Expected {(args, kwargs)}, got {self.call_args_list[0]}"

    def assert_not_called(self):
        """Assert the function was not called."""
        assert self.call_count == 0, f"Expected 0 calls, got {self.call_count}"

    def assert_has_calls(self, expected_calls):
        """Assert the function was called with the expected calls."""
        assert len(self.call_args_list) == len(
            expected_calls
        ), f"Expected {len(expected_calls)} calls, got {len(self.call_args_list)}"
        for i, expected in enumerate(expected_calls):
            assert (
                self.call_args_list[i] == expected
            ), f"Call {i}: expected {expected}, got {self.call_args_list[i]}"
