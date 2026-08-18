"""Shared test fixtures and mock classes for agent tests."""

import json
from typing import Any

import pytest


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
            elif "user_prompt" in prompt_name:
                return MockPrompt("Task: {{task_guide}}")
            else:
                return MockPrompt("Test prompt: {{task_guide}}")

    # Mock the PromptStore class in the promptstore module
    monkeypatch.setattr("promptstore.PromptStore", MockPromptStoreClass)
    # Also mock it where it's imported in the agents module
    monkeypatch.setattr("corral.agents.base_agent.PromptStore", MockPromptStoreClass)


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
