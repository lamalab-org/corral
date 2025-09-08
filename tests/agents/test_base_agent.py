"""Tests for the BaseAgent class."""

import litellm
import openai
import pytest
from litellm.types.utils import Message

from corral.agents.base_agent import BaseAgent
from corral.agents.utils import LiteLLMMessage
from corral.router import CorralRouter

# Import shared mock classes from conftest.py
from .conftest import MockBenchmarkInterface, MockLLMResponse, MockPrompt


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing purposes."""

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> str:
        """Simple implementation for testing."""
        return "test_answer"


@pytest.fixture()
def mock_benchmark_interface():
    """Mock CorralRouter for testing."""
    return MockBenchmarkInterface()


@pytest.fixture()
def concrete_agent(mock_prompt_store):
    """Create a concrete agent instance for testing."""
    return ConcreteAgent(
        model="test-model",
        prompt_store=mock_prompt_store,
        temperature=0.5,
        max_iterations=5,
    )


# Tests for BaseAgent initialization


def test_base_agent_default_initialization():
    """Test agent initialization with default values."""
    agent = ConcreteAgent()

    assert agent.model == "openai/gpt-4o"
    assert agent.max_iterations == 10
    assert agent.api_endpoint is None
    assert agent.temperature == 0.7
    assert agent.messages == []
    assert agent.token_usage == []


def test_base_agent_custom_initialization(mock_prompt_store):
    """Test agent initialization with custom values."""
    agent = ConcreteAgent(
        model="custom-model",
        max_iterations=15,
        api_endpoint="https://custom.endpoint",
        temperature=0.3,
        prompt_store=mock_prompt_store,
    )

    assert agent.model == "custom-model"
    assert agent.max_iterations == 15
    assert agent.api_endpoint == "https://custom.endpoint"
    assert agent.temperature == 0.3
    assert agent.store == mock_prompt_store


def test_base_agent_initialization_with_custom_prompts(mock_prompt_store):
    """Test initialization with custom prompt objects."""
    system_prompt = MockPrompt("Custom system prompt")
    user_prompt = MockPrompt("Custom user prompt")
    extractor_prompt = MockPrompt("Custom extractor prompt")

    agent = ConcreteAgent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extractor_prompt=extractor_prompt,
        prompt_store=mock_prompt_store,
    )

    assert agent.system_prompt == system_prompt
    assert agent.user_prompt == user_prompt
    assert agent.extractor_prompt == extractor_prompt


def test_base_agent_initialization_with_string_prompts(mock_prompt_store):
    """Test initialization with string prompts."""
    system_prompt = "You are a helpful assistant"
    user_prompt = "Task: {{task_guide}}"
    extractor_prompt = "Extract: {{answer}}"

    agent = ConcreteAgent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extractor_prompt=extractor_prompt,
        prompt_store=mock_prompt_store,
    )

    assert agent.system_prompt == system_prompt
    assert agent.user_prompt == user_prompt
    assert agent.extractor_prompt == extractor_prompt


def test_base_agent_default_prompt_store_creation(monkeypatch):
    """Test that default prompt store is created when none provided."""

    class MockPath:
        def __enter__(self):
            return "/mock/path"

        def __exit__(self, *args):
            return None

    monkeypatch.setattr(
        "corral.agents.base_agent.importlib.resources.path", lambda *args: MockPath()
    )

    class MockPromptStore:
        def __init__(self, path=None):
            self.path = path

        def get(self, prompt_id):
            return MockPrompt("Test prompt")

    monkeypatch.setattr("corral.agents.base_agent.PromptStore", MockPromptStore)

    # Actually instantiate the agent to trigger prompt store creation
    ConcreteAgent()


def test_base_agent_kwargs_passed_through(mock_prompt_store):
    """Test that additional kwargs are stored."""
    agent = ConcreteAgent(
        prompt_store=mock_prompt_store, custom_arg="test_value", another_arg=42
    )

    assert agent.kwargs["custom_arg"] == "test_value"
    assert agent.kwargs["another_arg"] == 42


# Tests for BaseAgent LLM response handling


def test_base_agent_get_llm_response_success(monkeypatch, concrete_agent):
    """Test successful LLM response."""
    mock_response = MockLLMResponse("Test response")
    mock_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }

    def mock_llm_call(*args, **kwargs):
        return (mock_response, mock_usage)

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call)

    concrete_agent.messages = [{"role": "user", "content": "Test message"}]

    response = concrete_agent.get_llm_response()

    assert response == mock_response
    assert len(concrete_agent.token_usage) == 1
    assert concrete_agent.token_usage[0] == mock_usage


def test_get_llm_response_with_tools(monkeypatch, concrete_agent):
    """Test LLM response with tools."""
    mock_response = MockLLMResponse()
    mock_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }

    def mock_llm_call(*args, **kwargs):
        return (mock_response, mock_usage)

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call)

    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    concrete_agent.messages = [{"role": "user", "content": "Test message"}]

    response = concrete_agent.get_llm_response(tools=tools)

    assert response == mock_response


def test_get_llm_response_rate_limit_error(monkeypatch, concrete_agent):
    """Test handling of rate limit errors."""

    # Create a mock response for the exception
    class MockResponse:
        def __init__(self):
            self.status_code = 429
            self.request = type("MockRequest", (), {})()
            self.headers = {"x-request-id": "mock-request-id"}

    mock_response = MockResponse()
    mock_body = {"error": {"message": "Rate limit exceeded"}}

    def mock_llm_call_with_error(*args, **kwargs):
        raise openai.RateLimitError(
            "Rate limit exceeded", response=mock_response, body=mock_body
        )

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call_with_error)

    concrete_agent.messages = [
        {"role": "user", "content": "A very long message " * 100},
        {
            "role": "user",
            "content": "Another user message",
        },  # No assistant message to break the loop
    ]

    response = concrete_agent.get_llm_response()

    assert isinstance(response, Message)
    assert response.role == "user"
    assert "RateLimitError" in response.content
    # Check that long messages are truncated (processes in reverse order)
    assert len(concrete_agent.messages[0]["content"]) <= 103  # 100 + "..."


def test_get_llm_response_context_window_error(monkeypatch, concrete_agent):
    """Test handling of context window exceeded errors."""

    def mock_llm_call_with_error(*args, **kwargs):
        raise litellm.ContextWindowExceededError(
            "Context window exceeded", model="test-model", llm_provider="test-provider"
        )

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call_with_error)

    concrete_agent.messages = [
        {"role": "user", "content": "Test message"},
        {"role": "assistant", "content": "Response"},
    ]

    response = concrete_agent.get_llm_response()

    assert isinstance(response, Message)
    assert response.role == "user"
    assert "ContextWindowExceededError" in response.content


def test_get_llm_response_generic_error(monkeypatch, concrete_agent):
    """Test handling of generic errors."""

    def mock_llm_call_with_error(*args, **kwargs):
        raise Exception("Generic error")

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call_with_error)

    with pytest.raises(Exception, match="Generic error"):
        concrete_agent.get_llm_response()


# Tests for BaseAgent run_agent method


def test_run_agent_success(monkeypatch, concrete_agent, mock_benchmark_interface):
    """Test successful run_agent execution."""

    # Mock the run method
    def mock_run(interface, task_id, history=None, task_prompt=None, examples=None):
        return "test_answer"

    monkeypatch.setattr(concrete_agent, "run", mock_run)

    # Mock llm_call for extractor
    mock_response = MockLLMResponse("extracted_answer")

    def mock_llm_call(*args, **kwargs):
        return mock_response

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call)

    concrete_agent.messages = [
        {"role": "user", "content": "Test task"},
        {"role": "assistant", "content": "Test response"},
    ]

    result, usage = concrete_agent.run_agent(
        interface=mock_benchmark_interface,
        task_id="test_task",
        history=[],
        task_prompt="Test prompt",
        examples=["example1", "example2"],
    )

    assert result == "extracted_answer"
    assert isinstance(usage, dict)
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


def test_run_agent_with_system_message(
    monkeypatch, concrete_agent, mock_benchmark_interface
):
    """Test run_agent with system message."""
    # Set up a proper extractor prompt
    concrete_agent.extractor_prompt = MockPrompt("Extract from message: {{message}}")

    # Mock the run method
    def mock_run(interface, task_id, history=None, task_prompt=None, examples=None):
        return "test_answer"

    monkeypatch.setattr(concrete_agent, "run", mock_run)

    # Mock llm_call for extractor
    mock_response = MockLLMResponse("extracted_answer")

    def mock_llm_call(*args, **kwargs):
        return mock_response

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call)

    concrete_agent.messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Test task"},
        {"role": "assistant", "content": "Test response"},
    ]

    result, usage = concrete_agent.run_agent(
        interface=mock_benchmark_interface, task_id="test_task"
    )

    # Check that extractor was called and the result is from the extractor
    assert result == "extracted_answer"


def test_run_agent_with_error_in_answer(
    monkeypatch, concrete_agent, mock_benchmark_interface
):
    """Test run_agent when answer contains error."""

    def mock_run(interface, task_id, history=None, task_prompt=None, examples=None):
        return "Error: Something went wrong"

    monkeypatch.setattr(concrete_agent, "run", mock_run)

    result, usage = concrete_agent.run_agent(
        interface=mock_benchmark_interface, task_id="test_task"
    )

    assert "Error: Something went wrong" in result
    assert isinstance(usage, dict)


def test_run_agent_with_exception(
    monkeypatch, concrete_agent, mock_benchmark_interface
):
    """Test run_agent when run method raises exception."""

    def mock_run_with_error(
        interface, task_id, history=None, task_prompt=None, examples=None
    ):
        raise Exception("Run failed")

    monkeypatch.setattr(concrete_agent, "run", mock_run_with_error)

    result, usage = concrete_agent.run_agent(
        interface=mock_benchmark_interface, task_id="test_task"
    )

    assert "Error running agent: Run failed" in result
    assert isinstance(usage, dict)


def test_run_agent_verbose_mode(monkeypatch, concrete_agent, mock_benchmark_interface):
    """Test run_agent in verbose mode."""

    # Mock the run method
    def mock_run(interface, task_id, history=None, task_prompt=None, examples=None):
        return "test_answer"

    monkeypatch.setattr(concrete_agent, "run", mock_run)

    # Mock llm_call for extractor
    mock_response = MockLLMResponse("extracted_answer")

    def mock_llm_call(*args, **kwargs):
        return mock_response

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call)

    # Mock save_agent_messages
    save_calls = []

    def mock_save(*args):
        save_calls.append(args)

    monkeypatch.setattr("corral.agents.base_agent.save_agent_messages", mock_save)

    concrete_agent.messages = [
        {"role": "user", "content": "Test task"},
        {"role": "assistant", "content": "Test response"},
    ]

    result, usage = concrete_agent.run_agent(
        interface=mock_benchmark_interface,
        task_id="test_task",
        verbose=True,
    )

    assert len(save_calls) == 1
    call_args = save_calls[0]
    assert call_args[0] == concrete_agent.messages
    assert call_args[1] == "test_task"
    assert call_args[2] == "ConcreteAgent"


def test_run_agent_extractor_error(
    monkeypatch, concrete_agent, mock_benchmark_interface
):
    """Test run_agent when extractor fails."""

    # Mock the run method
    def mock_run(interface, task_id, history=None, task_prompt=None, examples=None):
        return "test_answer"

    monkeypatch.setattr(concrete_agent, "run", mock_run)

    # Mock llm_call to raise an error
    def mock_llm_call_with_error(*args, **kwargs):
        raise Exception("Extractor failed")

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call_with_error)

    concrete_agent.messages = [
        {"role": "user", "content": "Test task"},
        {"role": "assistant", "content": "Test response"},
    ]

    result, usage = concrete_agent.run_agent(
        interface=mock_benchmark_interface, task_id="test_task"
    )

    assert result == "test_answer"  # Should return original answer
    assert isinstance(usage, dict)


# Tests for BaseAgent token usage tracking


def test_get_total_token_usage_empty(concrete_agent):
    """Test token usage calculation with no usage data."""
    usage = concrete_agent.get_total_token_usage()

    assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def test_get_total_token_usage_with_data(concrete_agent):
    """Test token usage calculation with usage data."""
    concrete_agent.token_usage = [
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        {"prompt_tokens": 200, "completion_tokens": 75, "total_tokens": 275},
        {"prompt_tokens": 50, "completion_tokens": 25, "total_tokens": 75},
    ]

    usage = concrete_agent.get_total_token_usage()

    assert usage == {
        "prompt_tokens": 350,
        "completion_tokens": 150,
        "total_tokens": 500,
    }


def test_get_total_token_usage_with_missing_keys(concrete_agent):
    """Test token usage calculation with missing keys."""
    concrete_agent.token_usage = [
        {"prompt_tokens": 100, "total_tokens": 150},  # Missing completion_tokens
        {"completion_tokens": 75, "total_tokens": 275},  # Missing prompt_tokens
        {},  # Missing all keys
    ]

    usage = concrete_agent.get_total_token_usage()

    assert usage == {
        "prompt_tokens": 100,
        "completion_tokens": 75,
        "total_tokens": 425,
    }


def test_reset_token_usage(concrete_agent):
    """Test resetting token usage."""
    concrete_agent.token_usage = [
        {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
    ]

    concrete_agent.reset_token_usage()

    assert not concrete_agent.token_usage


# Tests for BaseAgent abstract methods


def test_cannot_instantiate_base_agent():
    """Test that BaseAgent cannot be instantiated directly."""
    with pytest.raises(TypeError, match="Can't instantiate abstract class"):
        BaseAgent()


def test_concrete_agent_implements_run(concrete_agent, mock_benchmark_interface):
    """Test that concrete agent implements run method."""
    result = concrete_agent.run(interface=mock_benchmark_interface, task_id="test_task")

    assert result == "test_answer"


# Tests for BaseAgent prompt handling


def test_prompt_fill_method():
    """Test that prompts have fill method."""
    mock_prompt = MockPrompt("Test {{variable}}")
    filled = mock_prompt.fill({"variable": "value"})

    assert filled == "Test value"


def test_extractor_prompt_filling(
    monkeypatch, concrete_agent, mock_benchmark_interface
):
    """Test that extractor prompt is filled correctly."""
    concrete_agent.extractor_prompt = MockPrompt(
        "Answer: {{answer}}, Message: {{message}}"
    )

    # Mock the run method
    def mock_run(interface, task_id, history=None, task_prompt=None, examples=None):
        return "test_answer"

    monkeypatch.setattr(concrete_agent, "run", mock_run)

    # Mock llm_call for extractor and capture the call
    call_args_captured = []
    mock_response = MockLLMResponse("extracted_answer")

    def mock_llm_call(*args, **kwargs):
        call_args_captured.append((args, kwargs))
        return mock_response

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call)

    concrete_agent.messages = [
        {"role": "user", "content": "Test task"},
        {"role": "assistant", "content": "Test response"},
    ]

    result, usage = concrete_agent.run_agent(
        interface=mock_benchmark_interface, task_id="test_task"
    )

    # Check that the extractor prompt was called with correct parameters
    assert len(call_args_captured) == 1
    call_args = call_args_captured[0]
    prompt_content = call_args[1]["messages"][0]["content"]
    assert "Answer: test_answer" in prompt_content
    assert "Message: " in prompt_content
