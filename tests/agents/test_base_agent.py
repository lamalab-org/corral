"""Tests for the BaseAgent class."""

import pytest

from corral.agents.base_agent import BaseAgent
from corral.agents.schema import AgentRunResult
from corral.router import CorralRouter

# Import shared mock classes from conftest.py
from .conftest import MockBenchmarkInterface, MockLLMResponse, MockPrompt


def _unpack(result: AgentRunResult) -> tuple:
    """Unpack an AgentRunResult into (answer, messages, token_usage)."""
    return result.answer, result.messages, result.token_usage


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing purposes."""

    def __init__(
        self,
        model: str = "openai/gpt-4o",
        max_iterations: int = 10,
        api_endpoint: str | None = None,
        system_prompt: str | None = None,
        user_prompt: str | None = None,
        extractor_prompt: str | None = None,
        surrender_prompt: str | None = None,
        temperature: float = 0.7,
        **kwargs,
    ):
        """Initialize the concrete agent for testing."""
        # Provide default user_prompt if not specified (like ReActAgent does)
        if user_prompt is None:
            user_prompt = "Task: {{task_guide}}"

        super().__init__(
            model=model,
            max_iterations=max_iterations,
            api_endpoint=api_endpoint,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            extractor_prompt=extractor_prompt,
            surrender_prompt=surrender_prompt,
            temperature=temperature,
            **kwargs,
        )

    def run(
        self,
        interface: CorralRouter,
        task_id: str,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
        **kwargs,
    ) -> str:
        """Simple implementation for testing."""
        return "test_answer"


@pytest.fixture()
def mock_benchmark_interface():
    """Mock CorralRouter for testing."""
    return MockBenchmarkInterface()


@pytest.fixture()
def concrete_agent():
    """Create a concrete agent instance for testing."""
    return ConcreteAgent(
        model="test-model",
        system_prompt="You are a helpful assistant.",
        extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        temperature=0.5,
        max_iterations=5,
    )


# Tests for BaseAgent initialization


def test_base_agent_default_initialization():
    """Test agent initialization with default values."""
    agent = ConcreteAgent(
        system_prompt="You are a helpful assistant.",
        extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
    )

    assert agent.model == "openai/gpt-4o"
    assert agent.max_iterations == 10
    assert agent.api_endpoint is None
    assert agent.temperature == 0.7
    assert agent.messages == []
    assert agent.token_usage == {}


def test_base_agent_custom_initialization():
    """Test agent initialization with custom values."""
    agent = ConcreteAgent(
        model="custom-model",
        max_iterations=15,
        api_endpoint="https://custom.endpoint",
        temperature=0.3,
        system_prompt="You are a helpful assistant.",
        extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
    )

    assert agent.model == "custom-model"
    assert agent.max_iterations == 15
    assert agent.api_endpoint == "https://custom.endpoint"
    assert agent.temperature == 0.3
    assert agent.store is not None  # Has a store, not necessarily the mock


def test_base_agent_initialization_with_custom_prompts():
    """Test initialization with custom prompt objects."""
    system_prompt = MockPrompt("Custom system prompt")
    user_prompt = MockPrompt("Custom user prompt")
    extractor_prompt = MockPrompt("Custom extractor prompt")

    agent = ConcreteAgent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extractor_prompt=extractor_prompt,
    )

    # system_prompt is converted to string by fill({})
    assert agent.system_prompt == "Custom system prompt"
    # user_prompt and extractor_prompt remain as objects
    assert hasattr(agent.user_prompt, "fill")
    assert hasattr(agent.extractor_prompt, "fill")


def test_base_agent_initialization_with_string_prompts():
    """Test initialization with string prompts."""
    system_prompt = "You are a helpful assistant"
    user_prompt = "Task: {{task_guide}}"
    extractor_prompt = "Extract: {{answer}}"

    agent = ConcreteAgent(
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        extractor_prompt=extractor_prompt,
    )

    # system_prompt is converted to string
    assert agent.system_prompt == system_prompt
    # user_prompt and extractor_prompt are wrapped in StringPrompt
    assert hasattr(agent.user_prompt, "fill")
    assert hasattr(agent.extractor_prompt, "fill")


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


def test_base_agent_kwargs_passed_through():
    """Test that additional kwargs are stored."""
    agent = ConcreteAgent(
        system_prompt="You are a helpful assistant.",
        extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
        custom_arg="test_value",
        another_arg=42,
    )

    assert agent.kwargs["custom_arg"] == "test_value"
    assert agent.kwargs["another_arg"] == 42


# Tests for BaseAgent LLM response handling


def test_base_agent_get_llm_response_success(monkeypatch, concrete_agent):
    """Test successful LLM response."""
    mock_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    mock_response = MockLLMResponse("Test response", usage=mock_usage)

    def mock_llm_call(*args, **kwargs):
        return mock_response

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call)

    concrete_agent.messages = [{"role": "user", "content": "Test message"}]

    response = concrete_agent.get_llm_response()

    assert response == mock_response
    # token_usage is now a dict with the three token counts
    assert len(concrete_agent.token_usage) == 3
    assert concrete_agent.token_usage == mock_usage


def test_get_llm_response_with_tools(monkeypatch, concrete_agent):
    """Test LLM response with tools."""
    mock_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    mock_response = MockLLMResponse(usage=mock_usage)

    def mock_llm_call(*args, **kwargs):
        return mock_response

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call)

    tools = [{"type": "function", "function": {"name": "test_tool"}}]
    concrete_agent.messages = [{"role": "user", "content": "Test message"}]

    response = concrete_agent.get_llm_response(tools=tools)

    assert response == mock_response


def test_get_llm_response_generic_error(monkeypatch, concrete_agent):
    """Test handling of generic errors."""

    def mock_llm_call_with_error(*args, **kwargs):
        raise Exception("Generic error")

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call_with_error)

    # Need to provide messages so count_tokens_and_add doesn't fail
    concrete_agent.messages = [{"role": "user", "content": "Test message"}]

    with pytest.raises(Exception, match="Generic error"):
        concrete_agent.get_llm_response()


# Tests for BaseAgent run_agent method


def test_run_agent_success(monkeypatch, concrete_agent, mock_benchmark_interface):
    """Test successful run_agent execution."""

    # Mock the run method
    def mock_run(
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        enable_surrender=False,
    ):
        return "test_answer"

    monkeypatch.setattr(concrete_agent, "run", mock_run)

    # Mock llm_call for extractor
    mock_response = MockLLMResponse("extracted_answer")

    def mock_llm_call(*args, **kwargs):
        return mock_response

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call)

    # Mock save_agent_messages to capture both args and kwargs
    save_calls = []

    def mock_save(*args, **kwargs):
        save_calls.append((args, kwargs))

    monkeypatch.setattr("corral.agents.base_agent.save_agent_messages", mock_save)

    concrete_agent.messages = [
        {"role": "user", "content": "Test task"},
        {"role": "assistant", "content": "Test response"},
    ]

    result, messages, usage = _unpack(
        concrete_agent.run_agent(
            interface=mock_benchmark_interface,
            task_id="test_task",
            verbose=True,
        )
    )

    assert len(save_calls) == 1
    call_args, call_kwargs = save_calls[0]
    assert call_kwargs["messages"] == concrete_agent.messages
    assert call_kwargs["task_id"] == "test_task"
    assert call_kwargs["agent_name"] == "ConcreteAgent"
    assert call_kwargs["model"] == concrete_agent.model
    assert "tools" in call_kwargs  # tools may be None
    assert "tool_verbosity" in call_kwargs

    # Now test with a system message and no verbose
    concrete_agent.messages = [
        {"role": "system", "content": "System message"},
        {"role": "user", "content": "Test task"},
        {"role": "assistant", "content": "Test response"},
    ]

    result, messages, usage = _unpack(
        concrete_agent.run_agent(
            interface=mock_benchmark_interface, task_id="test_task"
        )
    )

    # Check that extractor was called and the result is from the extractor
    assert result == "extracted_answer"


def test_run_agent_with_error_in_answer(
    monkeypatch, concrete_agent, mock_benchmark_interface
):
    """Test run_agent when answer contains error."""

    def mock_run(
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        enable_surrender=False,
    ):
        return "Error: Something went wrong"

    monkeypatch.setattr(concrete_agent, "run", mock_run)

    result, messages, usage = _unpack(
        concrete_agent.run_agent(
            interface=mock_benchmark_interface, task_id="test_task"
        )
    )

    assert "Error: Something went wrong" in result
    assert isinstance(usage, dict)


def test_run_agent_with_exception(
    monkeypatch, concrete_agent, mock_benchmark_interface
):
    """Test run_agent when run method raises exception."""

    def mock_run_with_error(
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        enable_surrender=False,
    ):
        raise Exception("Run failed")

    monkeypatch.setattr(concrete_agent, "run", mock_run_with_error)

    result, messages, usage = _unpack(
        concrete_agent.run_agent(
            interface=mock_benchmark_interface, task_id="test_task"
        )
    )

    assert "Error running agent" in result
    assert isinstance(usage, dict)


def test_run_agent_cancelled_skips_transcript_save(
    monkeypatch, concrete_agent, mock_benchmark_interface
):
    """A cancelled (Ctrl+C) run must NOT persist its partial transcript.

    Cancellation surfaces as a `BaseException` (`KeyboardInterrupt` /
    `CancelledError`), which is *not* caught by the `except Exception` arm.
    The transcript save in the `finally` block must be skipped so an abandoned
    trial is not recorded, and the cancellation must propagate untouched.
    """

    def mock_run_cancelled(
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        enable_surrender=False,
    ):
        raise KeyboardInterrupt

    monkeypatch.setattr(concrete_agent, "run", mock_run_cancelled)

    save_calls = []
    monkeypatch.setattr(
        "corral.agents.base_agent.save_agent_messages",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    with pytest.raises(KeyboardInterrupt):
        concrete_agent.run_agent(
            interface=mock_benchmark_interface,
            task_id="test_task",
            verbose=True,
        )

    assert save_calls == []


def test_arun_agent_cancelled_skips_transcript_save(
    monkeypatch, concrete_agent, mock_benchmark_interface
):
    """Async twin of :func:`test_run_agent_cancelled_skips_transcript_save`.

    A cancelled concurrent trial (the `abench` path) must likewise skip the
    verbose transcript save and re-raise the cancellation.
    """
    import anyio

    async def mock_arun_cancelled(
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        enable_surrender=False,
    ):
        raise KeyboardInterrupt

    monkeypatch.setattr(concrete_agent, "arun", mock_arun_cancelled)

    save_calls = []
    monkeypatch.setattr(
        "corral.agents.base_agent.save_agent_messages",
        lambda *args, **kwargs: save_calls.append((args, kwargs)),
    )

    async def _go():
        return await concrete_agent.arun_agent(
            interface=mock_benchmark_interface,
            task_id="test_task",
            verbose=True,
        )

    with pytest.raises(KeyboardInterrupt):
        anyio.run(_go)

    assert save_calls == []


def test_run_agent_verbose_mode(monkeypatch, concrete_agent, mock_benchmark_interface):
    """Test run_agent in verbose mode."""

    # Mock the run method
    def mock_run(
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        verbose=True,
        **kwargs,
    ):
        return "test_answer"

    monkeypatch.setattr(concrete_agent, "run", mock_run)

    # Mock llm_call for extractor
    mock_response = MockLLMResponse("extracted_answer")

    def mock_llm_call(*args, **kwargs):
        return mock_response

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call)

    # No need to check save_agent_messages call
    concrete_agent.messages = [
        {"role": "user", "content": "Test task"},
        {"role": "assistant", "content": "Test response"},
    ]

    result, messages, usage = _unpack(
        concrete_agent.run_agent(
            interface=mock_benchmark_interface,
            task_id="test_task",
            verbose=True,
        )
    )

    assert result == "extracted_answer"
    assert isinstance(usage, dict)
    assert "prompt_tokens" in usage
    assert "completion_tokens" in usage
    assert "total_tokens" in usage


def test_run_agent_extractor_error(
    monkeypatch, concrete_agent, mock_benchmark_interface
):
    """Test run_agent when extractor fails."""

    # Mock the run method
    def mock_run(
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        enable_surrender=False,
    ):
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

    result, messages, usage = _unpack(
        concrete_agent.run_agent(
            interface=mock_benchmark_interface, task_id="test_task"
        )
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
    concrete_agent.cumulative_token_usage = {
        "prompt_tokens": 50,
        "completion_tokens": 25,
        "total_tokens": 75,
    }

    usage = concrete_agent.get_total_token_usage()

    assert usage == {
        "prompt_tokens": 50,
        "completion_tokens": 25,
        "total_tokens": 75,
    }


def test_get_total_token_usage_with_missing_keys(concrete_agent):
    """Test token usage calculation with missing keys."""
    concrete_agent.cumulative_token_usage = {
        "completion_tokens": 75,
        "total_tokens": 275,
    }  # Missing prompt_tokens

    usage = concrete_agent.get_total_token_usage()

    assert usage == {
        "prompt_tokens": 0,
        "completion_tokens": 75,
        "total_tokens": 275,
    }


def test_get_total_token_usage_accumulates_across_calls(monkeypatch, concrete_agent):
    """get_total_token_usage sums usage over the whole run, not just the last call."""

    def mock_llm_call(*args, **kwargs):
        return MockLLMResponse(
            "response",
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
        )

    monkeypatch.setattr("corral.agents.base_agent.llm_call", mock_llm_call)
    concrete_agent.messages = [{"role": "user", "content": "Test message"}]

    concrete_agent.get_llm_response()
    concrete_agent.get_llm_response()

    # Last call's usage is retained per-call for context-size tracking...
    assert concrete_agent.token_usage["total_tokens"] == 150
    # ...while the run total spans both calls.
    assert concrete_agent.get_total_token_usage() == {
        "prompt_tokens": 200,
        "completion_tokens": 100,
        "total_tokens": 300,
    }


def test_reset_token_usage(concrete_agent):
    """Test resetting token usage."""
    concrete_agent.token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }
    concrete_agent.cumulative_token_usage = {
        "prompt_tokens": 100,
        "completion_tokens": 50,
        "total_tokens": 150,
    }

    concrete_agent.reset_token_usage()

    assert not concrete_agent.token_usage
    assert not concrete_agent.cumulative_token_usage


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
    def mock_run(
        interface,
        task_id,
        task_prompt=None,
        examples=None,
        enable_surrender=False,
    ):
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

    result, messages, usage = _unpack(
        concrete_agent.run_agent(
            interface=mock_benchmark_interface, task_id="test_task"
        )
    )

    # Check that the extractor prompt was called with correct parameters
    assert len(call_args_captured) == 1
    call_args = call_args_captured[0]
    prompt_content = call_args[1]["messages"][0]["content"]
    assert "Answer: test_answer" in prompt_content
    assert "Message: " in prompt_content


# Tests for kwargs handling in run() method


def test_agent_run_accepts_enable_surrender_via_kwargs(
    monkeypatch, mock_benchmark_interface
):
    """Test that agents can accept enable_surrender via kwargs without error."""

    class AgentWithKwargs(BaseAgent):
        """Agent that accepts kwargs like ReflexionAgent."""

        def __init__(self, **kwargs):
            super().__init__(
                user_prompt="Task: {{task_guide}}",
                system_prompt="You are a helpful assistant.",
                extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
                **kwargs,
            )

        def run(
            self,
            interface: CorralRouter,
            task_id: str,
            task_prompt: str | None = None,
            examples: list[str] | None = None,
            **kwargs,
        ) -> str:
            """Run method that captures enable_surrender in kwargs."""
            # Verify enable_surrender is in kwargs
            assert "enable_surrender" in kwargs
            self.received_kwargs = kwargs
            # Add messages for extractor
            self.messages = [
                {"role": "user", "content": "Test task"},
                {"role": "assistant", "content": "answer"},
            ]
            return "answer"

    agent = AgentWithKwargs()

    # Mock llm_call for the extractor
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        lambda *args, **kwargs: MockLLMResponse("extracted"),
    )

    # Call run_agent with enable_surrender=True
    result, messages, usage = _unpack(
        agent.run_agent(
            interface=mock_benchmark_interface, task_id="test", enable_surrender=True
        )
    )

    # Verify kwargs were passed correctly
    assert agent.received_kwargs["enable_surrender"] is True


def test_agent_run_with_explicit_enable_surrender_parameter(
    monkeypatch, mock_benchmark_interface
):
    """Test that agents with explicit enable_surrender parameter work correctly."""

    class AgentWithExplicitParam(BaseAgent):
        """Agent that explicitly declares enable_surrender like ReActAgent."""

        def __init__(self, **kwargs):
            super().__init__(
                user_prompt="Task: {{task_guide}}",
                system_prompt="You are a helpful assistant.",
                extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
                **kwargs,
            )

        def run(
            self,
            interface: CorralRouter,
            task_id: str,
            task_prompt: str | None = None,
            examples: list[str] | None = None,
            enable_surrender: bool = False,
            **kwargs,
        ) -> str:
            """Run method with explicit enable_surrender parameter."""
            self.received_enable_surrender = enable_surrender
            self.messages = [
                {"role": "user", "content": "Test task"},
                {"role": "assistant", "content": "answer"},
            ]
            return "answer"

    agent = AgentWithExplicitParam()

    # Mock llm_call for the extractor
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        lambda *args, **kwargs: MockLLMResponse("extracted"),
    )

    # Test with enable_surrender=True
    result, messages, usage = _unpack(
        agent.run_agent(
            interface=mock_benchmark_interface, task_id="test", enable_surrender=True
        )
    )
    assert agent.received_enable_surrender is True

    # Test with enable_surrender=False (default)
    result, messages, usage = _unpack(
        agent.run_agent(
            interface=mock_benchmark_interface, task_id="test", enable_surrender=False
        )
    )
    assert agent.received_enable_surrender is False


def test_agent_run_without_enable_surrender_uses_default(
    monkeypatch, mock_benchmark_interface
):
    """Test that enable_surrender defaults to False when not provided."""

    class AgentWithExplicitParam(BaseAgent):
        """Agent with explicit enable_surrender parameter."""

        def __init__(self, **kwargs):
            super().__init__(
                user_prompt="Task: {{task_guide}}",
                system_prompt="You are a helpful assistant.",
                extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
                **kwargs,
            )

        def run(
            self,
            interface: CorralRouter,
            task_id: str,
            task_prompt: str | None = None,
            examples: list[str] | None = None,
            enable_surrender: bool = False,
            **kwargs,
        ) -> str:
            """Run method with explicit enable_surrender parameter."""
            self.received_enable_surrender = enable_surrender
            self.messages = [
                {"role": "user", "content": "Test task"},
                {"role": "assistant", "content": "answer"},
            ]
            return "answer"

    agent = AgentWithExplicitParam()

    # Mock llm_call for the extractor
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        lambda *args, **kwargs: MockLLMResponse("extracted"),
    )

    # Call without enable_surrender parameter
    result, messages, usage = _unpack(
        agent.run_agent(interface=mock_benchmark_interface, task_id="test")
    )

    # Should default to False
    assert agent.received_enable_surrender is False


def test_agent_run_kwargs_dont_interfere_with_agents_not_using_them(
    monkeypatch, mock_benchmark_interface
):
    """Test that agents ignoring enable_surrender via kwargs don't error."""

    class AgentIgnoringKwargs(BaseAgent):
        """Agent that doesn't use enable_surrender at all."""

        def __init__(self, **kwargs):
            super().__init__(
                user_prompt="Task: {{task_guide}}",
                system_prompt="You are a helpful assistant.",
                extractor_prompt="Extract the answer from: {{answer}}. Context: {{message}}",
                **kwargs,
            )
            self.run_called = False

        def run(
            self,
            interface: CorralRouter,
            task_id: str,
            task_prompt: str | None = None,
            examples: list[str] | None = None,
            **kwargs,
        ) -> str:
            """Run method that completely ignores kwargs."""
            self.run_called = True
            self.messages = [
                {"role": "user", "content": "Test task"},
                {"role": "assistant", "content": "answer"},
            ]
            return "answer"

    agent = AgentIgnoringKwargs()

    # Mock llm_call for the extractor
    monkeypatch.setattr(
        "corral.agents.base_agent.llm_call",
        lambda *args, **kwargs: MockLLMResponse("extracted"),
    )

    # Should not raise an error even when enable_surrender is passed
    result, messages, usage = _unpack(
        agent.run_agent(
            interface=mock_benchmark_interface, task_id="test", enable_surrender=True
        )
    )

    assert agent.run_called is True
    assert result == "extracted"
