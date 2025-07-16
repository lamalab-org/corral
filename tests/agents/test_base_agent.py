"""Tests for the BaseAgent class."""

from typing import Any
from unittest.mock import Mock, patch

import pytest
from litellm.types.utils import Message
from promptstore import PromptStore

from corral.agents.base_agent import BaseAgent
from corral.agents.utils import LiteLLMMessage
from corral.evaluate import BenchmarkInterface


class MockPrompt:
    """Mock prompt class that implements the required fill method."""

    def __init__(self, content: str):
        self.content = content

    def fill(self, replacements: dict[str, Any]) -> str:
        """Fill the prompt with replacements."""
        result = self.content
        for key, value in replacements.items():
            result = result.replace(f"{{{{{key}}}}}", str(value))
        return result


class ConcreteAgent(BaseAgent):
    """Concrete implementation of BaseAgent for testing purposes."""

    def run(
        self,
        interface: BenchmarkInterface,
        task_id: str,
        history: list[LiteLLMMessage] | None = None,
        task_prompt: str | None = None,
        examples: list[str] | None = None,
    ) -> str:
        """Simple implementation for testing."""
        return "test_answer"


@pytest.fixture()
def mock_prompt_store():
    """Mock PromptStore for testing."""
    store = Mock(spec=PromptStore)
    store.get.return_value = MockPrompt("Test prompt: {{task_guide}}")
    return store


@pytest.fixture()
def mock_benchmark_interface():
    """Mock BenchmarkInterface for testing."""
    return Mock(spec=BenchmarkInterface)


@pytest.fixture()
def concrete_agent(mock_prompt_store):
    """Create a concrete agent instance for testing."""
    return ConcreteAgent(
        model="test-model",
        prompt_store=mock_prompt_store,
        temperature=0.5,
        max_iterations=5,
    )


class TestBaseAgentInitialization:
    """Test BaseAgent initialization."""

    def test_default_initialization(self):
        """Test agent initialization with default values."""
        agent = ConcreteAgent()

        assert agent.model == "openai/gpt-4o"
        assert agent.max_iterations == 10
        assert agent.api_endpoint is None
        assert agent.temperature == 0.7
        assert agent.messages == []
        assert agent.token_usage == []

    def test_custom_initialization(self, mock_prompt_store):
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

    def test_initialization_with_custom_prompts(self, mock_prompt_store):
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

    def test_initialization_with_string_prompts(self, mock_prompt_store):
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

    @patch("corral.agents.base_agent.importlib.resources.path")
    def test_default_prompt_store_creation(self, mock_resources_path):
        """Test that default prompt store is created when none provided."""
        mock_resources_path.return_value.__enter__.return_value = "/mock/path"

        with patch("corral.agents.base_agent.PromptStore") as mock_promptstore:
            mock_promptstore.assert_called_once_with("/mock/path/prompts")

    def test_kwargs_passed_through(self, mock_prompt_store):
        """Test that additional kwargs are stored."""
        agent = ConcreteAgent(
            prompt_store=mock_prompt_store, custom_arg="test_value", another_arg=42
        )

        assert agent.kwargs["custom_arg"] == "test_value"
        assert agent.kwargs["another_arg"] == 42


class TestBaseAgentLLMResponse:
    """Test BaseAgent LLM response handling."""

    @patch("corral.agents.base_agent.llm_call")
    def test_get_llm_response_success(self, mock_llm_call, concrete_agent):
        """Test successful LLM response."""
        mock_response = Mock()
        mock_response.content = "Test response"
        mock_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        mock_llm_call.return_value = (mock_response, mock_usage)

        concrete_agent.messages = [{"role": "user", "content": "Test message"}]

        response = concrete_agent.get_llm_response()

        assert response == mock_response
        assert len(concrete_agent.token_usage) == 1
        assert concrete_agent.token_usage[0] == mock_usage

        mock_llm_call.assert_called_once_with(
            model="test-model",
            messages=concrete_agent.messages,
            tools=None,
            temperature=0.5,
            api_endpoint=None,
            return_usage=True,
        )

    @patch("corral.agents.base_agent.llm_call")
    def test_get_llm_response_with_tools(self, mock_llm_call, concrete_agent):
        """Test LLM response with tools."""
        mock_response = Mock()
        mock_usage = {
            "prompt_tokens": 100,
            "completion_tokens": 50,
            "total_tokens": 150,
        }
        mock_llm_call.return_value = (mock_response, mock_usage)

        tools = [{"type": "function", "function": {"name": "test_tool"}}]
        concrete_agent.messages = [{"role": "user", "content": "Test message"}]

        response = concrete_agent.get_llm_response(tools=tools)

        assert response == mock_response
        mock_llm_call.assert_called_once_with(
            model="test-model",
            messages=concrete_agent.messages,
            tools=tools,
            temperature=0.5,
            api_endpoint=None,
            return_usage=True,
        )

    @patch("corral.agents.base_agent.llm_call")
    def test_get_llm_response_rate_limit_error(self, mock_llm_call, concrete_agent):
        """Test handling of rate limit errors."""
        from unittest.mock import Mock

        import openai

        # Create a mock response for the exception
        mock_response = Mock()
        mock_response.status_code = 429
        mock_body = {"error": {"message": "Rate limit exceeded"}}

        mock_llm_call.side_effect = openai.RateLimitError(
            "Rate limit exceeded", response=mock_response, body=mock_body
        )

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

    @patch("corral.agents.base_agent.llm_call")
    def test_get_llm_response_context_window_error(self, mock_llm_call, concrete_agent):
        """Test handling of context window exceeded errors."""
        import litellm

        mock_llm_call.side_effect = litellm.ContextWindowExceededError(
            "Context window exceeded", model="test-model", llm_provider="test-provider"
        )

        concrete_agent.messages = [
            {"role": "user", "content": "Test message"},
            {"role": "assistant", "content": "Response"},
        ]

        response = concrete_agent.get_llm_response()

        assert isinstance(response, Message)
        assert response.role == "user"
        assert "ContextWindowExceededError" in response.content

    @patch("corral.agents.base_agent.llm_call")
    def test_get_llm_response_generic_error(self, mock_llm_call, concrete_agent):
        """Test handling of generic errors."""
        mock_llm_call.side_effect = Exception("Generic error")

        with pytest.raises(Exception, match="Generic error"):
            concrete_agent.get_llm_response()


class TestBaseAgentRunAgent:
    """Test BaseAgent run_agent method."""

    def test_run_agent_success(self, concrete_agent, mock_benchmark_interface):
        """Test successful run_agent execution."""
        with (
            patch.object(concrete_agent, "run", return_value="test_answer") as mock_run,
            patch("corral.agents.base_agent.llm_call") as mock_llm_call,
        ):
            mock_response = Mock()
            mock_response.content = "extracted_answer"
            mock_llm_call.return_value = mock_response

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

            mock_run.assert_called_once_with(
                mock_benchmark_interface,
                "test_task",
                [],
                "Test prompt",
                ["example1", "example2"],
            )

    def test_run_agent_with_system_message(
        self, concrete_agent, mock_benchmark_interface
    ):
        """Test run_agent with system message."""
        # Set up a proper extractor prompt
        concrete_agent.extractor_prompt = MockPrompt(
            "Extract from message: {{message}}"
        )

        with (
            patch.object(concrete_agent, "run", return_value="test_answer"),
            patch("corral.agents.base_agent.llm_call") as mock_llm_call,
        ):
            mock_response = Mock()
            mock_response.content = "extracted_answer"
            mock_llm_call.return_value = mock_response

            concrete_agent.messages = [
                {"role": "system", "content": "System message"},
                {"role": "user", "content": "Test task"},
                {"role": "assistant", "content": "Test response"},
            ]

            result, usage = concrete_agent.run_agent(
                interface=mock_benchmark_interface, task_id="test_task"
            )

            # Check that extractor prompt was called with both system and user content
            call_args = mock_llm_call.call_args
            prompt_content = call_args[1]["messages"][0]["content"]
            assert "System message" in prompt_content
            assert "Test task" in prompt_content

    def test_run_agent_with_error_in_answer(
        self, concrete_agent, mock_benchmark_interface
    ):
        """Test run_agent when answer contains error."""
        with patch.object(
            concrete_agent, "run", return_value="Error: Something went wrong"
        ):
            result, usage = concrete_agent.run_agent(
                interface=mock_benchmark_interface, task_id="test_task"
            )

            assert "Error: Something went wrong" in result
            assert isinstance(usage, dict)

    def test_run_agent_with_exception(self, concrete_agent, mock_benchmark_interface):
        """Test run_agent when run method raises exception."""
        with patch.object(concrete_agent, "run", side_effect=Exception("Run failed")):
            result, usage = concrete_agent.run_agent(
                interface=mock_benchmark_interface, task_id="test_task"
            )

            assert "Error running agent: Run failed" in result
            assert isinstance(usage, dict)

    def test_run_agent_verbose_mode(self, concrete_agent, mock_benchmark_interface):
        """Test run_agent in verbose mode."""
        with (
            patch.object(concrete_agent, "run", return_value="test_answer"),
            patch("corral.agents.base_agent.llm_call") as mock_llm_call,
            patch("corral.agents.base_agent.save_agent_messages") as mock_save,
        ):
            mock_response = Mock()
            mock_response.content = "extracted_answer"
            mock_llm_call.return_value = mock_response

            concrete_agent.messages = [
                {"role": "user", "content": "Test task"},
                {"role": "assistant", "content": "Test response"},
            ]

            result, usage = concrete_agent.run_agent(
                interface=mock_benchmark_interface,
                task_id="test_task",
                verbose=True,
            )

            mock_save.assert_called_once()
            call_args = mock_save.call_args
            assert call_args[0][0] == concrete_agent.messages
            assert call_args[0][1] == "test_task"
            assert call_args[0][2] == "ConcreteAgent"

    def test_run_agent_extractor_error(self, concrete_agent, mock_benchmark_interface):
        """Test run_agent when extractor fails."""
        with (
            patch.object(concrete_agent, "run", return_value="test_answer"),
            patch(
                "corral.agents.base_agent.llm_call",
                side_effect=Exception("Extractor failed"),
            ),
        ):
            concrete_agent.messages = [
                {"role": "user", "content": "Test task"},
                {"role": "assistant", "content": "Test response"},
            ]

            result, usage = concrete_agent.run_agent(
                interface=mock_benchmark_interface, task_id="test_task"
            )

            assert result == "test_answer"  # Should return original answer
            assert isinstance(usage, dict)


class TestBaseAgentTokenUsage:
    """Test BaseAgent token usage tracking."""

    def test_get_total_token_usage_empty(self, concrete_agent):
        """Test token usage calculation with no usage data."""
        usage = concrete_agent.get_total_token_usage()

        assert usage == {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

    def test_get_total_token_usage_with_data(self, concrete_agent):
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

    def test_get_total_token_usage_with_missing_keys(self, concrete_agent):
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

    def test_reset_token_usage(self, concrete_agent):
        """Test resetting token usage."""
        concrete_agent.token_usage = [
            {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150}
        ]

        concrete_agent.reset_token_usage()

        assert concrete_agent.token_usage == []


class TestBaseAgentAbstractMethods:
    """Test BaseAgent abstract methods."""

    def test_cannot_instantiate_base_agent(self):
        """Test that BaseAgent cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseAgent()

    def test_concrete_agent_implements_run(
        self, concrete_agent, mock_benchmark_interface
    ):
        """Test that concrete agent implements run method."""
        result = concrete_agent.run(
            interface=mock_benchmark_interface, task_id="test_task"
        )

        assert result == "test_answer"


class TestBaseAgentPromptHandling:
    """Test BaseAgent prompt handling."""

    def test_prompt_fill_method(self):
        """Test that prompts have fill method."""
        mock_prompt = MockPrompt("Test {{variable}}")
        filled = mock_prompt.fill({"variable": "value"})

        assert filled == "Test value"

    def test_extractor_prompt_filling(self, concrete_agent, mock_benchmark_interface):
        """Test that extractor prompt is filled correctly."""
        concrete_agent.extractor_prompt = MockPrompt(
            "Answer: {{answer}}, Message: {{message}}"
        )

        with (
            patch.object(concrete_agent, "run", return_value="test_answer"),
            patch("corral.agents.base_agent.llm_call") as mock_llm_call,
        ):
            mock_response = Mock()
            mock_response.content = "extracted_answer"
            mock_llm_call.return_value = mock_response

            concrete_agent.messages = [
                {"role": "user", "content": "Test task"},
                {"role": "assistant", "content": "Test response"},
            ]

            result, usage = concrete_agent.run_agent(
                interface=mock_benchmark_interface, task_id="test_task"
            )

            # Check that the extractor prompt was called with correct parameters
            call_args = mock_llm_call.call_args
            prompt_content = call_args[1]["messages"][0]["content"]
            assert "Answer: test_answer" in prompt_content
            assert "Message: " in prompt_content
