"""Tests for the prompt_utils module."""

import pytest
from promptstore import Prompt

from corral.agents.prompt_utils import (
    ValidatedPrompt,
    build_user_content,
    create_prompt,
    get_prompt,
)
from corral.agents.utils import LiteLLMMessage


class TestValidatedPrompt:
    """Test cases for the ValidatedPrompt class."""

    def test_init(self):
        """Test ValidatedPrompt initialization."""
        content = "Hello {{name}}!"
        base_prompt = Prompt(content=content, version=1)
        prompt = ValidatedPrompt(base_prompt)
        assert prompt._prompt == base_prompt

    def test_fill_single_replacement(self):
        """Test filling a prompt with a single replacement."""
        base_prompt = Prompt(content="Hello {{name}}!", version=1)
        prompt = ValidatedPrompt(base_prompt)
        result = prompt.fill({"name": "Alice"})
        assert result == "Hello Alice!"

    def test_fill_multiple_replacements(self):
        """Test filling a prompt with multiple replacements."""
        base_prompt = Prompt(
            content="Hello {{name}}, you are {{age}} years old!", version=1
        )
        prompt = ValidatedPrompt(base_prompt)
        result = prompt.fill({"name": "Bob", "age": 30})
        assert result == "Hello Bob, you are 30 years old!"

    def test_fill_empty_replacements(self):
        """Test filling a prompt with empty replacements."""
        base_prompt = Prompt(content="Hello world!", version=1)
        prompt = ValidatedPrompt(base_prompt)
        result = prompt.fill({})
        assert result == "Hello world!"

    def test_fill_with_non_string_values(self):
        """Test filling a prompt with non-string values (should be converted to string)."""
        base_prompt = Prompt(content="Count: {{count}}, Price: {{price}}", version=1)
        prompt = ValidatedPrompt(base_prompt)
        result = prompt.fill({"count": 5, "price": 19.99})
        assert result == "Count: 5, Price: 19.99"

    def test_fill_extra_replacements(self):
        """Test filling a prompt with extra replacements that don't match placeholders."""
        base_prompt = Prompt(content="Hello {{name}}!", version=1)
        prompt = ValidatedPrompt(base_prompt)
        with pytest.raises(
            KeyError, match="Extra keys provided that don't match any placeholders"
        ):
            prompt.fill({"name": "David", "extra": "ignored"})

    def test_fill_extra_replacements_with_underscore_prefix(self):
        """Test that framework keys with underscore prefix are allowed as extra replacements."""
        base_prompt = Prompt(content="Hello {{name}}!", version=1)
        prompt = ValidatedPrompt(base_prompt)
        result = prompt.fill({"name": "David", "_internal_key": "framework_value"})
        assert result == "Hello David!"

    def test_fill_legitimate_extra_field(self):
        """Test filling a prompt where 'extra' is a legitimate placeholder."""
        base_prompt = Prompt(
            content="Hello {{name}}, here's {{extra}} info!", version=1
        )
        prompt = ValidatedPrompt(base_prompt)
        result = prompt.fill({"name": "Alice", "extra": "bonus"})
        assert result == "Hello Alice, here's bonus info!"

    def test_fill_typo_in_extra_field_caught(self):
        """Test that typos in field names are now caught."""
        base_prompt = Prompt(
            content="Hello {{name}}, here's {{extra}} info!", version=1
        )
        prompt = ValidatedPrompt(base_prompt)
        with pytest.raises(
            KeyError, match="Extra keys provided that don't match any placeholders"
        ):
            prompt.fill(
                {"name": "Alice", "extra": "bonus", "exrta": "typo"}
            )  # typo in 'extra'


class TestGetPrompt:
    """Test cases for the get_prompt function."""

    @pytest.fixture()
    def mock_prompt(self, mocker):
        """Create a mock prompt."""
        mock = mocker.Mock()
        mock.fill = mocker.Mock(return_value="filled content")
        return mock

    @pytest.fixture()
    def mock_store(self, mocker, mock_prompt):
        """Create a mock store."""
        mock = mocker.Mock()
        mock.get = mocker.Mock(return_value=mock_prompt)
        return mock

    def test_get_prompt_with_none_input_and_default_uuid(self, mock_store, mock_prompt):
        """Test getting prompt with None input and valid default UUID."""
        result = get_prompt(mock_store, None, "test-uuid")
        assert result == mock_prompt
        mock_store.get.assert_called_once_with("test-uuid")

    def test_get_prompt_with_none_input_and_none_default_uuid(self):
        """Test getting prompt with None input and None default UUID raises ValueError."""
        with pytest.raises(
            ValueError, match="default_uuid cannot be None when prompt_input is None"
        ):
            get_prompt(self.mock_store, None, None)

    def test_get_prompt_with_string_input(self, mock_store):
        """Test getting prompt with string input creates ValidatedPrompt."""
        result = get_prompt(mock_store, "test content", "unused-uuid")
        assert isinstance(result, ValidatedPrompt)
        assert result._prompt.content == "test content"
        mock_store.get.assert_not_called()

    def test_get_prompt_with_existing_prompt_object(self, mock_store, mocker):
        """Test getting prompt with existing prompt object returns it unchanged."""
        existing_prompt = mocker.Mock()
        result = get_prompt(mock_store, existing_prompt, "unused-uuid")
        assert result is existing_prompt
        mock_store.get.assert_not_called()


class TestCreatePrompt:
    """Test cases for the create_prompt function."""

    @pytest.fixture()
    def mock_system_prompt(self, mocker):
        """Create a mock system prompt."""
        return mocker.Mock()

    @pytest.fixture()
    def mock_user_prompt(self, mocker):
        """Create a mock user prompt."""
        mock = mocker.Mock()
        mock.fill = mocker.Mock(return_value="filled user content")
        return mock

    def test_create_prompt_basic(self, mock_system_prompt, mock_user_prompt):
        """Test creating a basic prompt with system and user prompts."""
        result = create_prompt(mock_system_prompt, mock_user_prompt, "test task guide")

        assert len(result) == 2
        assert result[0].get("role") == "system"
        assert result[0].get("content") == mock_system_prompt
        assert result[1].get("role") == "user"
        assert result[1].get("content") == "filled user content"

    def test_create_prompt_with_history(self, mock_system_prompt, mock_user_prompt):
        """Test creating a prompt with message history."""
        history = [
            LiteLLMMessage(role="user", content="previous message"),
            LiteLLMMessage(role="assistant", content="previous response"),
        ]

        result = create_prompt(
            mock_system_prompt,
            mock_user_prompt,
            "test task guide",
            history=history,
        )

        assert len(result) == 4
        assert result[0].get("role") == "user"
        assert result[0].get("content") == "previous message"
        assert result[1].get("role") == "assistant"
        assert result[1].get("content") == "previous response"
        assert result[2].get("role") == "system"
        assert result[2].get("content") == mock_system_prompt
        assert result[3].get("role") == "user"
        assert result[3].get("content") == "filled user content"

    def test_create_prompt_with_none_system_prompt(self, mock_user_prompt):
        """Test creating a prompt with None system prompt."""
        result = create_prompt(None, mock_user_prompt, "test task guide")

        assert len(result) == 1
        assert result[0].get("role") == "user"
        assert result[0].get("content") == "filled user content"

    def test_create_prompt_with_kwargs(self, mock_system_prompt, mock_user_prompt):
        """Test creating a prompt with additional keyword arguments."""
        create_prompt(
            mock_system_prompt,
            mock_user_prompt,
            "test task guide",
            extra_param="test value",
        )

        # Verify that the user prompt's fill method was called with the extra parameter
        expected_call_args = {
            "task_guide": "test task guide",
            "extra_param": "test value",
            "surrender_instructions": "",
        }
        mock_user_prompt.fill.assert_called_once_with(expected_call_args)

    def test_create_prompt_with_empty_history(
        self, mock_system_prompt, mock_user_prompt
    ):
        """Test creating a prompt with empty history list."""
        result = create_prompt(
            mock_system_prompt,
            mock_user_prompt,
            "test task guide",
            history=[],
        )

        assert len(result) == 2
        assert result[0].get("role") == "system"
        assert result[1].get("role") == "user"


class TestBuildUserContent:
    """Test cases for the build_user_content function."""

    @pytest.fixture()
    def mock_user_prompt(self, mocker):
        """Create a mock user prompt."""
        mock = mocker.Mock()
        mock.fill = mocker.Mock(return_value="filled content")
        return mock

    def test_build_user_content_with_string_task_guide(self, mock_user_prompt):
        """Test building user content with string task guide."""
        result = build_user_content(
            mock_user_prompt, "test task guide", param1="value1", param2="value2"
        )

        assert result == "filled content"
        expected_call_args = {
            "task_guide": "test task guide",
            "param1": "value1",
            "param2": "value2",
            "surrender_instructions": "",
        }
        mock_user_prompt.fill.assert_called_once_with(expected_call_args)

    def test_build_user_content_with_list_task_guide(self, mock_user_prompt):
        """Test building user content with list task guide."""
        task_guide = [
            {"type": "image", "image_url": "test.jpg"},
            {"type": "text", "text": "analyze this"},
        ]

        result = build_user_content(mock_user_prompt, task_guide, param1="value1")

        assert isinstance(result, list)
        assert len(result) == 3  # text + 2 task guide items
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "filled content"
        assert result[1] == {"type": "image", "image_url": "test.jpg"}
        assert result[2] == {"type": "text", "text": "analyze this"}

        # Verify the prompt was filled with the correct parameters
        expected_call_args = {
            "task_guide": "The task is to correctly answer the question with an image specified below.",
            "param1": "value1",
            "surrender_instructions": "",
        }
        mock_user_prompt.fill.assert_called_once_with(expected_call_args)

    def test_build_user_content_with_invalid_task_guide_type(self, mock_user_prompt):
        """Test building user content with invalid task guide type."""
        with pytest.raises(ValueError, match="task_guide should be str or list"):
            build_user_content(
                mock_user_prompt,
                123,  # type: ignore # Invalid type for testing
                param1="value1",
            )

    def test_build_user_content_string_task_guide_fill_error(self, mock_user_prompt):
        """Test building user content with string task guide when fill raises exception."""
        mock_user_prompt.fill.side_effect = Exception("Missing placeholder")

        with pytest.raises(
            KeyError, match="Prompt template contains undefined placeholders"
        ):
            build_user_content(mock_user_prompt, "test task guide", param1="value1")

    def test_build_user_content_list_task_guide_fill_error(self, mock_user_prompt):
        """Test building user content with list task guide when fill raises exception."""
        mock_user_prompt.fill.side_effect = Exception("Missing placeholder")
        task_guide = [{"type": "text", "text": "test"}]

        with pytest.raises(
            KeyError, match="Prompt template contains undefined placeholders"
        ):
            build_user_content(mock_user_prompt, task_guide, param1="value1")

    def test_build_user_content_no_additional_kwargs(self, mock_user_prompt):
        """Test building user content with only task guide, no additional kwargs."""
        result = build_user_content(mock_user_prompt, "test task guide")

        assert result == "filled content"
        expected_call_args = {
            "task_guide": "test task guide",
            "surrender_instructions": "",
        }
        mock_user_prompt.fill.assert_called_once_with(expected_call_args)

    def test_build_user_content_empty_list_task_guide(self, mock_user_prompt):
        """Test building user content with empty list task guide."""
        result = build_user_content(mock_user_prompt, [], param1="value1")

        assert isinstance(result, list)
        assert len(result) == 1  # Only text content
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "filled content"
