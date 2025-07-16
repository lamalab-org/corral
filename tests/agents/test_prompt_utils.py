"""Tests for the prompt_utils module."""

from unittest.mock import Mock

import pytest

from corral.agents.prompt_utils import (
    StringPrompt,
    build_user_content,
    create_prompt,
    get_prompt,
)
from corral.agents.utils import LiteLLMMessage


class TestStringPrompt:
    """Test cases for the StringPrompt class."""

    def test_init(self):
        """Test StringPrompt initialization."""
        content = "Hello {name}!"
        prompt = StringPrompt(content)
        assert prompt.content == content

    def test_fill_single_replacement(self):
        """Test filling a prompt with a single replacement."""
        prompt = StringPrompt("Hello {name}!")
        result = prompt.fill({"name": "Alice"})
        assert result == "Hello Alice!"

    def test_fill_multiple_replacements(self):
        """Test filling a prompt with multiple replacements."""
        prompt = StringPrompt("Hello {name}, you are {age} years old!")
        result = prompt.fill({"name": "Bob", "age": 30})
        assert result == "Hello Bob, you are 30 years old!"

    def test_fill_empty_replacements(self):
        """Test filling a prompt with empty replacements."""
        prompt = StringPrompt("Hello world!")
        result = prompt.fill({})
        assert result == "Hello world!"

    def test_fill_with_non_string_values(self):
        """Test filling a prompt with non-string values (should be converted to string)."""
        prompt = StringPrompt("Count: {count}, Price: {price}")
        result = prompt.fill({"count": 5, "price": 19.99})
        assert result == "Count: 5, Price: 19.99"

    def test_fill_missing_placeholder(self):
        """Test filling a prompt where not all placeholders are provided."""
        prompt = StringPrompt("Hello {name}, you are {age} years old!")
        result = prompt.fill({"name": "Charlie"})
        assert result == "Hello Charlie, you are {age} years old!"

    def test_fill_extra_replacements(self):
        """Test filling a prompt with extra replacements that don't match placeholders."""
        prompt = StringPrompt("Hello {name}!")
        result = prompt.fill({"name": "David", "extra": "ignored"})
        assert result == "Hello David!"


class TestGetPrompt:
    """Test cases for the get_prompt function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_store = Mock()
        self.mock_prompt = Mock()
        self.mock_prompt.fill = Mock(return_value="filled content")
        self.mock_store.get = Mock(return_value=self.mock_prompt)

    def test_get_prompt_with_none_input_and_default_uuid(self):
        """Test getting prompt with None input and valid default UUID."""
        result = get_prompt(self.mock_store, None, "test-uuid")
        assert result == self.mock_prompt
        self.mock_store.get.assert_called_once_with("test-uuid")

    def test_get_prompt_with_none_input_and_none_default_uuid(self):
        """Test getting prompt with None input and None default UUID raises ValueError."""
        with pytest.raises(
            ValueError, match="default_uuid cannot be None when prompt_input is None"
        ):
            get_prompt(self.mock_store, None, None)

    def test_get_prompt_with_string_input(self):
        """Test getting prompt with string input creates StringPrompt."""
        result = get_prompt(self.mock_store, "test content", "unused-uuid")
        assert isinstance(result, StringPrompt)
        assert result.content == "test content"
        self.mock_store.get.assert_not_called()

    def test_get_prompt_with_existing_prompt_object(self):
        """Test getting prompt with existing prompt object returns it unchanged."""
        existing_prompt = Mock()
        result = get_prompt(self.mock_store, existing_prompt, "unused-uuid")
        assert result is existing_prompt
        self.mock_store.get.assert_not_called()


class TestCreatePrompt:
    """Test cases for the create_prompt function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_system_prompt = Mock()
        self.mock_user_prompt = Mock()
        self.mock_user_prompt.fill = Mock(return_value="filled user content")

    def test_create_prompt_basic(self):
        """Test creating a basic prompt with system and user prompts."""
        result = create_prompt(
            self.mock_system_prompt, self.mock_user_prompt, "test task guide"
        )

        assert len(result) == 2
        assert result[0].get("role") == "system"
        assert result[0].get("content") == self.mock_system_prompt
        assert result[1].get("role") == "user"
        assert result[1].get("content") == "filled user content"

    def test_create_prompt_with_history(self):
        """Test creating a prompt with message history."""
        history = [
            LiteLLMMessage(role="user", content="previous message"),
            LiteLLMMessage(role="assistant", content="previous response"),
        ]

        result = create_prompt(
            self.mock_system_prompt,
            self.mock_user_prompt,
            "test task guide",
            history=history,
        )

        assert len(result) == 4
        assert result[0].get("role") == "user"
        assert result[0].get("content") == "previous message"
        assert result[1].get("role") == "assistant"
        assert result[1].get("content") == "previous response"
        assert result[2].get("role") == "system"
        assert result[2].get("content") == self.mock_system_prompt
        assert result[3].get("role") == "user"
        assert result[3].get("content") == "filled user content"

    def test_create_prompt_with_none_system_prompt(self):
        """Test creating a prompt with None system prompt."""
        result = create_prompt(None, self.mock_user_prompt, "test task guide")

        assert len(result) == 1
        assert result[0].get("role") == "user"
        assert result[0].get("content") == "filled user content"

    def test_create_prompt_with_kwargs(self):
        """Test creating a prompt with additional keyword arguments."""
        # Verify that the user prompt's fill method was called with the extra parameter
        expected_call_args = {
            "task_guide": "test task guide",
            "extra_param": "test value",
        }
        self.mock_user_prompt.fill.assert_called_once_with(expected_call_args)

    def test_create_prompt_with_empty_history(self):
        """Test creating a prompt with empty history list."""
        result = create_prompt(
            self.mock_system_prompt,
            self.mock_user_prompt,
            "test task guide",
            history=[],
        )

        assert len(result) == 2
        assert result[0].get("role") == "system"
        assert result[1].get("role") == "user"


class TestBuildUserContent:
    """Test cases for the build_user_content function."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_user_prompt = Mock()
        self.mock_user_prompt.fill = Mock(return_value="filled content")

    def test_build_user_content_with_string_task_guide(self):
        """Test building user content with string task guide."""
        result = build_user_content(
            self.mock_user_prompt, "test task guide", param1="value1", param2="value2"
        )

        assert result == "filled content"
        expected_call_args = {
            "task_guide": "test task guide",
            "param1": "value1",
            "param2": "value2",
        }
        self.mock_user_prompt.fill.assert_called_once_with(expected_call_args)

    def test_build_user_content_with_list_task_guide(self):
        """Test building user content with list task guide."""
        task_guide = [
            {"type": "image", "image_url": "test.jpg"},
            {"type": "text", "text": "analyze this"},
        ]

        result = build_user_content(self.mock_user_prompt, task_guide, param1="value1")

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
        }
        self.mock_user_prompt.fill.assert_called_once_with(expected_call_args)

    def test_build_user_content_with_invalid_task_guide_type(self):
        """Test building user content with invalid task guide type."""
        with pytest.raises(ValueError, match="task_guide should be str or list"):
            build_user_content(
                self.mock_user_prompt,
                123,  # type: ignore # Invalid type for testing
                param1="value1",
            )

    def test_build_user_content_string_task_guide_fill_error(self):
        """Test building user content with string task guide when fill raises exception."""
        self.mock_user_prompt.fill.side_effect = Exception("Missing placeholder")

        with pytest.raises(
            KeyError, match="Prompt template contains undefined placeholders"
        ):
            build_user_content(
                self.mock_user_prompt, "test task guide", param1="value1"
            )

    def test_build_user_content_list_task_guide_fill_error(self):
        """Test building user content with list task guide when fill raises exception."""
        self.mock_user_prompt.fill.side_effect = Exception("Missing placeholder")
        task_guide = [{"type": "text", "text": "test"}]

        with pytest.raises(
            KeyError, match="Prompt template contains undefined placeholders"
        ):
            build_user_content(self.mock_user_prompt, task_guide, param1="value1")

    def test_build_user_content_no_additional_kwargs(self):
        """Test building user content with only task guide, no additional kwargs."""
        result = build_user_content(self.mock_user_prompt, "test task guide")

        assert result == "filled content"
        expected_call_args = {"task_guide": "test task guide"}
        self.mock_user_prompt.fill.assert_called_once_with(expected_call_args)

    def test_build_user_content_empty_list_task_guide(self):
        """Test building user content with empty list task guide."""
        result = build_user_content(self.mock_user_prompt, [], param1="value1")

        assert isinstance(result, list)
        assert len(result) == 1  # Only text content
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "filled content"
