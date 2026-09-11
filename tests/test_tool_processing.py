import json
from typing import Any
from unittest.mock import Mock, patch

import pytest

from corral.core.action import Action
from corral.core.environment import Environment, Toolset
from corral.core.state import ActionState, ExecutionState
from corral.core.task import TaskDefinition
from corral.core.tool import (
    Tool,
    ToolArgument,
    ToolConcurrency,
)
from corral.core.transition import execute_action


class _TestEnv(Environment):
    """Mock Environment for testing purposes"""

    def __init__(self, task_id: str, base_work_dir: str, fs_manager: Any):
        task = TaskDefinition(
            name="mock",
            description="Mock task",
            tools=[],
            scoring_fn=lambda answer: 100,
            submission_format={},
        )
        super().__init__(
            task_id,
            task,
            base_work_dir,
            fs_manager=fs_manager,
            toolset=Toolset(workspace_factory=None),
        )

    def get_task_prompt(self, state) -> str:
        del state
        return "Mock task prompt"


class TestEnvironmentPreprocessing:
    """Test suite for Environment argument preprocessing functionality"""

    def setup_method(self):
        """Set up test environment with mock tools"""

        # Now create the environment instance
        self.env = _TestEnv(
            task_id="test_task", base_work_dir="/tmp/test", fs_manager=None
        )

        # Add test tools with different parameter types
        self._add_test_tools()

    def _add_test_tools(self):
        """Add mock tools with various parameter types for testing"""

        # Tool with dict parameter
        dict_tool = Mock(spec=Tool)
        dict_tool.name = "dict_tool"
        dict_tool.arguments = [
            ToolArgument(
                name="config", type="dict[str, str]", description="Config dict"
            ),
            ToolArgument(name="output_path", type="str", description="Output path"),
        ]
        dict_tool.validate_arguments.return_value = (True, None)
        dict_tool.execute.return_value = "dict_tool_result"

        # Tool with list parameter
        list_tool = Mock(spec=Tool)
        list_tool.name = "list_tool"
        list_tool.arguments = [
            ToolArgument(name="items", type="list[str]", description="List of items"),
            ToolArgument(name="count", type="int", description="Count"),
        ]
        list_tool.validate_arguments.return_value = (True, None)
        list_tool.execute.return_value = "list_tool_result"

        # Tool with string parameter (should not be parsed)
        string_tool = Mock(spec=Tool)
        string_tool.name = "string_tool"
        string_tool.arguments = [
            ToolArgument(name="content", type="str", description="String content"),
            ToolArgument(name="path", type="str", description="File path"),
        ]
        string_tool.validate_arguments.return_value = (True, None)
        string_tool.execute.return_value = "string_tool_result"

        # Tool with mixed parameter types
        mixed_tool = Mock(spec=Tool)
        mixed_tool.name = "mixed_tool"
        mixed_tool.arguments = [
            ToolArgument(name="data", type="dict", description="Data object"),
            ToolArgument(name="tags", type="list", description="Tag list"),
            ToolArgument(name="name", type="str", description="Name string"),
            ToolArgument(name="count", type="int", description="Count number"),
        ]
        mixed_tool.validate_arguments.return_value = (True, None)
        mixed_tool.execute.return_value = "mixed_tool_result"

        # Add tools to environment
        self.env.tools = {
            "dict_tool": dict_tool,
            "list_tool": list_tool,
            "string_tool": string_tool,
            "mixed_tool": mixed_tool,
        }
        for name, tool_object in self.env.tools.items():
            tool_object.hidden_args = {}
            tool_object.concurrency = ToolConcurrency.SERIAL
            tool_object.get_openai_tool_format.return_value = {
                "type": "function",
                "function": {"name": name, "parameters": {}},
            }

    def test_preprocess_dict_parameter(self):
        """Test parsing of JSON string to dictionary"""
        args = {
            "config": '{"key1": "value1", "key2": "value2"}',
            "output_path": "/tmp/output.json",
        }

        result = self.env.preprocess_arguments("dict_tool", args)

        assert result["config"] == {"key1": "value1", "key2": "value2"}
        assert isinstance(result["config"], dict)
        assert result["output_path"] == "/tmp/output.json"
        assert isinstance(result["output_path"], str)

    def test_preprocess_list_parameter(self):
        """Test parsing of JSON string to list"""
        args = {"items": '["item1", "item2", "item3"]', "count": 5}

        result = self.env.preprocess_arguments("list_tool", args)

        assert result["items"] == ["item1", "item2", "item3"]
        assert isinstance(result["items"], list)
        assert result["count"] == 5
        assert isinstance(result["count"], int)

    def test_preserve_string_parameters(self):
        """Test that string parameters are not parsed even if they contain JSON"""
        args = {
            "content": '{"json": "content", "should": "stay", "as": "string"}',
            "path": "/tmp/file.json",
        }

        result = self.env.preprocess_arguments("string_tool", args)

        # Content should remain as string (not parsed)
        assert (
            result["content"] == '{"json": "content", "should": "stay", "as": "string"}'
        )
        assert isinstance(result["content"], str)
        assert result["path"] == "/tmp/file.json"
        assert isinstance(result["path"], str)

    def test_mixed_parameter_types(self):
        """Test tool with multiple different parameter types"""
        args = {
            "data": '{"nested": {"key": "value"}, "array": [1, 2, 3]}',
            "tags": '["tag1", "tag2", "tag3"]',
            "name": "test_name",
            "count": 42,
        }

        result = self.env.preprocess_arguments("mixed_tool", args)

        # Dict should be parsed
        assert result["data"] == {"nested": {"key": "value"}, "array": [1, 2, 3]}
        assert isinstance(result["data"], dict)

        # List should be parsed
        assert result["tags"] == ["tag1", "tag2", "tag3"]
        assert isinstance(result["tags"], list)

        # String should remain string
        assert result["name"] == "test_name"
        assert isinstance(result["name"], str)

        # Number should remain number
        assert result["count"] == 42
        assert isinstance(result["count"], int)

    def test_nested_objects(self):
        """Test parsing of deeply nested JSON structures"""
        args = {
            "config": '{"level1": {"level2": {"level3": {"deep": "value"}}, "array": [{"item": 1}, {"item": 2}]}}',
            "output_path": "/tmp/nested.json",
        }

        result = self.env.preprocess_arguments("dict_tool", args)

        expected = {
            "level1": {
                "level2": {"level3": {"deep": "value"}},
                "array": [{"item": 1}, {"item": 2}],
            }
        }
        assert result["config"] == expected
        assert isinstance(result["config"], dict)

    def test_malformed_json_fallback(self):
        """Test that malformed JSON gracefully falls back to original string"""
        args = {
            "config": '{"malformed": json, "missing": quotes}',  # Invalid JSON
            "output_path": "/tmp/output.json",
        }

        result = self.env.preprocess_arguments("dict_tool", args)

        # Should fallback to original string on JSON parse error
        assert result["config"] == '{"malformed": json, "missing": quotes}'
        assert isinstance(result["config"], str)
        assert result["output_path"] == "/tmp/output.json"

    def test_non_json_string_preservation(self):
        """Test that non-JSON strings are preserved unchanged"""
        args = {"config": "not_json_at_all", "output_path": "/tmp/output.json"}

        result = self.env.preprocess_arguments("dict_tool", args)

        # Non-JSON string should be preserved
        assert result["config"] == "not_json_at_all"
        assert isinstance(result["config"], str)

    def test_already_parsed_objects(self):
        """Test that already-parsed objects are left unchanged"""
        args = {
            "config": {"already": "parsed", "dict": True},  # Already a dict
            "output_path": "/tmp/output.json",
        }

        result = self.env.preprocess_arguments("dict_tool", args)

        # Should remain unchanged
        assert result["config"] == {"already": "parsed", "dict": True}
        assert isinstance(result["config"], dict)

    def test_unknown_tool_fallback(self):
        """Test behavior when tool is not found"""
        args = {
            "param1": '{"should": "not", "be": "parsed"}',
            "param2": "normal_string",
        }

        result = self.env.preprocess_arguments("unknown_tool", args)

        # Should return original args unchanged
        assert result == args

    def test_list_with_mixed_types(self):
        """Test parsing of JSON arrays with mixed content types"""
        args = {"items": '["string", 123, {"object": "value"}, [1, 2, 3], true, null]'}

        result = self.env.preprocess_arguments("list_tool", args)

        expected = ["string", 123, {"object": "value"}, [1, 2, 3], True, None]
        assert result["items"] == expected
        assert isinstance(result["items"], list)

    def test_type_variations(self):
        """Test different type annotation formats"""
        # Test tool with various type formats
        varied_tool = Mock(spec=Tool)
        varied_tool.name = "varied_tool"
        varied_tool.arguments = [
            ToolArgument(name="dict_param", type="dict", description="Generic dict"),
            ToolArgument(name="list_param", type="list", description="Generic list"),
            ToolArgument(name="array_param", type="array", description="Array type"),
            ToolArgument(name="object_param", type="object", description="Object type"),
        ]
        varied_tool.validate_arguments.return_value = (True, None)
        varied_tool.execute.return_value = "varied_tool_result"

        self.env.tools["varied_tool"] = varied_tool

        args = {
            "dict_param": '{"key": "value"}',
            "list_param": '["item1", "item2"]',
            "array_param": "[1, 2, 3]",
            "object_param": '{"nested": {"data": true}}',
        }

        result = self.env.preprocess_arguments("varied_tool", args)

        assert result["dict_param"] == {"key": "value"}
        assert result["list_param"] == ["item1", "item2"]
        assert result["array_param"] == [1, 2, 3]
        assert result["object_param"] == {"nested": {"data": True}}

    def test_integration_with_call_tool(self):
        """Test that preprocessing is applied during actual tool calls"""
        args = {
            "config": '{"integration": "test", "working": true}',
            "output_path": "/tmp/integration.json",
        }

        # Mock the time functions to avoid real timing
        with patch("time.perf_counter", side_effect=[0.0, 0.1]):
            action = Action(name="dict_tool", arguments=args)
            state = ExecutionState(
                through_commit_hash="a" * 64,
                execution_id="execution",
                branch_id="main",
                actions={
                    action.id: ActionState(
                        action=action,
                        requested_by_run_id="agent",
                    )
                },
            )
            result = execute_action(self.env, state, action)

        # Verify tool was called with processed arguments
        dict_tool = self.env.tools["dict_tool"]
        dict_tool.execute.assert_called_once()

        # Get the actual arguments passed to the tool
        call_args = dict_tool.execute.call_args[1]  # kwargs

        assert call_args["config"] == {"integration": "test", "working": True}
        assert isinstance(call_args["config"], dict)
        assert call_args["output_path"] == "/tmp/integration.json"

        assert result.status == "success"
        assert result.observation == "dict_tool_result"

    def test_empty_and_null_values(self):
        """Test handling of empty and null values"""
        args = {
            "output_path": None,
        }

        result = self.env.preprocess_arguments("mixed_tool", args)

        assert result["output_path"] is None

    def test_unicode_and_special_characters(self):
        """Test handling of unicode and special characters in JSON"""
        args = {
            "config": '{"unicode": "こんにちは", "special": "chars: \\n\\t\\r", "emoji": "🎯"}',
            "output_path": "/tmp/unicode.json",
        }

        result = self.env.preprocess_arguments("dict_tool", args)

        expected = {"unicode": "こんにちは", "special": "chars: \n\t\r", "emoji": "🎯"}
        assert result["config"] == expected

    @pytest.mark.parametrize(
        ("json_string", "expected"),
        [
            ('{"simple": "dict"}', {"simple": "dict"}),
            ('["simple", "list"]', ["simple", "list"]),
            (
                '{"nested": {"deep": {"value": 123}}}',
                {"nested": {"deep": {"value": 123}}},
            ),
            (
                '[{"mixed": "array"}, {"of": "objects"}]',
                [{"mixed": "array"}, {"of": "objects"}],
            ),
        ],
    )
    def test_parametrized_json_parsing(self, json_string, expected):
        """Parametrized test for various JSON parsing scenarios"""
        args = {"config": json_string, "output_path": "/tmp/test.json"}
        result = self.env.preprocess_arguments("dict_tool", args)
        assert result["config"] == expected

    def _setup_for_additional_tests(self):
        # Create a mock environment that implements the abstract methods

        self.env = _TestEnv(
            task_id="test_task", base_work_dir="/tmp/test", fs_manager=None
        )

    def test_recursive_nested_structures(self):
        """Test handling of deeply recursive structures"""
        # Set up environment for this test
        self._setup_for_additional_tests()

        # Create a deeply nested structure
        nested_json = '{"level": {"level": {"level": {"level": {"deep": "value"}}}}}'

        mock_tool = Mock(spec=Tool)
        mock_tool.name = "recursive_tool"
        mock_tool.arguments = [
            ToolArgument(name="data", type="dict", description="Nested data")
        ]

        self.env.tools["recursive_tool"] = mock_tool

        args = {"data": nested_json}
        _result = self.env.preprocess_arguments("recursive_tool", args)

    def test_large_json_structures(self):
        """Test handling of large JSON structures"""
        # Set up environment for this test
        self._setup_for_additional_tests()

        # Create a large list
        large_list = json.dumps([f"item_{i}" for i in range(1000)])

        mock_tool = Mock(spec=Tool)
        mock_tool.name = "large_tool"
        mock_tool.arguments = [
            ToolArgument(name="items", type="list[str]", description="Large list")
        ]

        self.env.tools["large_tool"] = mock_tool

        args = {"items": large_list}
        result = self.env.preprocess_arguments("large_tool", args)

        assert len(result["items"]) == 1000

    def test_concurrent_tool_calls(self):
        """Test that preprocessing works correctly with concurrent tool calls"""
        # Set up environment for this test
        self._setup_for_additional_tests()

        # This test would be more meaningful with actual threading,
        # but we can at least verify the method is stateless

        mock_tool = Mock(spec=Tool)
        mock_tool.name = "concurrent_tool"
        mock_tool.arguments = [
            ToolArgument(name="data", type="dict", description="Data")
        ]

        self.env.tools["concurrent_tool"] = mock_tool

        # Simulate multiple calls with different data
        args1 = {"data": '{"call": 1, "data": "first"}'}
        args2 = {"data": '{"call": 2, "data": "second"}'}

        result1 = self.env.preprocess_arguments("concurrent_tool", args1)
        result2 = self.env.preprocess_arguments("concurrent_tool", args2)

        assert result1["data"] == {"call": 1, "data": "first"}
        assert result2["data"] == {"call": 2, "data": "second"}

        # Ensure no cross-contamination
        assert result1["data"] != result2["data"]

        # Ensure no cross-contamination
        assert result1["data"] != result2["data"]


class TestGetAvailableTools:
    """Tests for Environment.get_available_tools()"""

    def setup_method(self):
        self.env = _TestEnv(
            task_id="test_task", base_work_dir="/tmp/test", fs_manager=None
        )

    def _make_tool(self, name: str, openai_format: dict) -> Mock:
        t = Mock(spec=Tool)
        t.name = name
        t.get_openai_tool_format.return_value = openai_format
        return t

    def test_returns_empty_list_when_no_tools(self):
        """get_available_tools returns an empty list when no tools have been added."""
        assert self.env.get_available_tools() == []

    def test_returns_openai_format_for_single_tool(self):
        """get_available_tools delegates to each tool's get_openai_tool_format."""
        expected = {
            "type": "function",
            "function": {
                "name": "my_tool",
                "description": "Does something",
                "parameters": {"type": "object", "properties": {}},
            },
        }
        self.env.tools["my_tool"] = self._make_tool("my_tool", expected)

        result = self.env.get_available_tools()

        assert result == [expected]

    def test_returns_one_entry_per_tool(self):
        """get_available_tools returns one entry for each registered tool."""
        for name in ("tool_a", "tool_b", "tool_c"):
            fmt = {
                "type": "function",
                "function": {"name": name, "description": "", "parameters": {}},
            }
            self.env.tools[name] = self._make_tool(name, fmt)

        result = self.env.get_available_tools()

        assert len(result) == 3
        names = {entry["function"]["name"] for entry in result}
        assert names == {"tool_a", "tool_b", "tool_c"}

    def test_each_tool_get_openai_tool_format_called_once(self):
        """get_available_tools calls get_openai_tool_format exactly once per tool."""
        mock_tool = self._make_tool(
            "t",
            {
                "type": "function",
                "function": {"name": "t", "description": "", "parameters": {}},
            },
        )
        self.env.tools["t"] = mock_tool

        self.env.get_available_tools()

        mock_tool.get_openai_tool_format.assert_called_once()
