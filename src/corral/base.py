import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel


class Role(StrEnum):
    """Defines the role type of a component in the system."""

    AGENT = "agent"
    ENVIRONMENT = "environment"
    TOOL = "tool"


class ToolCallStatus(Enum):
    SUCCESS = "success"
    INVALID_TOOL = "invalid_tool"
    INVALID_ARGS = "invalid_args"
    EXECUTION_ERROR = "execution_error"


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


@dataclass
class ToolArgument:
    """Specification for a tool argument"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None
    choices: list[Any] | None = None  # from transformers


@dataclass
class ToolCall:
    """Record of a tool being called"""

    tool_name: str
    arguments: dict[str, Any]
    result: str | None
    status: ToolCallStatus
    error_message: str | None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMMessage:
    role: Role  # 'agent' or 'environment'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskState:
    task_id: str
    task_prompt: str
    trial_id: str = "0"
    messages: list[LLMMessage] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    is_completed: bool = False
    score: float | None = None
    submitted_answer: str | None = None
    feedback: str | None = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: datetime | None = None

    def get_tool_statistics(self) -> dict[str, Any]:
        """Get statistics about tool usage"""
        return {
            "total_calls": len(self.tool_calls),
            "successful_calls": len(
                [t for t in self.tool_calls if t.status == ToolCallStatus.SUCCESS]
            ),
            "failed_calls": len(
                [t for t in self.tool_calls if t.status != ToolCallStatus.SUCCESS]
            ),
            "tools_used": {t.tool_name for t in self.tool_calls},
            "error_types": {
                status: len([t for t in self.tool_calls if t.status == status])
                for status in ToolCallStatus
                if status != ToolCallStatus.SUCCESS
            },
        }


class Tool:
    """Base class for tools
    Inherit from this class to create new tools.
    Should have an execute method that performs the tool's functionality.
    TODO: might need to take state
    TODO: add descriptions of the arguments of the class, i.e., name, description, arguments
    """

    def __init__(self, name: str, description: str, arguments: list[ToolArgument]):
        self.name = name
        self.description = description
        self.arguments = arguments

    def validate_arguments(
        self, provided_args: dict[str, Any]
    ) -> tuple[bool, str | None]:
        """Validate that all required arguments are provided with correct types"""
        for arg in self.arguments:
            if arg.required and arg.name not in provided_args:
                return False, f"Missing required argument: {arg.name}"

            if arg.name in provided_args:
                value = provided_args[arg.name]

                # Check choices if specified
                if arg.choices is not None and value not in arg.choices:
                    return (
                        False,
                        f"Invalid value for {arg.name}. Must be one of: {arg.choices}",
                    )

                try:
                    # Basic type checking
                    if arg.type == "int":
                        int(value)
                    elif arg.type == "float":
                        float(value)
                    elif arg.type == "bool":
                        isinstance(value, bool)
                except ValueError:
                    return (
                        False,
                        f"Invalid type for argument {arg.name}. Expected {arg.type}",
                    )

        return True, None

    def execute(self, **kwargs) -> str:
        """Execute the tool functionality"""
        raise NotImplementedError

    def get_usage_guide(self) -> str:
        """Generate a usage guide for the tool"""
        args_desc = []
        for arg in self.arguments:
            required = (
                "required" if arg.required else f"optional, default: {arg.default}"
            )
            args_desc.append(
                f"- {arg.name} ({arg.type}, {required}): {arg.description}"
            )

        return f"""Tool: {self.name}
Description: {self.description}
Arguments:
{chr(10).join(args_desc)}
"""


class ModalTool(Tool):
    def __init__(
        self,
        modal_func: Callable,
        name: str,
        description: str,
        arguments: list[ToolArgument],
    ):
        super().__init__(
            name=name,
            description=description,
            arguments=arguments,
        )
        self._modal_func = modal_func

    def execute(self, **kwargs):
        return self._modal_func.remote(**kwargs)


class Environment(ABC):
    """Base class for task environments"""

    def __init__(self, task_id: str):
        self.task_id = task_id
        self.tools: dict[str, Tool] = {}
        self.trial_states: dict[str, TaskState] = {}
        self.trial_counter = -1
        self.reset_state()

    def save_current_state(self) -> TaskState:
        """
        Archive the current state as a snapshot.

        Returns:
            TaskState: A deep copy of the current task state
        """
        # Create a deep copy of the entire TaskState object
        return copy.deepcopy(self.state)

    def reset_state(self) -> str:
        """Reset the environment state with a new trial id and fresh TaskState and return finished trail id."""
        if hasattr(self, "state") and self.state is not None:
            archived_snapshot = self.save_current_state()
            self.trial_states[self.state.trial_id] = archived_snapshot

        self.trial_counter += 1
        new_trial_id = str(self.trial_counter)

        self.state = TaskState(
            task_id=self.task_id,
            trial_id=new_trial_id,
            task_prompt=self.get_task_prompt(),
        )
        return self.state.trial_id

    def get_unique_trail_identifier(self) -> str:
        """
        Generate a combined identifier using task_id, trial_id, and a timestamp.

        Returns:
            A string combining task_id, trial_id, and timestamp in format: "{task_id}_{trial_id}_{timestamp}"
        """
        if not hasattr(self, "state") or self.state is None:
            return f"{self.task_id}_no_trial_{datetime.now(tz=timezone.utc).strftime('%m%d%H%M')}"

        timestamp = datetime.now(tz=timezone.utc).strftime("%m%d%H%M")
        return f"{self.task_id}_{self.state.trial_id}_{timestamp}"

    @abstractmethod
    def get_task_prompt(self) -> str | list[dict]:
        """Return the task prompt for the agent"""

    @abstractmethod
    def score(self) -> float:
        """Evaluate the agent's solution and return a score"""

    def add_tool(self, tool: Tool):
        """Add a tool to the environment"""
        self.tools[tool.name] = tool

    def get_available_tools(self) -> list[dict[str, str | list[ToolArgument]]]:
        """Get list of available tools with their descriptions and arguments.

        Returns:
            list[dict[str, str | list[ToolArgument]]]: A list of dictionaries where each dictionary contains:
                - 'name': the tool's name as a string.
                - 'description': a string describing the tool.
                - 'arguments': a list of ToolArgument objects representing the tool's arguments.
        """
        return [
            {
                "name": t.name,
                "description": t.description,
                "arguments": ", ".join(arg.name for arg in t.arguments),
            }
            for t in self.tools.values()
        ]

    def get_tools_guide(self) -> str:
        """Generate a guide for the available tools"""
        tools_guide = "\n\n".join(
            tool.get_usage_guide() for tool in self.tools.values()
        )
        # TODO: make it configurable
        return (
            "Available Tools:\n"
            f"{tools_guide}\n\n"
            "How to use tools:\n"
            "1. Each tool call must specify the tool name and required arguments\n"
            "2. Tools may return errors if arguments are invalid\n"
            "3. You can make multiple tool calls as needed. The tools will be executed sequentially in the order they are called.\n"
            "4. All tool calls are recorded and affect your final score\n"
            "Example tool call format:\n"
            "{{\n"
            '    "tool_name": "tool_name",\n'
            '    "arguments": {{\n'
            '        "arg1": value1,\n'
            '        "arg2": value2\n'
            "    }}\n"
            "}}\n"
        )

    def get_environment_guide(self) -> str:
        """Generate a complete guide for the environment and its tools"""
        tools_guide = self.get_tools_guide()

        # TODO: make it configurable
        return f"""Task: {self.get_task_prompt()}

{tools_guide}
"""

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolCall:
        """Execute a tool and record the call with enhanced error handling"""
        # Check if tool exists
        if tool_name not in self.tools:
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                status=ToolCallStatus.INVALID_TOOL,
                error_message=f"Tool {tool_name} not found",
            )
            self.state.tool_calls.append(tool_call)
            return tool_call

        tool = self.tools[tool_name]

        # Validate arguments
        is_valid, error_message = tool.validate_arguments(arguments)
        if not is_valid:
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                status=ToolCallStatus.INVALID_ARGS,
                error_message=error_message,
            )
            self.state.tool_calls.append(tool_call)
            return tool_call

        # Execute tool
        try:
            result = tool.execute(**arguments)
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                status=ToolCallStatus.SUCCESS,
                error_message=None,
            )
        except Exception as e:
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                status=ToolCallStatus.EXECUTION_ERROR,
                error_message=str(e),
            )

        self.state.tool_calls.append(tool_call)
        return tool_call

    def submit_answer(self, answer: str) -> float:
        """Submit final answer and get score"""
        self.state.submitted_answer = answer
        score = self.score()  # Using existing abstract score method
        self.state.score = score
        self.state.is_completed = True
        self.state.end_time = datetime.now(tz=timezone.utc)
        return score
