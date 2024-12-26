from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


from pydantic import BaseModel, Field


class ToolCallStatus(Enum):
    SUCCESS = "success"
    INVALID_TOOL = "invalid_tool"
    INVALID_ARGS = "invalid_args"
    EXECUTION_ERROR = "execution_error"


class ToolRequest(BaseModel):
    tool_name: str
    arguments: Dict[str, Any]


@dataclass
class ToolArgument:
    """Specification for a tool argument"""

    name: str
    type: str
    description: str
    required: bool = True
    default: Any = None


@dataclass
class ToolCall:
    """Enhanced record of a tool being called"""

    tool_name: str
    arguments: Dict[str, Any]
    result: Optional[str]
    status: ToolCallStatus
    error_message: Optional[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class LLMMessage:
    role: str  # 'agent' or 'environment'
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TaskState:
    task_id: str
    task_prompt: str
    messages: List[LLMMessage] = field(default_factory=list)
    tool_calls: List[ToolCall] = field(default_factory=list)
    is_completed: bool = False
    score: Optional[float] = None
    start_time: datetime = field(default_factory=datetime.now)
    end_time: Optional[datetime] = None

    def get_tool_statistics(self) -> Dict[str, Any]:
        """Get statistics about tool usage"""
        stats = {
            "total_calls": len(self.tool_calls),
            "successful_calls": len(
                [t for t in self.tool_calls if t.status == ToolCallStatus.SUCCESS]
            ),
            "failed_calls": len(
                [t for t in self.tool_calls if t.status != ToolCallStatus.SUCCESS]
            ),
            "tools_used": set(t.tool_name for t in self.tool_calls),
            "error_types": {
                status: len([t for t in self.tool_calls if t.status == status])
                for status in ToolCallStatus
                if status != ToolCallStatus.SUCCESS
            },
        }
        return stats


class Tool:
    """Enhanced base class for all tools"""

    def __init__(self, name: str, description: str, arguments: List[ToolArgument]):
        self.name = name
        self.description = description
        self.arguments = arguments

    def validate_arguments(
        self, provided_args: Dict[str, Any]
    ) -> tuple[bool, Optional[str]]:
        """Validate that all required arguments are provided with correct types"""
        for arg in self.arguments:
            if arg.required and arg.name not in provided_args:
                return False, f"Missing required argument: {arg.name}"

            if arg.name in provided_args:
                try:
                    # Basic type checking - could be enhanced
                    if arg.type == "int":
                        int(provided_args[arg.name])
                    elif arg.type == "float":
                        float(provided_args[arg.name])
                    elif arg.type == "bool":
                        isinstance(provided_args[arg.name], bool)
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


class Environment(ABC):
    """Enhanced base class for task environments"""

    def __init__(self, task_id: str):
        self.tools: Dict[str, Tool] = {}
        self.state = TaskState(task_id=task_id, task_prompt=self.get_task_prompt())

    @abstractmethod
    def get_task_prompt(self) -> str:
        """Return the task prompt for the agent"""
        pass

    @abstractmethod
    def score(self) -> float:
        """Evaluate the agent's solution and return a score"""
        pass

    def add_tool(self, tool: Tool):
        """Add a tool to the environment"""
        self.tools[tool.name] = tool

    def get_available_tools(self) -> List[Dict[str, str]]:
        """Get list of available tools and their descriptions"""
        return [
            {"name": t.name, "description": t.description} for t in self.tools.values()
        ]

    def get_environment_guide(self) -> str:
        """Generate a complete guide for the environment and its tools"""
        tools_guide = "\n\n".join(
            tool.get_usage_guide() for tool in self.tools.values()
        )
        # TODO: make it configurable
        return f"""Task: {self.get_task_prompt()}

Available Tools:
{tools_guide}

How to use tools:
1. Each tool call must specify the tool name and required arguments
2. Tools may return errors if arguments are invalid
3. You can make multiple tool calls as needed
4. All tool calls are recorded and affect your final score

Example tool call format:
{{
    "tool_name": "tool_name",
    "arguments": {{
        "arg1": value1,
        "arg2": value2
    }}
}}
"""

    def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolCall:
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
