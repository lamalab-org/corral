import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from loguru import logger

from corral.backend.tool import Tool, ToolCallStatus


class Role(StrEnum):
    """Defines the role type of a component in the system."""

    AGENT = "agent"
    ENVIRONMENT = "environment"
    TOOL = "tool"


@dataclass
class ToolCall:
    """Record of a tool being called"""

    tool_name: str
    arguments: dict[str, Any]
    result: str | None
    status: ToolCallStatus
    error_message: str | None
    duration: float | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class LLMMessage:
    role: Role  # 'agent' or 'environment'
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class TaskState:
    task_id: str
    task_prompt: str | list[dict]
    trial_id: str = "0"
    messages: list[LLMMessage] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    is_completed: bool = False
    score: float | None = None
    submitted_answer: str | None = None
    feedback: str | None = None
    start_time: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))
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

    def get_duration(self) -> float | None:
        """Get trial duration in seconds"""
        if self.end_time and self.start_time:
            return (self.end_time - self.start_time).total_seconds()
        return None


class Environment(ABC):
    """Base class for task environments"""

    def __init__(self, task_id: str, base_work_dir: str, fs_manager=None):
        self.task_id = task_id
        self.base_work_dir = base_work_dir
        self.tools: dict[str, Tool] = {}
        self.trial_states: dict[str, TaskState] = {}
        self.trial_counter = -1
        self.fs_manager = fs_manager
        self.reset_state()

    def save_current_state(self) -> TaskState:
        """
        Archive the current state as a snapshot.

        Returns:
            TaskState: A deep copy of the current task state
        """
        # Create a deep copy of the entire TaskState object
        return deepcopy(self.state)

    def reset_state(self) -> str:
        """Reset the environment state with a new trial id and fresh TaskState and return finished trail id."""
        if hasattr(self, "state") and self.state is not None:
            if self.state.is_completed and self.state.end_time is None:
                self.state.end_time = datetime.now(tz=timezone.utc)
            archived_snapshot = self.save_current_state()
            self.trial_states[self.state.trial_id] = archived_snapshot

        self.trial_counter += 1
        new_trial_id = str(self.trial_counter)

        if self.base_work_dir:
            self.current_work_dir = self._create_trial_workspace(new_trial_id)
        else:
            self.current_work_dir = None

        self.state = TaskState(
            task_id=self.task_id,
            trial_id=new_trial_id,
            task_prompt=self.get_task_prompt(),
        )
        return self.state.trial_id

    def _create_trial_workspace(self, trial_id: str) -> str:
        """Create workspace directory for this trial"""
        workspace = Path(self.base_work_dir) / f"{self.task_id}_trial_{trial_id}"
        logger.info(f"Creating workspace: {workspace}")
        if self.fs_manager:
            self.fs_manager.mkdir(str(workspace), create_parents=True)
        else:
            workspace.mkdir(parents=True, exist_ok=True)
        return str(workspace)

    def get_current_work_dir(self) -> str:
        """Get the current working directory for this trial"""
        return self.current_work_dir or self.base_work_dir or ""

    @abstractmethod
    def get_task_prompt(self) -> str | list[dict]:
        """Return the task prompt for the agent"""

    @abstractmethod
    def score(self) -> float:
        """Evaluate the agent's solution and return a score"""

    def add_tool(self, tool: Tool):
        """Add a tool to the environment"""
        self.tools[tool.name] = tool

    def get_available_tools(self) -> list[dict[str, str | list[dict]]]:
        return [
            {
                "name": t.name,
                "description": t.description,
                "arguments": [
                    {
                        "name": arg.name,
                        "type": arg.type,
                        "description": arg.description,
                        "required": arg.required,
                        "default": arg.default,
                        "choices": arg.choices,
                    }
                    for arg in t.arguments
                ],
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
        start_time = time.perf_counter()
        # Check if tool exists
        if tool_name not in self.tools:
            duration = time.perf_counter() - start_time
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                status=ToolCallStatus.INVALID_TOOL,
                error_message=f"Tool {tool_name} not found",
                duration=duration,
            )
            self.state.tool_calls.append(tool_call)
            return tool_call

        tool = self.tools[tool_name]

        # Validate arguments
        is_valid, error_message = tool.validate_arguments(arguments)
        if not is_valid:
            duration = time.perf_counter() - start_time
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                status=ToolCallStatus.INVALID_ARGS,
                error_message=error_message,
                duration=duration,
            )
            self.state.tool_calls.append(tool_call)
            return tool_call

        # Execute tool
        try:
            result = tool.execute(**arguments)
            duration = time.perf_counter() - start_time
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=result,
                status=ToolCallStatus.SUCCESS,
                error_message=None,
                duration=duration,
            )
        except Exception as e:
            duration = time.perf_counter() - start_time
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=arguments,
                result=None,
                status=ToolCallStatus.EXECUTION_ERROR,
                error_message=str(e),
                duration=duration,
            )

        self.state.tool_calls.append(tool_call)
        return tool_call

    def submit_answer(self, answer: str) -> float:
        """Submit final answer and get score"""
        self.state.submitted_answer = answer
        score = self.score()  # Using existing abstract score method
        self.state.score = score
        self.state.is_completed = True
        if self.state.end_time is None:
            self.state.end_time = datetime.now(tz=timezone.utc)
        return score

    def get_completed_trial_data(self) -> dict:
        """Get all data for the completed trial"""
        # Calculate duration
        duration = None
        if self.state.start_time and self.state.end_time:
            duration = (self.state.end_time - self.state.start_time).total_seconds()

        # Build state dict with all needed data
        state_data = {
            "task_id": self.state.task_id,
            "trial_id": self.state.trial_id,
            "is_completed": self.state.is_completed,
            "score": self.state.score,
            "submitted_answer": self.state.submitted_answer,
            "duration": duration,
            "tool_statistics": self._get_complete_tool_statistics(),
        }

        return {"trial_id": self.state.trial_id, "state": state_data}

    def _get_complete_tool_statistics(self) -> dict:
        """Get complete tool statistics including individual tool calls"""
        stats = self.state.get_tool_statistics()

        # Add individual tool calls with duration
        stats["tool_calls"] = [
            {
                "tool_name": call.tool_name,
                "arguments": call.arguments,
                "result": call.result,
                "status": call.status.value,
                "error_message": call.error_message,
                "duration": call.duration,
                "timestamp": call.timestamp.isoformat() if call.timestamp else None,
            }
            for call in self.state.tool_calls
        ]

        return stats
