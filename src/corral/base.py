import copy
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, StrEnum
from typing import Any

from pydantic import BaseModel

from corral.graph import GraphTrackerFactory, NodeType


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

    def __init__(self, task_id: str, graph_factory: GraphTrackerFactory | None):
        self.task_id = task_id
        self.tools: dict[str, Tool] = {}
        self.trial_states: dict[str, TaskState] = {}
        self.trial_counter = -1

        # Graph tracking components
        self.graph_factory = graph_factory or GraphTrackerFactory()
        self.graph_tracker = None

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

            # Save the graph data for the completed trial if tracking is enabled
            if self.graph_tracker:
                self.graph_factory.save_tracker(
                    task_id=self.task_id, trial_id=self.state.trial_id, visualize=True
                )

        self.trial_counter += 1
        new_trial_id = str(self.trial_counter)

        self.state = TaskState(
            task_id=self.task_id,
            trial_id=new_trial_id,
            task_prompt=self.get_task_prompt(),
        )
        # Create a new graph tracker for this trial
        self.graph_tracker = self.graph_factory.create_tracker(
            task_id=self.task_id, agent_type="Environment", trial_id=new_trial_id
        )

        # Track the initial task prompt
        if self.graph_tracker:
            self.graph_tracker.add_node(
                node_type=NodeType.LLM_PROMPT,
                content=self.get_task_prompt(),
                metadata={"type": "task_prompt"},
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

    def add_message(self, role: Role, content: str) -> None:
        """Add a message to the environment state and track it in the graph

        Args:
            role: Role of the message sender
            content: Content of the message
        """
        # Create message
        message = LLMMessage(role=role, content=content)
        self.state.messages.append(message)

        # Track in graph
        if self.graph_tracker:
            if role == Role.AGENT:
                self.graph_tracker.add_node(
                    node_type=NodeType.LLM_RESPONSE,
                    content=content,
                    metadata={"role": role.value},
                )
            else:
                self.graph_tracker.add_node(
                    node_type=NodeType.LLM_PROMPT,
                    content=content,
                    metadata={"role": role.value},
                )

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolCall:
        """Execute a tool and record the call"""
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

            # Track in graph
            if self.graph_tracker:
                tool_node_id = self.graph_tracker.track_tool_call(tool_call)

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

            # Track in graph
            if self.graph_tracker:
                tool_node_id = self.graph_tracker.track_tool_call(tool_call)

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

        # Track in graph
        if self.graph_tracker:
            tool_node_id = self.graph_tracker.track_tool_call(tool_call)

            # Track the response if successful
            if tool_call.status == ToolCallStatus.SUCCESS and tool_call.result:
                self.graph_tracker.track_tool_response(tool_call.result, tool_node_id)

        return tool_call

    def submit_answer(self, answer: str) -> float:
        """Submit final answer and get score"""
        self.state.submitted_answer = answer
        score = self.score()  # Using existing abstract score method
        self.state.score = score
        self.state.is_completed = True
        self.state.end_time = datetime.now(tz=timezone.utc)

        # Track in graph
        if self.graph_tracker:
            self.graph_tracker.track_final_submission(answer, score)

        return score

    def get_graph_statistics(self) -> dict[str, Any]:
        """Get statistics about the current graph

        Returns:
            Dictionary of statistics or empty dict if no graph tracker
        """
        if self.graph_tracker:
            return self.graph_tracker.get_statistics()
        return {}


### Classes related to subtasks


@dataclass
class TaskDefinition:
    """Definition of a task with its requirements and scoring"""

    name: str
    description: str
    tools: list[str]
    scoring_fn: Callable[[dict | str], float]
    submission_format: dict[str, str]
    # Either use output from another task or custom input
    input_from_tasks: list[str] = field(
        default_factory=list
    )  # input required from some of the previous tasks in the group
    initial_input: dict[str, Any] = field(
        default_factory=dict
    )  # initial input for the task, if required

    # Helper method to check if task has dependencies
    def has_dependencies(self) -> bool:
        return len(self.input_from_tasks) > 0


@dataclass
class TaskGroup:
    """Container for related tasks"""

    group_id: str
    tasks: dict[str, TaskDefinition]
    results: dict[str, Any] = field(default_factory=dict)
    scores: dict[str, float] = field(default_factory=dict)

    def get_task_input(self, task_id: str) -> dict[str, Any]:
        """Get input for a task either from other tasks or initial input"""
        task = self.tasks.get(task_id)
        if not task:
            return {}

        # Start with the initial input
        combined_input = task.initial_input.copy() if task.initial_input else {}

        # Add inputs from dependent tasks
        for dep_task_id in task.input_from_tasks:
            if dep_task_id in self.results:
                # Add the dependent task's result to the input
                dep_result = self.results[dep_task_id]
                if isinstance(dep_result, dict) and "answer" in dep_result:
                    # Extract the relevant part of the result
                    combined_input[f"result_from_{dep_task_id}"] = dep_result["answer"]
                else:
                    combined_input[f"result_from_{dep_task_id}"] = dep_result

        return combined_input

    def store_result(self, task_id: str, result: dict[str, Any], score: float) -> None:
        """Store task result and score"""
        self.results[task_id] = result
        self.scores[task_id] = score

    def get_task_dependencies(self) -> dict[str, list[str]]:
        """Get dictionary of task dependencies"""
        return {task_id: task.input_from_tasks for task_id, task in self.tasks.items()}

    def get_ordered_tasks(self) -> list[str]:
        """Return tasks in dependency order"""
        # Simple topological sort
        dependencies = self.get_task_dependencies()
        visited = set()
        ordered = []

        def visit(task_id):
            if task_id in visited:
                return
            visited.add(task_id)
            for dep in dependencies.get(task_id, []):
                visit(dep)
            ordered.append(task_id)

        for task_id in self.tasks:
            visit(task_id)

        return ordered

    def check_dependencies_satisfied(self, task_id: str) -> bool:
        """Check if all dependencies for a task are satisfied"""
        task = self.tasks.get(task_id)
        if not task:
            return False

        return all(
            dep_task_id in self.results and self.results[dep_task_id] is not None
            for dep_task_id in task.input_from_tasks
        )


@dataclass
class TaskGroups:
    pass
