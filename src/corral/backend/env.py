import inspect
import time
from abc import ABC, abstractmethod
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from loguru import logger

from corral.backend.schema import ToolCall, ToolCallStatus
from corral.backend.tool import Tool


class Role(StrEnum):
    """Defines the role type of a component in the system."""

    AGENT = "agent"
    ENVIRONMENT = "environment"
    TOOL = "tool"


@dataclass
class LLMMessage:
    role: Role  # 'agent' or 'environment'
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))


@dataclass
class TaskState:
    task_id: str
    task_prompt: str
    trial_id: str = "0"
    messages: list[LLMMessage] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    is_attempted: bool = False
    score: float | None = None
    submitted_answer: str | None = None
    feedback: str | None = None
    surrendered: bool = False
    env_start_time: datetime = field(
        default_factory=lambda: datetime.now(tz=timezone.utc)
    )
    env_end_time: datetime | None = None

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

    def get_env_duration(self) -> float | None:
        """Get how long the environment was active (not task duration)."""
        if self.env_end_time and self.env_start_time:
            return (self.env_end_time - self.env_start_time).total_seconds()
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
            if self.state.is_attempted and self.state.env_end_time is None:
                self.state.env_end_time = datetime.now(tz=timezone.utc)
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

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Return tools in OpenAI function-calling format."""
        return [t.get_openai_tool_format() for t in self.tools.values()]

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

    def _preprocess_arguments(
        self, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Parse JSON strings based on the tool's expected argument types"""
        import json

        # Get the tool definition
        tool = self.tools.get(tool_name)
        if not tool:
            return args

        # Build a map of argument name -> expected type
        expected_types = {}
        for arg in tool.arguments:
            expected_types[arg.name] = arg.type

        def should_parse_as_json(expected_type: str) -> bool:
            """Determine if a type should be parsed from JSON string"""
            if expected_type is None:
                return False

            # Handle union types like "dict | None", "list[str] | None"
            type_parts = [part.strip() for part in expected_type.split("|")]

            for type_part in type_parts:
                # Skip None type
                if type_part.lower() in ("none", "nonetype"):
                    continue

                # Check if any part of the union should be parsed
                if (
                    type_part in ["dict", "object", "list", "array"]
                    or "dict[" in type_part
                    or "list[" in type_part
                    or type_part.startswith(("list", "array"))
                ):
                    return True

            return False

        def parse_value(key: str, value: Any) -> Any:
            expected_type = expected_types.get(key)

            if isinstance(value, str) and value.startswith(("{", "[")):
                if should_parse_as_json(expected_type):
                    try:
                        parsed = json.loads(value)
                        logger.debug(
                            f"🎯 Parsed {key} from JSON string (type: {expected_type})"
                        )
                        return parsed
                    except json.JSONDecodeError as e:
                        logger.error(
                            f"Failed to parse JSON for argument '{key}': {e}. Returning original value."
                        )
                        return value
                else:
                    logger.debug(f"⏭️  Skipping {key} (expected type: {expected_type})")
                    return value
            return value

        return {key: parse_value(key, val) for key, val in args.items()}

    def call_tool(self, tool_name: str, arguments: dict[str, Any]) -> ToolCall:
        """Execute a tool and record the call with enhanced error handling.

        The hidden arguments, if they exist, will be merged into the call arguments,
        with the hidden arguments taking precedence.
        This is needed for cases in which the arguments are fixed and should not be modified and/or provided by the agent."""

        logger.debug(f"🔍 BEFORE preprocessing - {tool_name}:")
        logger.debug(f"   Arguments: {arguments}")
        for key, value in arguments.items():
            logger.debug(f"   {key}: {type(value)} = {value!r}")

        arguments = self._preprocess_arguments(tool_name, arguments)

        logger.debug(f"✅ AFTER preprocessing - {tool_name}:")
        logger.debug(f"   Arguments: {arguments}")
        for key, value in arguments.items():
            logger.debug(f"   {key}: {type(value)} = {value!r}")

        # Store original arguments for the ToolCall record (after preprocessing)
        original_arguments = arguments.copy()

        start_time = time.perf_counter()
        # Check if tool exists
        if tool_name not in self.tools:
            duration = time.perf_counter() - start_time
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=original_arguments,
                result=None,
                status=ToolCallStatus.INVALID_TOOL,
                error_message=f"Tool {tool_name} not found",
                duration=duration,
            )
            self.state.tool_calls.append(tool_call)
            return tool_call

        tool = self.tools[tool_name]

        # Merge tool-specific hidden_args if present
        call_args = arguments.copy()
        if hasattr(tool, "hidden_args") and tool.hidden_args:
            # tool.hidden_args is a list of argument names to hide
            if not hasattr(self, "hidden_args") or self.hidden_args is None:
                raise AttributeError(
                    "Environment is missing required 'hidden_args' attribute."
                )
            for hidden_arg in tool.hidden_args:
                if hidden_arg in self.hidden_args:
                    call_args[hidden_arg] = self.hidden_args[hidden_arg]
                else:
                    raise KeyError(
                        f"Hidden argument '{hidden_arg}' required by tool '{tool_name}' not found in environment's hidden_args."
                    )

        # Validate arguments
        is_valid, error_message = tool.validate_arguments(call_args)
        if not is_valid:
            duration = time.perf_counter() - start_time
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=original_arguments,
                result=None,
                status=ToolCallStatus.INVALID_ARGS,
                error_message=error_message,
                duration=duration,
            )
            self.state.tool_calls.append(tool_call)
            return tool_call

        # Execute tool
        try:
            result = tool.execute(**call_args)
            duration = time.perf_counter() - start_time
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=original_arguments,
                result=result,
                status=ToolCallStatus.SUCCESS,
                error_message=None,
                duration=duration,
            )
        except Exception as e:
            duration = time.perf_counter() - start_time
            tool_call = ToolCall(
                tool_name=tool_name,
                arguments=original_arguments,
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
        self.state.is_attempted = True
        return score

    def surrender(self) -> float:
        """Surrender from the current task without submitting an answer"""
        self.state.surrendered = True
        self.state.is_attempted = True
        return 0.0 if self.state.score is None else self.state.score

    def get_completed_trial_data(self) -> dict:
        """Get all data for the completed trial"""

        # Build state dict with all needed data
        state_data = {
            "task_id": self.state.task_id,
            "trial_id": self.state.trial_id,
            "is_attempted": self.state.is_attempted,
            "score": self.state.score,
            "submitted_answer": self.state.submitted_answer,
            "surrendered": self.state.surrendered,
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

    def configure_additional_apps(self):
        """Configure any external apps/services (for example, experimental instruments or robots) needed for the environment.
        This method is called at the start of each trial."""
        # This can be overridden by subclasses to set up external dependencies

        return "No external app/service configuration needed for this trial."

    def _extract_scoring_fn_details(self, fn: Any) -> dict[str, Any]:
        """
        Extract details from a scoring function for LaTeX documentation.

        Args:
            fn: The scoring function to extract details from

        Returns:
            Dictionary with name, description, arguments, and returns info
        """
        fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)
        docstring = fn.__doc__ or ""

        # Parse docstring to extract description, args, and returns
        description = ""
        args_section = ""
        returns_section = ""

        if docstring:
            lines = docstring.strip().split("\n")
            current_section = "description"
            description_lines = []
            args_lines = []
            returns_lines = []

            for line in lines:
                stripped = line.strip()
                if stripped.lower().startswith("args:"):
                    current_section = "args"
                    continue
                if stripped.lower().startswith("returns:"):
                    current_section = "returns"
                    continue
                if stripped.lower().startswith("raises:"):
                    current_section = "raises"
                    continue

                if current_section == "description":
                    description_lines.append(line)
                elif current_section == "args":
                    args_lines.append(line)
                elif current_section == "returns":
                    returns_lines.append(line)

            description = "\n".join(description_lines).strip()
            args_section = "\n".join(args_lines).strip()
            returns_section = "\n".join(returns_lines).strip()

        # Extract arguments from signature
        structured_args = []
        try:
            sig = inspect.signature(fn)
            for param_name, param in sig.parameters.items():
                arg_type = ""
                if param.annotation != inspect.Parameter.empty:
                    arg_type = (
                        param.annotation.__name__
                        if hasattr(param.annotation, "__name__")
                        else str(param.annotation)
                    )

                required = param.default == inspect.Parameter.empty
                default = (
                    None if param.default == inspect.Parameter.empty else param.default
                )

                # Try to find description in docstring args section
                arg_description = ""
                if args_section:
                    # Look for pattern like "param_name (type): description" or "param_name: description"
                    for raw_arg_line in args_section.split("\n"):
                        stripped_arg_line = raw_arg_line.strip()
                        if stripped_arg_line.startswith(param_name):
                            # Extract description after the colon
                            if ":" in stripped_arg_line:
                                arg_description = stripped_arg_line.split(":", 1)[
                                    1
                                ].strip()
                            break

                structured_args.append(
                    {
                        "name": param_name,
                        "type": arg_type,
                        "description": arg_description,
                        "required": required,
                        "default": default,
                    }
                )
        except (ValueError, TypeError):
            # If we can't get signature, just use empty args
            pass

        return {
            "name": fn_name,
            "description": description,
            "arguments": structured_args,
            "returns": returns_section,
        }

    def to_latex(
        self,
        output_dir: str,
        level: int | str,
        env_name: str | None = None,
        task_name: str | None = None,
        verbosity: str | None = None,
    ) -> tuple[str, str, str | None]:
        """
        Generate LaTeX documentation for this task.

        This method creates a `TaskDefinition` from the environment's task data
        and delegates to `Code2Latex.colorbox()` for generating formatted LaTeX files
        and `Code2Latex.longtable()` for generating tools documentation.

        If the environment has a `current_task` attribute (e.g., `TaskGroupEnvironment`),
        it will automatically detect:
        - Whether this is a subtask (based on `input_from_tasks`)
        - Dependencies on other tasks
        - `env_name` from `task_group.group_id` if not provided

        Args:
            output_dir: Directory for output .tex files
            level: Task level identifier (e.g., 1, 2, "advanced")
            env_name: Environment name (e.g., "afm", "catalyst"). If not provided,
                     will try to get from task_group.group_id
            task_name: Optional custom name for the task (defaults to task_id)
            verbosity: Tool verbosity level used to filter tool descriptions and
                       return sections. Accepts a `ToolVerbosity` value string
                       (e.g. "brief", "detailed"). Defaults to
                       `ToolVerbosity.DETAILED` when not provided.

        Returns:
            Tuple of (task_tex_path, tools_tex_path, scoring_tex_path) - paths to the generated .tex files
        """
        # Import here to avoid circular imports
        from corral.router.verbosity import ToolVerbosity, VerbosityConfig
        from corral.utils.code2latex import Code2Latex, LatexMetadata

        # Resolve verbosity level (default to DETAILED)
        if verbosity is None:
            resolved_verbosity = ToolVerbosity.DETAILED
        elif isinstance(verbosity, ToolVerbosity):
            resolved_verbosity = verbosity
        else:
            resolved_verbosity = ToolVerbosity.FULL

        # Map verbosity levels to the RETURNS_* sections that should be included.
        RETURNS_VERBOSITY_MAP: dict[ToolVerbosity, list[str]] = {
            ToolVerbosity.BRIEF: ["RETURNS_BRIEF"],
            ToolVerbosity.DETAILED: ["RETURNS_BRIEF", "RETURNS_DETAILED"],
        }

        # Get task description from prompt
        description = str(self.get_task_prompt())

        # Get list of tool names
        tools = list(self.tools.keys())

        # Get detailed tool information for longtable using the resolved verbosity
        tools_details = []

        for tool in self.tools.values():
            filtered_description = VerbosityConfig.filter_tool_description(
                tool.description, resolved_verbosity
            )

            # Extract RETURNS section from the original description and filter it
            sections = VerbosityConfig.extract_all_sections(tool.description)
            return_keys = RETURNS_VERBOSITY_MAP.get(
                resolved_verbosity,
                ["RETURNS_BRIEF", "RETURNS_DETAILED", "RETURNS_EXAMPLES"],
            )
            returns_parts = [sections.get(key, "") for key in return_keys]
            returns_raw = "\n\n".join(part for part in returns_parts if part)
            returns_info = (
                VerbosityConfig.filter_argument_description(
                    returns_raw, resolved_verbosity
                )
                if returns_raw
                else ""
            )

            structured_args = []
            for arg in tool.arguments:
                filtered_arg_desc = VerbosityConfig.filter_argument_description(
                    arg.description, resolved_verbosity
                )
                structured_args.append(
                    {
                        "name": arg.name,
                        "type": arg.type,
                        "description": filtered_arg_desc,
                        "required": arg.required,
                        "default": arg.default,
                        "choices": arg.choices,
                    }
                )

            tools_details.append(
                {
                    "name": tool.name,
                    "description": filtered_description,
                    "arguments": structured_args,
                    "returns": returns_info,
                }
            )

        # Check if this is a TaskGroupEnvironment with current_task
        scoring_fn = self.score
        if (
            hasattr(self, "current_task")
            and self.current_task is not None
            and hasattr(self.current_task, "scoring_fn")
            and self.current_task.scoring_fn is not None
        ):
            scoring_fn = self.current_task.scoring_fn

        # Try to get env_name from task_group if not provided
        if env_name is None:
            if hasattr(self, "task_group") and self.task_group is not None:
                env_name = self.task_group.group_id
            else:
                env_name = "unknown"

        # Create LatexMetadata
        metadata = LatexMetadata(
            env_name=env_name,
            level=level,
        )

        # Generate LaTeX via Code2Latex.colorbox for tasks
        task_tex_path = Code2Latex.colorbox(
            name=task_name or self.task_id,
            description=description,
            tools=tools,
            scoring_fn=scoring_fn,
            metadata=metadata,
            output_dir=output_dir,
        )

        # Generate LaTeX via Code2Latex.longtable for tools
        tools_tex_path = Code2Latex.longtable(
            tools=tools_details,
            metadata=metadata,
            output_dir=output_dir,
        )

        # Collect scoring functions from all tasks in the task group
        scoring_fns_details = []
        seen_scoring_fns = set()  # Track by function name to deduplicate

        if hasattr(self, "task_group") and self.task_group is not None:
            for task in self.task_group.tasks.values():
                if task.scoring_fn is not None:
                    fn = task.scoring_fn
                    fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)

                    # Skip if we've already processed this function
                    if fn_name in seen_scoring_fns:
                        continue
                    seen_scoring_fns.add(fn_name)

                    # Extract function details
                    fn_details = self._extract_scoring_fn_details(fn)
                    scoring_fns_details.append(fn_details)
        else:
            # Single task environment - use self.score method
            if hasattr(self, "score") and callable(self.score):
                fn = self.score
                fn_name = fn.__name__ if hasattr(fn, "__name__") else "score"
                if fn_name not in seen_scoring_fns:
                    fn_details = self._extract_scoring_fn_details(fn)
                    scoring_fns_details.append(fn_details)

        # Generate LaTeX via Code2Latex.scoring_longtable for scoring functions
        scoring_tex_path = None
        if scoring_fns_details:
            scoring_tex_path = Code2Latex.scoring_longtable(
                scoring_functions=scoring_fns_details,
                metadata=metadata,
                output_dir=output_dir,
            )

        return task_tex_path, tools_tex_path, scoring_tex_path
