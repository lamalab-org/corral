import inspect
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from loguru import logger

from corral.backend.schema import ToolCall, ToolCallStatus
from corral.backend.state import CorralState, TaskRunState
from corral.backend.task import (
    TaskDefinition,
    build_dependency_graph,
    connected_components,
    topological_order,
    validate_task_graph,
)
from corral.backend.tool import Tool


def default_file_tools(workspace: str) -> dict[str, Tool]:
    """Build the standard filesystem tools bound to a trial workspace."""
    # Imported lazily to avoid a circular import at module load time.
    from corral.utils.io_tools import (
        CatFilesTool,
        CopyFileTool,
        FileInfoTool,
        FSManager,
        ListFilesTool,
        ReadFileTool,
        WriteFileTool,
    )

    fs_manager = FSManager("file", base_path=workspace)
    return {
        "list_files": ListFilesTool(fs_manager),
        "read_file": ReadFileTool(fs_manager),
        "write_file": WriteFileTool(fs_manager),
        "file_info": FileInfoTool(fs_manager),
        "cat_files": CatFilesTool(fs_manager),
        "copy_file": CopyFileTool(fs_manager),
    }


@dataclass(frozen=True)
class Toolset:
    """A single description of "the tools for this benchmark."

    Collapses the old `available_tools` / `common_tools` / `file_tools` /
    `file_tool_factory` quartet into one value object. Every tool falls on two
    axes: *when* it can be built (statically, or bound per-trial to a workspace
    path) and *how* it is selected (by name from the pool, or for every task).
    The environment owns resolution; callers pass one `Toolset`.
    """

    # Named pool: a task selects from this via `task.tools`.
    pool: dict[str, Tool] = field(default_factory=dict)
    # Tools every task receives regardless of its `tools` list.
    common: dict[str, Tool] = field(default_factory=dict)
    # Per-trial, workspace-bound tools. `None` => no workspace tools
    # (replaces `file_tools=False`); a custom callable replaces the old
    # `file_tool_factory`; the default reproduces today's file tools.
    workspace_factory: Callable[[str], dict[str, Tool]] | None = default_file_tools
    # Opt-in: an empty `task.tools` means "the whole pool" (wetlab semantics).
    select_all_when_unspecified: bool = False

    def resolve(self, task: TaskDefinition, workspace: str | None) -> dict[str, Tool]:
        """Resolve the concrete tools for `task` in `workspace`.

        Named tools are picked from the pool (or the whole pool when
        `select_all_when_unspecified` and `task.tools` is empty), minus any
        `task.excluded_tools`; `common` tools are always added; and
        workspace-bound tools are appended when a workspace exists.
        """
        excluded = set(getattr(task, "excluded_tools", ()) or ())
        names = task.tools or (
            list(self.pool) if self.select_all_when_unspecified else []
        )
        tools: dict[str, Tool] = {}
        for name in names:
            if name in excluded:
                continue
            tool = self.pool.get(name)
            if tool is None:
                logger.warning(f"Tool {name!r} not in pool for task {task.name!r}")
                continue
            tools[tool.name] = tool
        tools.update(self.common)
        if workspace and self.workspace_factory is not None:
            tools.update(self.workspace_factory(workspace))
        return tools


class Environment:
    """Runtime engine for a single task (chained or standalone).

    There is one concrete environment, not a hierarchy. A task is just a
    `TaskDefinition`; chaining is expressed through `input_map` and grouping is
    derived (see `build_environments`). Custom behaviour is injected through the
    definition's hooks (`prompt_fn`, `setup_fn`, `scoring_fn`) rather than by
    subclassing — subclassing remains only as an escape hatch for genuinely
    stateful integrations (hardware, pickled inventories).

    All mutable runtime data lives in a single `CorralState` instance
    (`self.state`); the environment itself only holds immutable configuration
    (task id, task definition, tool pools, base workspace dir). Linked tasks
    share the same `task_runs` store so each one sees its dependencies' outputs
    through `self.state`.
    """

    def __init__(
        self,
        task_id: str,
        task: TaskDefinition,
        base_work_dir: str = "",
        *,
        toolset: Toolset | None = None,
        group_tasks: dict[str, TaskDefinition] | None = None,
        component_id: str | None = None,
        shared_task_runs: dict[str, TaskRunState] | None = None,
        fs_manager=None,
    ):
        self.task_id = task_id
        self.current_task = task
        self.base_work_dir = base_work_dir
        self.toolset = toolset or Toolset()
        self.group_tasks = group_tasks or {task_id: task}
        self.fs_manager = fs_manager
        self.tools: dict[str, Tool] = {}

        # task_prompt stays an empty placeholder; the real prompt is fetched
        # live via get_task_prompt() and recorded in the agent's messages.
        self.state = CorralState(
            task_id=task_id,
            task_prompt="",
            task_group_id=component_id,
        )
        # Linked tasks share this dict (same object across environments)
        if shared_task_runs is not None:
            self.state.task_runs = shared_task_runs

        # reset_state resolves the tools for the first trial (named + common +
        # workspace-bound), so there is no separate tool-setup step here.
        self.reset_state()

    def save_current_state(self) -> dict[str, Any]:
        """Snapshot the current trial state as a plain dict."""
        return self.state.snapshot(include_trials=False)

    def reset_state(self) -> str:
        """Archive the finished trial (if any) and start a fresh one.

        Returns the new trial id.
        """
        next_trial_id = str(self.state.trial_counter + 1)

        workspace = (
            self._create_trial_workspace(next_trial_id) if self.base_work_dir else None
        )

        trial_id = self.state.start_new_trial(task_prompt="", workspace=workspace)
        # Resolve all tools for the new trial: named tools from the pool, common
        # tools, and workspace-bound tools rebound to the fresh workspace.
        self.tools = self._resolve_tools(self.state.workspace)
        # The prompt is intentionally NOT built here: at construction time deps
        # have not run yet. It is fetched live when actually requested (agent,
        # guide, LaTeX) — always after dependencies are satisfied under the
        # runner's enforced topological order — and is captured in the agent's
        # messages, which is what the trace is saved from. `state.task_prompt`
        # stays an empty placeholder; nothing downstream reads it.
        return trial_id

    def _create_trial_workspace(self, trial_id: str) -> str:
        """Create workspace directory for this trial"""
        workspace = Path(self.base_work_dir) / f"{self.task_id}_trial_{trial_id}"
        logger.info(f"Creating workspace: {workspace}")
        if self.fs_manager:
            self.fs_manager.mkdir(str(workspace), create_parents=True)
        else:
            workspace.mkdir(parents=True, exist_ok=True)
        return str(workspace)

    @property
    def current_work_dir(self) -> str | None:
        """Workspace of the active trial (proxy for `state.workspace`)."""
        return self.state.workspace

    @current_work_dir.setter
    def current_work_dir(self, value: str | None) -> None:
        self.state.workspace = value

    @property
    def hidden_args(self) -> dict[str, Any]:
        """Hidden tool arguments (proxy for `state.hidden_args`)."""
        return self.state.hidden_args

    @hidden_args.setter
    def hidden_args(self, value: dict[str, Any] | None) -> None:
        self.state.hidden_args = value or {}

    @property
    def trial_states(self) -> dict[str, dict[str, Any]]:
        """Archived trial snapshots (proxy for `state.trials`)."""
        return self.state.trials

    def get_current_work_dir(self) -> str:
        """Get the current working directory for this trial"""
        return self.state.workspace or self.base_work_dir or ""

    def get_task_prompt(self) -> str | list[dict]:
        """Return the task prompt, using the definition's hook if provided."""
        if self.current_task.prompt_fn is not None:
            return self.current_task.prompt_fn(self)
        return self._default_task_prompt()

    def _default_task_prompt(self) -> str:
        """Render description + submission format + resolved inputs.

        Dependency inputs are resolved strictly: by the time the prompt is
        requested every dependency has run (the runner enforces topological
        order), so an unsatisfied dependency is a misuse error rather than a
        state to render. There is therefore no "not yet available" placeholder.
        """
        task = self.current_task
        prompt = (
            f"Task: {task.name}\n"
            f"Description: {task.description}\n\n"
            f"Required submission format:\n{task.submission_format}\n\n"
        )

        # Strict resolution: raises if a dependency has not produced an output.
        resolved = self.state.resolve_inputs(task)

        prompt += "\nAvailable input data:\n"

        # Display resolved inputs from dependencies
        for input_name, ref in task.input_map.items():
            prompt += f"- {input_name} (from {ref.task_id}): {resolved[input_name]}\n"

        # Display initial input data
        for key, value in task.initial_input.items():
            if key != "work_dir":
                prompt += f"- {key}: {value}\n"

        # Add workspace info
        if self.state.workspace:
            prompt += "\nIMPORTANT: You have access to filesystem tools. All files will be saved in your isolated workspace.\n"

        return prompt

    def score(self) -> float:
        """Score the submitted answer with the definition's scoring function.

        An output is *always* stored — including on the no-submission and
        scoring-error paths — so the run store never has holes for a task that
        ran. Dependent tasks then either consume a real output or are skipped by
        the runner's broken-chain short-circuit; `resolve_inputs` never raises
        by accident.
        """
        task = self.current_task
        if not self.state.submitted_answer:
            logger.warning(f"No submission found for task {self.task_id}")
            self.state.store_task_output(
                self.task_id, "", 0.0, feedback="no submission"
            )
            return 0.0

        try:
            answer = self.state.submitted_answer.strip()
            resolved = self._resolve_answer(answer)
            score = task.scoring_fn(resolved)
            # Store the output in the shared state so dependent tasks can use it
            self.state.store_task_output(self.task_id, resolved, score)
            logger.info(f"Task {self.task_id} scored: {score}")
            return score
        except Exception as e:
            logger.error(
                f"Error scoring submission for task {self.task_id}: {e!s}",
                exc_info=True,
            )
            logger.error(f"Submission was: {self.state.submitted_answer!r}")
            self.state.store_task_output(
                self.task_id,
                self.state.submitted_answer,
                0.0,
                feedback=f"scoring error: {e}",
            )
            return 0.0

    def _resolve_answer(self, answer: str) -> str:
        """Resolve a submitted answer to a file path when the task expects one."""
        if not self.current_task.resolve_answer:
            return answer
        # JSON submissions are values, not file paths.
        if answer.startswith("{") and answer.endswith("}"):
            return answer
        from corral.utils.tool_helpers import smart_resolve_path

        return smart_resolve_path(answer)

    def _resolve_tools(self, workspace: str | None) -> dict[str, Tool]:
        """Resolve the trial's tools — overridable seam over the toolset.

        Subclasses needing bespoke tool policy override this; the default
        delegates to `self.toolset` (named pool + common + workspace tools).
        """
        return self.toolset.resolve(self.current_task, workspace)

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

        # Preprocess (e.g. JSON-decode stringified args), then emit a single
        # concise debug line per call. Only spell out the before/after when
        # preprocessing actually changed something — the previous per-argument
        # dump logged 4-6 lines for every call (4 lines of nothing for no-arg
        # tools), which drowned the logs.
        preprocessed = self._preprocess_arguments(tool_name, arguments)
        if preprocessed != arguments:
            logger.debug(
                f"call_tool {tool_name}: preprocessed {arguments!r} -> {preprocessed!r}"
            )
        else:
            logger.debug(f"call_tool {tool_name}({arguments!r})")
        arguments = preprocessed

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
            self.state.record_tool_call(tool_call)
            return tool_call

        tool = self.tools[tool_name]

        # Merge tool-specific hidden_args if present
        call_args = arguments.copy()
        if hasattr(tool, "hidden_args") and tool.hidden_args:
            # tool.hidden_args is a list of argument names to hide
            for hidden_arg in tool.hidden_args:
                if hidden_arg in self.state.hidden_args:
                    call_args[hidden_arg] = self.state.hidden_args[hidden_arg]
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
            self.state.record_tool_call(tool_call)
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

        self.state.record_tool_call(tool_call)
        return tool_call

    def submit_answer(self, answer: str) -> float:
        """Submit final answer and get score"""
        self.state.submit(answer)
        score = self.score()
        self.state.set_score(score)
        return score

    def surrender(self) -> float:
        """Surrender from the current task without submitting an answer"""
        return self.state.surrender()

    def get_completed_trial_data(self) -> dict:
        """Get all data for the completed trial"""
        return {
            "trial_id": self.state.trial_id,
            "state": self.state.snapshot(include_trials=False),
        }

    def configure_additional_apps(self):
        """Configure any external apps/services (for example, experimental
        instruments or robots) needed for the environment, and populate any
        hidden tool arguments. Called at the start of each trial.

        Behaviour is supplied by the definition's `setup_fn`; the default is a
        no-op.
        """
        if self.current_task.setup_fn is not None:
            result = self.current_task.setup_fn(self)
            return result or "Additional apps/services configured for this trial."
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

        This method creates LaTeX from the environment's task data and delegates
        to `Code2Latex.colorbox()` for the task block, `Code2Latex.longtable()`
        for tools, and `Code2Latex.scoring_longtable()` for scoring functions.

        It automatically detects dependencies from the definition's `input_map`
        and falls back to `state.task_group_id` for `env_name` when not provided.

        Args:
            output_dir: Directory for output .tex files
            level: Task level identifier (e.g., 1, 2, "advanced")
            env_name: Environment name (e.g., "afm", "catalyst"). If not provided,
                     will try to get from state.task_group_id
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

        # The scoring function comes straight from the (immutable) definition
        scoring_fn = self.current_task.scoring_fn

        # Try to get env_name from the task group id if not provided
        if env_name is None:
            env_name = self.state.task_group_id or "unknown"

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

        # Collect scoring functions from all linked tasks in this component
        scoring_fns_details = []
        seen_scoring_fns = set()  # Track by function name to deduplicate

        for task in self.group_tasks.values():
            if task.scoring_fn is None:
                continue
            fn = task.scoring_fn
            fn_name = fn.__name__ if hasattr(fn, "__name__") else str(fn)

            # Skip if we've already processed this function
            if fn_name in seen_scoring_fns:
                continue
            seen_scoring_fns.add(fn_name)

            # Extract function details
            scoring_fns_details.append(self._extract_scoring_fn_details(fn))

        # Generate LaTeX via Code2Latex.scoring_longtable for scoring functions
        scoring_tex_path = None
        if scoring_fns_details:
            scoring_tex_path = Code2Latex.scoring_longtable(
                scoring_functions=scoring_fns_details,
                metadata=metadata,
                output_dir=output_dir,
            )

        return task_tex_path, tools_tex_path, scoring_tex_path


def build_environments(
    tasks: Mapping[str, TaskDefinition],
    *,
    base_work_dir: str = "",
    name: str | None = None,
    toolset: Toolset | None = None,
    fs_manager=None,
    env_cls: type[Environment] = Environment,
    **env_kwargs: Any,
) -> dict[str, Environment]:
    """Build one environment per task from a flat collection of definitions.

    Grouping is derived, not declared: each weakly-connected component of the
    dependency graph shares one run store, so chained tasks see their
    dependencies' outputs while independent tasks stay isolated. A single task
    is just the degenerate case of a one-node component.

    Args:
        tasks: Task definitions keyed by task id.
        base_work_dir: Base directory for per-trial workspaces.
        name: Optional benchmark label used for tracing/LaTeX (replaces the old
            `group_id` argument; never used to namespace task ids).
        toolset: Single description of the benchmark's tools (named pool, common
            tools, and the per-trial workspace tool factory). The environment
            resolves each task's concrete tools from it.
        fs_manager: Optional FSManager used to create trial workspaces. This is a
            storage backend, not a tool — kept separate from `toolset`.
        env_cls: Environment class to instantiate (escape hatch for stateful
            subclasses such as AFM/wetlab).
        **env_kwargs: Extra keyword arguments forwarded to `env_cls`.

    Returns:
        Environments keyed by task id, ready to serve.
    """
    validate_task_graph(tasks)

    logger.info("Task Dependencies:")
    for task_id, deps in build_dependency_graph(tasks).items():
        logger.info(f"- {task_id}: depends on {deps}")

    logger.info("Task Execution Order:")
    for i, task_id in enumerate(topological_order(tasks)):
        logger.info(f"{i + 1}. {task_id}")

    environments: dict[str, Environment] = {}
    for component in connected_components(tasks):
        component_tasks = {task_id: tasks[task_id] for task_id in component}
        # One shared run store per component; chained tasks read through it.
        shared_task_runs: dict[str, TaskRunState] = {}
        for task_id in component:
            environments[task_id] = env_cls(
                task_id=task_id,
                task=tasks[task_id],
                base_work_dir=base_work_dir,
                toolset=toolset,
                group_tasks=component_tasks,
                component_id=name,
                shared_task_runs=shared_task_runs,
                fs_manager=fs_manager,
                **env_kwargs,
            )

    return environments
