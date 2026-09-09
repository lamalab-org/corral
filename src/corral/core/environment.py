import asyncio
import contextlib
import hashlib
import json
import re
import threading
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

from corral.backend.background_tools import attach_background_tools
from corral.backend.jobs import (
    DEFAULT_JOB_CONCURRENCY,
    DEFAULT_POLL_AFTER_SECONDS,
    JobExecutor,
    JobManager,
)
from corral.core.action import with_submit_answer_tool
from corral.core.events import (
    ExecutionStarted,
    RuntimeUpdate,
    TaskConfigured,
    WorkspaceDelta,
)
from corral.core.state import (
    EnvironmentState,
    ExecutionState,
    RuntimeState,
    TaskOutput,
    TaskState,
)
from corral.core.task import (
    TaskDefinition,
    build_dependency_graph,
    connected_components,
    topological_order,
    validate_task_graph,
)
from corral.core.tool import Tool, ToolConcurrency
from corral.core.tool_catalog import (
    TOOL_CATALOG_METADATA_KEY,
    ToolCatalogSnapshot,
    validate_tool_catalog_binding,
)
from corral.core.transition import environment_operations
from corral.core.workspace import WorkspaceState
from corral.persistence.workspace import WorkspaceManager
from corral.report.logging import logger
from corral.workspace import WorkspaceFilesystem, build_workspace_tools


class _ReadWriteLock:
    """A readers-writer lock: many concurrent readers **or** one exclusive writer.

    Models the read/write split within a single `concurrency_key` resource
    (Section 9). `CONCURRENT_READ` tool calls take the read side and overlap
    each other; `CONCURRENT` calls and background jobs take the write side and
    run exclusively against every reader and writer on that key. It is
    writer-preferring — a waiting writer blocks *new* readers — so a steady
    stream of readers can never starve a background job that needs the resource,
    while already-admitted readers still finish first.

    Threading primitives are used because action execution is synchronous.
    `acquire`/`release` alias the *write* side
    so the lock is drop-in wherever an exclusive `threading.Lock` was expected
    (the JobManager's `key_lock_factory` treats a background job as a writer).
    """

    def __init__(self) -> None:
        self._cond = threading.Condition(threading.Lock())
        self._readers = 0
        self._writer = False
        self._writers_waiting = 0

    def acquire_read(self) -> None:
        with self._cond:
            while self._writer or self._writers_waiting > 0:
                self._cond.wait()
            self._readers += 1

    def release_read(self) -> None:
        with self._cond:
            self._readers -= 1
            if self._readers == 0:
                self._cond.notify_all()

    def acquire_write(self) -> None:
        with self._cond:
            self._writers_waiting += 1
            try:
                while self._writer or self._readers > 0:
                    self._cond.wait()
            finally:
                self._writers_waiting -= 1
            self._writer = True

    def release_write(self) -> None:
        with self._cond:
            self._writer = False
            self._cond.notify_all()

    # Write-side aliases so a `_ReadWriteLock` is drop-in for the JobManager's
    # `key_lock_factory` (a background job is a writer on its resource).
    def acquire(self) -> None:
        self.acquire_write()

    def release(self) -> None:
        self.release_write()

    @contextlib.contextmanager
    def read_lock(self) -> Iterator[None]:
        self.acquire_read()
        try:
            yield
        finally:
            self.release_read()

    @contextlib.contextmanager
    def write_lock(self) -> Iterator[None]:
        self.acquire_write()
        try:
            yield
        finally:
            self.release_write()


def _ensure_local_workspace_root(workspace: str | Path) -> Path:
    """Create and validate a local directory before binding workspace tools."""
    root = Path(workspace).expanduser()
    if root.is_symlink():
        raise ValueError("workspace root cannot be a symbolic link")
    if root.exists() and not root.is_dir():
        raise ValueError(f"workspace root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"workspace root must be a regular directory: {root}")
    return root.resolve()


def default_file_tools(workspace: str) -> dict[str, Tool]:
    """Build the standard filesystem tools bound to a task workspace."""
    root = _ensure_local_workspace_root(workspace)
    tools = build_workspace_tools(WorkspaceFilesystem(root))
    return {
        name: tools[name]
        for name in (
            "list_files",
            "read_file",
            "write_file",
            "file_info",
            "cat_files",
            "copy_file",
        )
    }


@dataclass(frozen=True)
class Toolset:
    """A single description of "the tools for this benchmark."

    Collapses the old `available_tools` / `common_tools` / `file_tools` /
    `file_tool_factory` quartet into one value object. Every tool falls on two
    axes: *when* it can be built (statically, or bound to a task workspace
    path) and *how* it is selected (by name from the pool, or for every task).
    The environment owns resolution; callers pass one `Toolset`.
    """

    # Named pool: a task selects from this via `task.tools`.
    pool: dict[str, Tool] = field(default_factory=dict)
    # Tools every task receives regardless of its `tools` list.
    common: dict[str, Tool] = field(default_factory=dict)
    # Per-task-execution workspace tools. `None` => no workspace tools
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
    """A stateless task definition and tool catalog.

    Execution State is never stored on this object. Callers append the event
    returned by :meth:`initial_event`, commit an agent decision, and pass the
    materialized projection and Action to :func:`corral.core.execute_action`. A task-bound
    definition may contain execution resources (a materialized workspace,
    locks, and a background-job executor), but those resources are not the
    durable source of truth and no interaction history lives here.
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
        fs_manager=None,
        task_execution_id: str | None = None,
        max_job_concurrency: int = DEFAULT_JOB_CONCURRENCY,
        job_executors: dict[str, JobExecutor] | None = None,
        workspace_manager: WorkspaceManager | None = None,
    ):
        self.task_id = task_id
        self.current_task = task
        self.base_work_dir = base_work_dir
        self.toolset = toolset or Toolset()
        self.group_tasks = group_tasks or {task_id: task}
        self.component_id = component_id
        self.fs_manager = fs_manager
        # Per-task background-job concurrency. Only used when a task's toolset
        # contains a background-capable tool, in which case each task execution
        # gets its own JobManager bounded by this many simultaneous jobs.
        self.max_job_concurrency = max_job_concurrency
        # Preconfigured job executors, keyed by the name tools request via
        # `@tool(executor=...)`. Built-in thread/process/subprocess backends
        # are created on demand; deployments can inject configured or custom
        # backends here.
        self.job_executors = job_executors
        self.job_manager: JobManager | None = None
        # The task orchestrator supplies an opaque execution id solely to
        # namespace materialization. Environment neither derives nor interprets
        # benchmark repetition indices.
        self.workspace_path = (
            self._create_task_workspace(task_execution_id)
            if task_execution_id is not None and self.base_work_dir
            else None
        )
        self.workspace_manager = workspace_manager
        if self.workspace_manager is None and self.workspace_path is not None:
            artifact_root = Path(self.base_work_dir) / ".corral" / "artifacts"
            self.workspace_manager = WorkspaceManager(artifact_root=artifact_root)
        # Templates resolve workspace tools against the configured root only to
        # expose their schemas. Mutable calls are accepted exclusively by a
        # bound task execution.
        schema_workspace = self.workspace_path or (self.base_work_dir or None)
        if schema_workspace and self.toolset.workspace_factory is not None:
            _ensure_local_workspace_root(schema_workspace)
        self.tools = self._resolve_tools(schema_workspace)

        # Per-runtime locks that protect shared resources from concurrent tool
        # calls. Agent adapters may issue tool calls in parallel, so execution
        # is serialised by each tool's `ToolConcurrency` mode.
        # `_resource_locks` are readers-writer locks keyed
        # by `concurrency_key`, shared with this runtime's JobManager, so one key
        # gives a read/write split — same-key `CONCURRENT_READ` readers overlap
        # while `CONCURRENT` writers and background jobs stay exclusive — and
        # different keys always run in parallel. They are threading locks
        # because action execution is synchronous.
        self._execution_lock = threading.RLock()
        self._resource_locks: dict[str, _ReadWriteLock] = {}
        self._resource_locks_guard = threading.Lock()

        self._attach_background_tools()

    def initial_event(
        self,
        *,
        dependency_outputs: Mapping[str, TaskOutput | Mapping[str, Any]] | None = None,
        actor_id: str = "agent_0",
        execution_id: str,
        started_at: datetime | None = None,
        model_metadata: Mapping[str, Any] | None = None,
        scaffold_metadata: Mapping[str, Any] | None = None,
    ) -> ExecutionStarted:
        """Build the small event that initializes an execution projection."""
        tool_catalog = self.tool_catalog_snapshot()
        workspace = WorkspaceState(
            id=str(uuid5(NAMESPACE_URL, f"corral:workspace:{execution_id}"))
        )
        if self.workspace_path:
            workspace = self.capture_workspace(workspace)
        environment_state = {
            "hidden_arguments": {},
            "jobs": self.job_manager.snapshot() if self.job_manager else {},
            "values": {},
        }
        runtime = RuntimeUpdate(
            status="running",
            started_at=started_at or datetime.now(timezone.utc),
            metadata={
                "actor_id": actor_id,
                "surrender_sentinel": "SURRENDER",
            },
        )
        dependency_values = {
            task_id: TaskOutput.model_validate(value).model_dump(mode="json")
            for task_id, value in (dependency_outputs or {}).items()
        }
        task_metadata: dict[str, Any] = {"id": self.task_id}
        provisional = ExecutionState(
            through_commit_hash="0" * 64,
            execution_id=execution_id,
            branch_id="main",
            task=TaskState(
                metadata=task_metadata,
                model=model_metadata or {},
                scaffold=scaffold_metadata or {},
                environment={
                    "name": self.component_id or type(self).__name__,
                    TOOL_CATALOG_METADATA_KEY: tool_catalog.model_dump(mode="json"),
                },
                dependency_outputs={
                    key: TaskOutput.model_validate(value)
                    for key, value in dependency_values.items()
                },
            ),
            environment=EnvironmentState(values=environment_state),
            workspace=workspace,
            runtime=RuntimeState(
                status=runtime.status or "created",
                started_at=runtime.started_at,
                metadata=runtime.metadata,
            ),
        )
        if not all(
            ref.task_id in provisional.dependency_outputs
            for ref in self.current_task.input_map.values()
        ):
            prompt = None
        else:
            prompt = self.get_task_prompt(provisional)

        if prompt is not None:
            task_metadata["prompt"] = prompt
        return ExecutionStarted(
            task=task_metadata,
            environment=environment_state,
            environment_metadata=provisional.task.environment,
            workspace=workspace,
            scaffold=scaffold_metadata or {},
            model=model_metadata or {},
            dependency_outputs=dependency_values,
            runtime=runtime,
        )

    def capture_workspace(
        self,
        previous: WorkspaceState,
        *,
        created_by_action: str | None = None,
    ) -> WorkspaceState:
        """Capture the task's materialized directory as a logical workspace."""
        if not self.workspace_path:
            return previous
        if self.workspace_manager is None:
            raise RuntimeError("a bound workspace requires a WorkspaceManager")
        return asyncio.run(
            self.workspace_manager.snapshot(
                self.workspace_path,
                previous=previous,
                created_by_action=created_by_action,
            )
        )

    def prepare_workspace(self, workspace: WorkspaceState) -> None:
        """Materialize a restored logical workspace into an empty task path."""
        if not self.workspace_path or not workspace.files:
            return
        destination = Path(self.workspace_path)
        if any(destination.iterdir()):
            # A running task already has its materialization. The directory is
            # exclusive, and capture_workspace verifies every
            # committed post-action manifest.
            return
        if self.workspace_manager is None:
            raise RuntimeError("a bound workspace requires a WorkspaceManager")
        asyncio.run(self.workspace_manager.materialize(workspace, destination))

    def for_task(
        self,
        task_execution_id: str,
        *,
        max_job_concurrency: int | None = None,
    ) -> "Environment":
        """Bind this definition to one isolated task execution.

        The returned definition shares immutable task/tool configuration and is
        bound to a workspace materialization namespaced by `task_execution_id`.
        Its projection is supplied by the runtime and never owned by Environment.

        Stateful subclasses that hold non-clonable resources (hardware handles,
        live clients, subprocess pools, or a bespoke `__init__` signature)
        should override this to build — or lease — their own isolated runtime
        rather than inherit this definition-only reconstruction.

        `max_job_concurrency` sizes this task execution's background-job pool;
        `None` falls back to the definition's default.
        """
        return type(self)(
            task_id=self.task_id,
            task=self.current_task,
            base_work_dir=self.base_work_dir,
            toolset=self.toolset,
            group_tasks=self.group_tasks,
            component_id=self.component_id,
            fs_manager=self.fs_manager,
            task_execution_id=task_execution_id,
            max_job_concurrency=(
                max_job_concurrency
                if max_job_concurrency is not None
                else self.max_job_concurrency
            ),
            job_executors=self.job_executors,
            workspace_manager=self.workspace_manager,
        )

    def _create_task_workspace(self, task_execution_id: str) -> str:
        """Create an unambiguous local materialization for one execution.

        Runtime identifiers are orchestration data, not path fragments. Hashing
        the complete `(task_id, execution_id)` tuple prevents separators,
        traversal text, and ambiguous concatenations from selecting another
        execution's directory.
        """
        identity = f"{self.task_id}\x00{task_execution_id}".encode()
        digest = hashlib.sha256(identity).hexdigest()[:24]
        readable_task = re.sub(r"[^A-Za-z0-9._-]+", "-", self.task_id)
        readable_task = readable_task.strip(".-")[:48] or "task"
        leaf = f"{readable_task}-{digest}"

        base = Path(self.base_work_dir).expanduser()
        if base.is_symlink():
            raise ValueError("base_work_dir cannot be a symbolic link")
        base.mkdir(parents=True, exist_ok=True)
        base = base.resolve()
        workspace = base / leaf
        if workspace.is_symlink():
            raise ValueError("task workspace cannot be a symbolic link")
        if workspace.exists() and not workspace.is_dir():
            raise ValueError(f"task workspace is not a directory: {workspace}")
        logger.debug(f"Creating workspace: {workspace}")
        if self.fs_manager:
            self.fs_manager.mkdir(str(workspace), create_parents=True)
        else:
            workspace.mkdir(parents=True, exist_ok=True)
        return str(workspace)

    def resolve_inputs(self, state: ExecutionState) -> dict[str, Any]:
        """Resolve task inputs exclusively from the supplied projection."""
        resolved = dict(self.current_task.initial_input)
        for input_name, ref in self.current_task.input_map.items():
            output = state.dependency_outputs.get(ref.task_id)
            if output is None:
                raise RuntimeError(
                    f"Task {self.current_task.name!r} is not ready: dependency "
                    f"{ref.task_id!r} has not completed."
                )
            value: Any = output.output
            try:
                for part in ref.key.split("."):
                    value = value[part]
            except (KeyError, TypeError) as exc:
                raise KeyError(
                    f"Task {ref.task_id!r} has no output key {ref.key!r}"
                ) from exc
            resolved[input_name] = value
        return resolved

    def get_task_output(self, state: ExecutionState) -> TaskOutput | None:
        """Publish a valid runtime output without consulting correctness."""
        if state.submission is None or state.runtime.status == "surrendered":
            return None
        return TaskOutput(output={"answer": state.submission})

    def get_task_prompt(self, state: ExecutionState) -> str | list[dict]:
        """Return the task prompt computed from an immutable projection."""
        if self.current_task.prompt_fn is not None:
            return self.current_task.prompt_fn(self, state)
        return self._default_task_prompt(state)

    def _default_task_prompt(self, state: ExecutionState) -> str:
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
        resolved = self.resolve_inputs(state)

        prompt += "\nAvailable input data:\n"

        # Display resolved inputs from dependencies
        for input_name, ref in task.input_map.items():
            prompt += f"- {input_name} (from {ref.task_id}): {resolved[input_name]}\n"

        # Display initial input data
        for key, value in task.initial_input.items():
            if key != "work_dir":
                prompt += f"- {key}: {value}\n"

        # Add workspace info
        if self.workspace_path:
            prompt += "\nIMPORTANT: You have access to filesystem tools. All files will be saved in your isolated workspace.\n"

        return prompt

    def _resolve_tools(self, workspace: str | None) -> dict[str, Tool]:
        """Resolve the task's tools — overridable seam over the toolset.

        Subclasses needing bespoke tool policy override this; the default
        delegates to `self.toolset` (named pool + common + workspace tools).
        """
        return self.toolset.resolve(self.current_task, workspace)

    def get_available_tools(self) -> list[dict[str, Any]]:
        """Return tools in OpenAI function-calling format."""
        return [t.get_openai_tool_format() for t in self.tools.values()]

    def tool_catalog_snapshot(self) -> ToolCatalogSnapshot:
        """Capture the complete catalog exposed by a new agent session."""
        return ToolCatalogSnapshot.capture(
            with_submit_answer_tool(self.get_available_tools())
        )

    def validate_state_tool_catalog(self, state: ExecutionState) -> ToolCatalogSnapshot:
        """Refuse to bind `state` when its executable tool catalog has drifted."""
        return validate_tool_catalog_binding(state, self.tool_catalog_snapshot())

    def preprocess_arguments(
        self, tool_name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        """Return provider arguments normalized for the selected tool schema."""
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
                        logger.warning(
                            f"Failed to parse JSON for argument '{key}': {e}. Returning original value."
                        )
                        return value
                else:
                    logger.debug(f"⏭️  Skipping {key} (expected type: {expected_type})")
                    return value
            return value

        return {key: parse_value(key, val) for key, val in args.items()}

    def execute_tool(
        self,
        _state: ExecutionState,
        tool: Tool,
        arguments: dict[str, Any],
    ) -> Any:
        """Execute a tool against values materialized from the supplied projection.

        The default tool contract is stateless. Environments with structured
        domain state may override this hook to decode an explicit environment
        namespace, call the tool, and return a `ToolExecutionResult` carrying
        the complete updated namespace. The Environment must never retain the
        supplied projection or decoded values.
        """
        return tool.execute(**arguments)

    def _resource_lock(self, key: str) -> _ReadWriteLock:
        """Return the shared readers-writer lock guarding one named resource.

        Keyed by `concurrency_key` and shared with this runtime's JobManager
        (which is handed this method as its `key_lock_factory`). The write side
        serialises foreground `CONCURRENT` tool calls and background jobs that
        share `key`, so "two tools writing the same file" never overlap
        regardless of which path runs them; the read side lets same-key
        `CONCURRENT_READ` calls overlap. Different keys always stay parallel.
        Locks are created lazily and cached. A background job calls
        `acquire`/`release` (the write-side aliases), so it is always a
        writer on its resource.
        """
        with self._resource_locks_guard:
            lock = self._resource_locks.get(key)
            if lock is None:
                lock = _ReadWriteLock()
                self._resource_locks[key] = lock
            return lock

    @contextlib.contextmanager
    def execution_guard(self, tool: Tool) -> Iterator[None]:
        """Hold the right lock while `tool` executes, per its concurrency mode.

        * `SERIAL` — the bound runtime's execution lock for the whole call, so the tool
          runs alone (the safe default: with every tool SERIAL the runtime is
          fully serialised exactly as before).
        * `CONCURRENT` with a `concurrency_key` — the resource's **write**
          lock, so it runs exclusively against every reader and writer on that
          key (only same-key work is serialised; other keys overlap).
        * `CONCURRENT_READ` with a `concurrency_key` — the resource's
          **read** lock, so same-key readers overlap each other but still
          exclude (and are excluded by) same-key writers and background jobs.
        * `READ_ONLY` (or either concurrent mode without a key) — no execution
          lock; only the result recording afterwards is serialised.
        """
        mode = getattr(tool, "concurrency", ToolConcurrency.SERIAL)
        key = getattr(tool, "concurrency_key", None)
        if mode == ToolConcurrency.SERIAL:
            with self._execution_lock:
                yield
        elif mode == ToolConcurrency.CONCURRENT and key is not None:
            with self._resource_lock(key).write_lock():
                yield
        elif mode == ToolConcurrency.CONCURRENT_READ and key is not None:
            with self._resource_lock(key).read_lock():
                yield
        else:
            yield

    def _attach_background_tools(self) -> None:
        """Give the bound task execution a JobManager and background tools.

        When the resolved toolset has no background-capable tool the manager is
        left `None`. Otherwise a fresh manager is created for this bound
        task execution so job ids never span independent commit streams.
        """
        has_background = any(
            getattr(t, "background_capable", False) for t in self.tools.values()
        )
        self.shutdown_jobs()
        if not has_background:
            self.job_manager = None
            return
        self.job_manager = JobManager(
            executors=self.job_executors,
            max_concurrency=self.max_job_concurrency,
            # Share the runtime's per-resource locks so a `concurrency_key`
            # serialises background jobs and foreground CONCURRENT tool calls
            # against each other, not just jobs among themselves (Section 9).
            key_lock_factory=self._resource_lock,
        )
        attach_background_tools(self)

    def submit_job(self, tool_name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Submit a background-capable tool as a job and return its handle.

        Receives hidden arguments injected from the projection by `execute_action`
        and resolves the workspace **now** (at submit time), so a later commit can
        never redirect a running job at a different task's workspace. Returns a
        job handle the agent polls with the generated control tools.
        """
        if self.job_manager is None:
            return {"error": "Background jobs are not enabled for this task."}

        tool = self.tools.get(tool_name)
        if tool is None:
            return {"error": f"Tool {tool_name!r} not found."}

        call_args = self.preprocess_arguments(tool_name, arguments)
        hidden_names = list(tool.hidden_args)
        missing = [name for name in hidden_names if name not in call_args]
        if missing:
            return {
                "error": (
                    f"Hidden argument(s) {missing!r} required by tool "
                    f"{tool_name!r} are not configured."
                )
            }
        visible_arguments = {
            name: value for name, value in call_args.items() if name not in hidden_names
        }

        is_valid, error_message = tool.validate_arguments(call_args)
        if not is_valid:
            return {"error": error_message}

        record = self.job_manager.submit(
            tool,
            visible_arguments=visible_arguments,
            call_arguments=call_args,
            hidden_arg_names=tuple(hidden_names),
            workspace=self.workspace_path or "",
            concurrency_key=getattr(tool, "concurrency_key", None),
        )
        return {
            "job_id": record.context.job_id,
            "tool_name": tool_name,
            "status": record.status.value,
            "submitted_at": record.submitted_at.isoformat(),
            "poll_after_seconds": DEFAULT_POLL_AFTER_SECONDS,
        }

    def capture_environment(self, environment: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return the complete explicit environment namespace after a tool."""
        return {
            **dict(environment),
            "jobs": self.job_manager.snapshot() if self.job_manager else {},
        }

    def shutdown_jobs(self) -> None:
        """Cancel outstanding jobs and release the manager's executor threads."""
        if self.job_manager is not None:
            self.job_manager.shutdown()

    def configure(self, state: ExecutionState) -> TaskConfigured:
        """Run the setup hook and return its typed shared-namespace effects."""
        if self.current_task.setup_fn is None:
            status = "No external app/service configuration needed for this task."
            environment = dict(state.environment.values)
        else:
            setup = self.current_task.setup_fn(self, state)
            if setup is None:
                status = "Additional apps/services configured for this task."
                environment = dict(state.environment.values)
            else:
                status = setup.status
                environment = {
                    **dict(state.environment.values),
                    "hidden_arguments": dict(setup.hidden_arguments),
                    "values": dict(setup.values),
                }
        environment = dict(self.capture_environment(environment))
        workspace = self.capture_workspace(state.workspace)
        operations = environment_operations(state.environment.values, environment)
        workspace_delta = (
            None
            if workspace.files == state.workspace.files
            and workspace.artifacts == state.workspace.artifacts
            else WorkspaceDelta.from_workspace(workspace)
        )
        return TaskConfigured(
            status=status,
            environment_operations=operations,
            workspace_delta=workspace_delta,
            expected_environment_revision=(
                state.environment.revision if operations else None
            ),
            expected_workspace_revision=(
                state.workspace.revision if workspace_delta is not None else None
            ),
        )


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

    Grouping is derived, not declared. Dependency outputs are supplied to each
    task's execution-start commit by the runtime, so definitions never share a mutable
    run store.

    Args:
        tasks: Task definitions keyed by task id.
        base_work_dir: Base directory for task workspaces.
        name: Optional benchmark label used for tracing/LaTeX (replaces the old
            `group_id` argument; never used to namespace task ids).
        toolset: Single description of the benchmark's tools (named pool, common
            tools, and the task workspace tool factory). The environment
            resolves each task's concrete tools from it.
        fs_manager: Optional FSManager used to create task workspaces. This is a
            storage backend, not a tool — kept separate from `toolset`.
        env_cls: Environment class to instantiate (escape hatch for stateful
            subclasses such as AFM/wetlab).
        **env_kwargs: Extra keyword arguments forwarded to `env_cls`.

    Returns:
        Environments keyed by task id, ready to serve.
    """
    validate_task_graph(tasks)

    logger.debug("Task Dependencies:")
    for task_id, deps in build_dependency_graph(tasks).items():
        logger.debug(f"- {task_id}: depends on {deps}")

    logger.debug("Task Execution Order:")
    for i, task_id in enumerate(topological_order(tasks)):
        logger.debug(f"{i + 1}. {task_id}")

    environments: dict[str, Environment] = {}
    for component in connected_components(tasks):
        component_tasks = {task_id: tasks[task_id] for task_id in component}
        for task_id in component:
            environments[task_id] = env_cls(
                task_id=task_id,
                task=tasks[task_id],
                base_work_dir=base_work_dir,
                toolset=toolset,
                group_tasks=component_tasks,
                component_id=name,
                fs_manager=fs_manager,
                **env_kwargs,
            )

    return environments
