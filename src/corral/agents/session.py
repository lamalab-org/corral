"""Task session with durable snapshots around every tool call.

Agent-only message and metadata edits may remain in memory while a decision is
being composed. Immediately before a tool executes they are squashed into a
complete immutable State checkpoint. The resulting observation is persisted as
the next checkpoint, producing an append-only, content-addressed State history.
"""

from __future__ import annotations

import json
import secrets
import socket
import threading
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import uuid4

import anyio
import uvicorn
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.lowlevel import Server as MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool
from starlette.applications import Starlette
from starlette.routing import Route

from corral.agents.schema import AgentOutcome, AgentUsage
from corral.core.action import Action
from corral.core.state import RuntimeState, State, UsageState, checkpoint_state
from corral.core.transition import execute_action, propose_action
from corral.core.workspace import WorkspaceState

if TYPE_CHECKING:
    from pydantic import JsonValue

    from corral.agents.hooks import AgentHooks, HookContext, HookPoint
    from corral.core.environment import Environment
    from corral.observability import ObservationContext, Observer
    from corral.persistence import StateStore


_AGENT_STATE_KEY = "agent_state"
_HOOK_STATE_NAMESPACE = "hooks"


@runtime_checkable
class Agent(Protocol):
    """The sole public execution contract implemented by Corral agents.

    A worker may call the same registered agent instance for overlapping task
    activities. Implementations must therefore keep invocation-specific state
    on ``AgentSession`` or in local variables rather than on the agent object.
    """

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        """Run the agent-owned loop against one task-bound session."""
        ...


@dataclass(frozen=True, slots=True)
class ToolResponse:
    """Result returned to an agent after one canonical tool transition."""

    success: bool
    result: str | None
    error: str | None


def _json_value(value: Any) -> JsonValue:
    """Return a detached JSON value suitable for immutable State metadata."""
    return json.loads(json.dumps(value, allow_nan=False, default=str))


def _tool_succeeded(state: State) -> bool:
    message = state.messages[-1]
    metadata = message.get("metadata")
    return bool(isinstance(metadata, Mapping) and metadata.get("success") is True)


def _collapsed_workspace(
    before: WorkspaceState, after: WorkspaceState
) -> WorkspaceState:
    """Represent any number of session-local snapshots as one State fork."""
    if before.files == after.files and before.artifacts == after.artifacts:
        return before
    return WorkspaceState(
        id=before.id,
        revision=before.revision + 1,
        files=after.files,
        artifacts=after.artifacts,
    )


class _SessionMCPTransport:
    """Task-local MCP facade used by black-box harness agents."""

    def __init__(self, interface: AgentSession) -> None:
        self.interface = interface
        # The random path is a per-session capability. Binding to loopback
        # prevents remote access; the token prevents a parallel local task from
        # discovering another task's tools merely by scanning listening ports.
        self._capability = secrets.token_urlsafe(32)
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._socket.bind(("127.0.0.1", 0))
        self.port = int(self._socket.getsockname()[1])

        server: MCPServer = MCPServer(f"corral-session-{interface.execution_id}")

        @server.list_tools()
        async def list_tools() -> list[MCPTool]:
            result: list[MCPTool] = []
            for raw in interface.tools:
                function = raw.get("function", {})
                result.append(
                    MCPTool(
                        name=str(function.get("name", "")),
                        description=str(function.get("description", "")),
                        inputSchema=dict(function.get("parameters", {})),
                    )
                )
            return result

        @server.call_tool()
        async def call_tool(name: str, arguments: dict[str, Any]) -> CallToolResult:
            response = await interface.execute(
                Action(
                    name=name,
                    arguments=arguments or {},
                )
            )
            content = response.result if response.success else response.error
            return CallToolResult(
                content=[TextContent(type="text", text=str(content or ""))],
                isError=not response.success,
            )

        self._manager = StreamableHTTPSessionManager(
            app=server,
            json_response=True,
            stateless=True,
        )
        endpoint = StreamableHTTPASGIApp(self._manager)
        app = Starlette(
            routes=[Route(f"/{self._capability}/mcp", endpoint=endpoint)],
            lifespan=lambda _app: self._manager.run(),
        )
        config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=self.port,
            log_level="warning",
            access_log=False,
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(
            target=self._server.run,
            kwargs={"sockets": [self._socket]},
            name=f"corral-session-mcp-{self.port}",
            daemon=True,
        )

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/{self._capability}/mcp"

    def start(self) -> None:
        self._thread.start()
        deadline = time.monotonic() + 10.0
        while not self._server.started:
            if not self._thread.is_alive():
                raise RuntimeError("task session MCP server failed to start")
            if time.monotonic() >= deadline:
                raise TimeoutError("task session MCP server did not start")
            time.sleep(0.01)

    def close(self) -> None:
        self._server.should_exit = True
        if self._thread.is_alive():
            self._thread.join(timeout=10.0)
        self._socket.close()


@dataclass(frozen=True, slots=True)
class MCPConnection:
    """One task-local MCP endpoint opened for a native SDK harness."""

    url: str


class AgentSession:
    """Agent-facing interface backed by durable tool-boundary State snapshots."""

    def __init__(
        self,
        environment: Environment,
        state: State,
        *,
        execution_id: str | None = None,
        last_score: Mapping[str, Any] | None = None,
        previous_state: State | None = None,
        max_iterations: int | None = None,
        state_store: StateStore | None = None,
        durable_state: State | None = None,
        advance_head: bool = True,
        observer: Observer | None = None,
        observation_context: ObservationContext | None = None,
        hooks: AgentHooks | None = None,
        agent: Agent | None = None,
    ) -> None:
        tool_catalog = environment.validate_state_tool_catalog(state)
        self.environment = environment
        self.execution_id = execution_id or state.id
        self.execution_workspace = environment.workspace_path
        self.observer = observer
        self.observation_context = observation_context
        self._hooks = hooks
        self._hook_agent = agent
        self.previous_state = previous_state
        self.state_store = state_store
        self.advance_head = advance_head
        self._state_lock = threading.RLock()
        self._action_lock = anyio.Lock()
        self._delegated_iteration_limit: ContextVar[int | None] = ContextVar(
            f"corral_delegate_budget_{id(self)}",
            default=None,
        )
        self._mcp_transport: _SessionMCPTransport | None = None
        self._tool_catalog = tool_catalog

        checkpoint = durable_state or state
        if checkpoint.id != state.id:
            raise ValueError("durable State must belong to the same execution")
        self._durable_state = checkpoint

        # Task-local configuration that affects the next agent decision belongs
        # to State.  The session remains the capability boundary for live
        # resources (the Environment and MCP transport), but it does not keep a
        # second, hidden copy of resumable run data.
        runtime_metadata = dict(state.runtime.metadata)
        if max_iterations is not None:
            runtime_metadata["max_iterations"] = max_iterations
        if last_score is not None:
            runtime_metadata["previous_evaluation"] = _json_value(last_score)
        if previous_state is not None:
            runtime_metadata["previous_state_hash"] = previous_state.state_hash
        if runtime_metadata != dict(state.runtime.metadata):
            state = state.fork(
                runtime=RuntimeState(
                    status=state.runtime.status,
                    started_at=state.runtime.started_at,
                    ended_at=state.runtime.ended_at,
                    metadata=runtime_metadata,
                )
            )
        self.initial_state = state
        self.state = state

    @property
    def durable_state(self) -> State:
        """Return the latest successfully persisted session checkpoint."""
        with self._state_lock:
            return self._durable_state

    @property
    def messages(self) -> tuple[Mapping[str, JsonValue], ...]:
        """Return the canonical conversation from the current State."""
        with self._state_lock:
            return self.state.messages

    @property
    def hook_manager(self) -> AgentHooks | None:
        """Return the hook configuration bound to this invocation."""
        return self._hooks

    @property
    def examples(self) -> tuple[Any, ...]:
        """Return task examples stored in durable scaffold metadata."""
        raw = self.state.metadata.scaffold.get("examples") or ()
        if isinstance(raw, list | tuple):
            return tuple(raw)
        return (raw,)

    @property
    def iteration_limit(self) -> int:
        """Return the authoritative or scoped delegated interaction limit."""
        delegated_limit = self._delegated_iteration_limit.get()
        if delegated_limit is not None:
            return delegated_limit
        raw = self.state.runtime.metadata.get("max_iterations")
        if raw is None:
            raise RuntimeError(
                "AgentSession has no max_iterations budget in State; "
                "bind the session through run_agent_session()"
            )
        value = int(raw)
        if value < 1:
            raise ValueError("AgentSession max_iterations must be at least 1")
        return value

    @property
    def prompt(self) -> str | list[dict[str, Any]]:
        """Return the complete prompt for this already-bound task."""
        prompt = self.state.metadata.task.get("prompt")
        if isinstance(prompt, str | list):
            return prompt
        return self.environment.get_task_prompt(self.state)

    @property
    def tools(self) -> tuple[dict[str, Any], ...]:
        """Return only the complete catalog snapshot persisted in State."""
        return self._tool_catalog.detached_tools()

    @property
    def workspace(self) -> str | None:
        """Return the physical workspace bound to this execution, if any."""
        return self.execution_workspace

    @property
    def previous_evaluation(self) -> Mapping[str, Any] | None:
        """Return the previous evaluation recorded in runtime State."""
        previous = self.state.runtime.metadata.get("previous_evaluation")
        return dict(previous) if isinstance(previous, Mapping) else None

    def get_agent_state(
        self,
        namespace: str,
        *,
        previous: bool = False,
    ) -> Mapping[str, JsonValue] | None:
        """Return one agent's detached, serializable runtime namespace.

        Agent objects contain configuration only. Any task- or attempt-specific
        value needed by a later decision belongs under this namespace in the
        immutable Corral State. ``previous=True`` reads the explicitly supplied
        prior State, which is how reflective agents consume an earlier attempt
        without retaining hidden Python-object memory.
        """
        if not namespace:
            raise ValueError("agent-state namespace cannot be empty")
        source = self.previous_state if previous else self.state
        if source is None:
            return None
        raw_namespaces = source.runtime.metadata.get(_AGENT_STATE_KEY)
        if not isinstance(raw_namespaces, Mapping):
            return None
        raw_value = raw_namespaces.get(namespace)
        if not isinstance(raw_value, Mapping):
            return None
        detached = _json_value(raw_value)
        if not isinstance(detached, Mapping):  # pragma: no cover - defensive
            raise TypeError("agent State namespace must be a JSON object")
        return detached

    def set_agent_state(
        self,
        namespace: str,
        value: Mapping[str, Any],
    ) -> None:
        """Replace one namespaced agent payload through an immutable State fork."""
        if not namespace:
            raise ValueError("agent-state namespace cannot be empty")
        detached = _json_value(value)
        if not isinstance(detached, Mapping):
            raise TypeError("agent State namespace must be a JSON object")
        with self._state_lock:
            runtime_metadata = dict(self.state.runtime.metadata)
            raw_namespaces = runtime_metadata.get(_AGENT_STATE_KEY)
            namespaces = (
                dict(raw_namespaces) if isinstance(raw_namespaces, Mapping) else {}
            )
            namespaces[namespace] = detached
            runtime_metadata[_AGENT_STATE_KEY] = namespaces
            self.state = self.state.fork(
                runtime=RuntimeState(
                    status=self.state.runtime.status,
                    started_at=self.state.runtime.started_at,
                    ended_at=self.state.runtime.ended_at,
                    metadata=runtime_metadata,
                )
            )

    @property
    def surrender_allowed(self) -> bool:
        return bool(self.state.metadata.scaffold.get("enable_surrender", False))

    @property
    def submission(self) -> str | None:
        """Return the answer accepted through ``submit_answer``, if any."""
        return self.state.submission

    @property
    def submission_status(self) -> str | None:
        """Return the terminal status created by submission, if any."""
        return self.state.runtime.status if self.state.submission is not None else None

    async def record_message(self, raw: Mapping[str, Any]) -> None:
        """Append one provider message to the canonical State conversation."""
        message = dict(raw)
        if message.get("role") == "tool" or message.get("tool_calls"):
            return
        if message.get("content") is None and not message.get("name"):
            return
        detached = _json_value(message)
        if not isinstance(detached, Mapping):
            raise TypeError("a provider transcript message must be a JSON object")
        with self._state_lock:
            self.state = self.state.fork(messages=(*self.state.messages, detached))

    async def run_hooks(
        self,
        hook_point: HookPoint | str,
        *,
        agent: Agent | None = None,
        hooks: AgentHooks | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> HookContext:
        """Run one lifecycle point against this session's canonical State.

        Hook callbacks receive this ``AgentSession`` directly. Their metadata is
        stored under the session's durable agent-state namespace so concurrent
        invocations never leak hook state through the shared agent object.
        """
        from corral.agents.hooks import (
            AgentHooks,
            HookContext,
            HookPoint,
        )

        point = (
            hook_point if isinstance(hook_point, HookPoint) else HookPoint(hook_point)
        )
        selected_agent = agent or self._hook_agent
        if selected_agent is None:
            raise ValueError("running session hooks requires an agent")
        selected_hooks = self._hooks if hooks is None else hooks
        if selected_hooks is not None and not isinstance(selected_hooks, AgentHooks):
            raise TypeError("agent hooks must be an AgentHooks instance")

        stored = self.get_agent_state(_HOOK_STATE_NAMESPACE)
        raw_metadata = stored.get("metadata") if stored is not None else None
        metadata = dict(raw_metadata) if isinstance(raw_metadata, Mapping) else {}
        context = HookContext(
            session=self,
            agent=selected_agent,
            hook_point=point,
            data=dict(data or {}),
            metadata=metadata,
        )
        if selected_hooks is None or not selected_hooks.has_hooks(point):
            return context

        previous_metadata = dict(metadata)
        await selected_hooks.run(point, context)
        if context.metadata != previous_metadata:
            self.set_agent_state(
                _HOOK_STATE_NAMESPACE,
                {
                    "schema_version": 1,
                    "metadata": context.metadata,
                },
            )
        return context

    @asynccontextmanager
    async def open_mcp(self) -> AsyncIterator[MCPConnection]:
        """Open the task-local MCP endpoint for one native harness lifecycle."""
        with self._state_lock:
            if self._mcp_transport is not None:
                raise RuntimeError(
                    "this agent session already has an open MCP endpoint"
                )
            transport = _SessionMCPTransport(self)
            self._mcp_transport = transport
        try:
            await anyio.to_thread.run_sync(transport.start)
            yield MCPConnection(url=transport.url)
        finally:
            await anyio.to_thread.run_sync(transport.close)
            with self._state_lock:
                if self._mcp_transport is transport:
                    self._mcp_transport = None

    async def _commit_action_snapshot(
        self,
        source: State,
        *,
        action: Action,
        phase: str,
    ) -> State:
        """Persist one complete before/after snapshot and advance its branch."""
        with self._state_lock:
            parent = self._durable_state
        snapshot = checkpoint_state(parent, source)
        if self.state_store is None:
            committed = snapshot
        elif self.observer is None or self.observation_context is None:
            committed = await self.state_store.save(
                snapshot,
                transition_id=f"action:{action.id}:{phase}",
                advance_head=self.advance_head,
            )
        else:
            from corral.observability import (
                Observation,
                observe_safely,
                update_safely,
            )

            with observe_safely(
                self.observer,
                Observation(
                    name="state.commit",
                    context=self.observation_context,
                    state_before=parent,
                    action=action,
                    metadata={"boundary": f"tool-{phase}"},
                ),
            ) as commit_span:
                committed = await self.state_store.save(
                    snapshot,
                    transition_id=f"action:{action.id}:{phase}",
                    advance_head=self.advance_head,
                )
                update_safely(commit_span, state_after=committed)

        with self._state_lock:
            self._durable_state = committed
            self.state = committed
        return committed

    async def _execute_proposed(
        self,
        proposed: State,
        action: Action,
    ) -> ToolResponse:
        """Execute an already durable pending action and persist its observation."""
        with self._state_lock:
            self.state = proposed

        if self.observer is None or self.observation_context is None:
            observed = await anyio.to_thread.run_sync(
                lambda: execute_action(self.environment, proposed, action)
            )
        else:
            from corral.observability import (
                Observation,
                observe_safely,
                update_safely,
            )

            with observe_safely(
                self.observer,
                Observation(
                    name=f"tool.{action.name}",
                    context=self.observation_context,
                    as_type="tool",
                    state_before=proposed,
                    action=action,
                ),
            ) as tool_span:
                observed = await anyio.to_thread.run_sync(
                    lambda: execute_action(self.environment, proposed, action)
                )
                update_safely(tool_span, state_after=observed)

        # Harness callbacks may append transcript-only messages while a tool is
        # running. Preserve them in the after snapshot instead of overwriting
        # the newer in-memory value with the raw observation fork.
        with self._state_lock:
            current = self.state
        if current != proposed:
            observed = current.fork(
                messages=(*current.messages, observed.messages[-1]),
                environment=observed.environment,
                workspace=observed.workspace,
                usage=observed.usage,
                runtime=observed.runtime,
                dependency_outputs=observed.dependency_outputs,
            )

        observed = await self._commit_action_snapshot(
            observed,
            action=action,
            phase="after",
        )
        content = str(observed.messages[-1].get("content", ""))
        success = _tool_succeeded(observed)
        return ToolResponse(
            success=success,
            result=content if success else None,
            error=None if success else content,
        )

    async def execute(self, action: Action) -> ToolResponse:
        """Persist, execute, and persist one explicit session action."""
        async with self._action_lock:
            with self._state_lock:
                if self.state.is_terminal:
                    return ToolResponse(
                        success=False,
                        result=None,
                        error=(
                            "the session is already terminal; no further actions "
                            "can run after submit_answer succeeds"
                        ),
                    )
                actor = self.initial_state.runtime.metadata.get("actor_id")
                trusted = Action(
                    id=action.id,
                    name=action.name,
                    arguments=action.arguments,
                    actor_id=str(actor) if actor is not None else None,
                    content=action.content,
                    metadata=action.metadata,
                )
                proposed = propose_action(self.state, trusted)
            proposed = await self._commit_action_snapshot(
                proposed,
                action=trusted,
                phase="before",
            )
            return await self._execute_proposed(proposed, trusted)

    async def run_delegate(
        self,
        agent: Agent,
        *,
        max_iterations: int | None = None,
    ) -> AgentOutcome:
        """Run a delegated agent through the canonical bound-session lifecycle.

        Composite agents use this method instead of calling a child's
        ``run_session`` method directly.  The child therefore receives the same
        hook lifecycle, typed-outcome validation, and authoritative
        ``submit_answer`` enforcement as a top-level agent, while continuing on
        this exact State branch. ``max_iterations`` scopes the child to the
        outer agent's remaining task budget without changing the durable task
        configuration.

        Usage is deliberately not folded into :class:`UsageState` here.  The
        composite agent remains responsible for combining its own and its
        delegate's :class:`AgentUsage`; the outer ``run_agent_session`` call then
        records that combined usage exactly once.
        """
        if not isinstance(agent, Agent):
            raise TypeError("a delegate must implement run_session(AgentSession)")
        if max_iterations is not None and max_iterations < 1:
            raise ValueError("a delegate max_iterations budget must be at least 1")
        delegated_limit = (
            self.iteration_limit if max_iterations is None else max_iterations
        )
        token = self._delegated_iteration_limit.set(delegated_limit)
        try:
            return await _run_bound_agent(agent, self)
        finally:
            self._delegated_iteration_limit.reset(token)

    def fork_branch(self, *, execution_id: str | None = None) -> AgentSession:
        """Create an isolated speculative branch through the session protocol.

        The branch starts from the caller's current State, persists its action
        snapshots without advancing the canonical execution head, and inherits
        the observer and hook configuration of the owning session.  Its
        Environment receives a distinct execution identity so mutable workspace
        and background-job resources remain isolated.
        """
        branch_execution_id = execution_id or (f"{self.execution_id}-branch-{uuid4()}")
        environment = self.environment.for_task(branch_execution_id)
        environment.prepare_workspace(self.state.workspace)
        return AgentSession(
            environment,
            self.state,
            execution_id=branch_execution_id,
            previous_state=self.previous_state,
            state_store=self.state_store,
            durable_state=self.durable_state,
            advance_head=False,
            observer=self.observer,
            observation_context=self.observation_context,
            hooks=self._hooks,
            agent=self._hook_agent,
        )

    def adopt_branch(self, branch: AgentSession) -> None:
        """Adopt the current State of a compatible speculative branch.

        Adoption changes only the in-memory candidate State.  The owning
        session keeps its canonical durable parent so the next action creates a
        normal checkpoint on the main execution path.
        """
        state = branch.state
        if state.id != self.initial_state.id:
            raise ValueError("a session can adopt only a branch of its own State")
        initial_messages = self.initial_state.messages
        if tuple(state.messages[: len(initial_messages)]) != initial_messages:
            raise ValueError("the adopted branch does not descend from the session")
        with self._state_lock:
            self.state = state

    async def resume_pending_action(self) -> ToolResponse:
        """Execute the exact pending action restored from the durable head."""
        async with self._action_lock:
            with self._state_lock:
                action = self.state.pending_action
                proposed = self.state
            if action is None:
                raise ValueError("the session State has no pending action to resume")
            return await self._execute_proposed(proposed, action)

    def close(self) -> None:
        transport = self._mcp_transport
        if transport is not None:
            transport.close()
            self._mcp_transport = None

    def final_messages(self) -> tuple[Mapping[str, JsonValue], ...]:
        """Return messages added after the session's state-backed starting point."""
        with self._state_lock:
            return self.state.messages[len(self.initial_state.messages) :]


@dataclass(frozen=True, slots=True)
class AgentSessionOutcome:
    """Result of one complete in-memory agent session."""

    outcome: AgentOutcome
    state: State
    durable_state: State
    messages: tuple[Mapping[str, JsonValue], ...]
    environment: Mapping[str, JsonValue]
    workspace: WorkspaceState
    usage: UsageState
    runtime: RuntimeState


def _session_usage(
    before: State,
    interface: AgentSession,
    result: AgentOutcome,
) -> UsageState:
    return UsageState(
        input_tokens=before.usage.input_tokens + result.usage.input_tokens,
        output_tokens=before.usage.output_tokens + result.usage.output_tokens,
        llm_calls=before.usage.llm_calls + result.usage.llm_calls,
        tool_calls=interface.state.usage.tool_calls,
        agent_steps=before.usage.agent_steps + 1,
        metadata={
            **dict(before.usage.metadata),
            **_json_value(result.usage.metadata),
        },
    )


def _normalize_outcome(result: AgentOutcome) -> AgentOutcome:
    """Detach arbitrary provider objects before they reach immutable State."""
    usage_metadata = _json_value(result.usage.metadata)
    metadata = _json_value(result.metadata)
    if not isinstance(usage_metadata, Mapping) or not isinstance(metadata, Mapping):
        raise TypeError("session outcome metadata must be JSON objects")
    return AgentOutcome(
        status=result.status,
        answer=result.answer,
        error=result.error,
        usage=AgentUsage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            llm_calls=result.usage.llm_calls,
            metadata=usage_metadata,
        ),
        metadata=metadata,
    )


def _enforce_tool_submission(
    result: AgentOutcome,
    interface: AgentSession,
) -> AgentOutcome:
    """Make the canonical ``submit_answer`` transition authoritative.

    A session agent may report completion only after executing the tool.  This
    prevents a plain-text or adapter-returned answer from bypassing the same
    permission checks, trace record, and terminal transition used by every
    other agent.
    """
    submission = interface.state.submission
    if submission is None:
        if result.status not in {"completed", "surrendered"}:
            return result
        return AgentOutcome(
            status="protocol_failure",
            error=(f"agent returned {result.status!r} without calling submit_answer"),
            usage=result.usage,
            metadata=result.metadata,
        )

    expected_status = (
        "surrendered"
        if interface.state.runtime.status == "surrendered"
        else "completed"
    )
    expected_answer = None if expected_status == "surrendered" else submission
    if result.status == expected_status and result.answer == expected_answer:
        return result

    # The successfully executed terminal tool is the source of truth. Preserve
    # the adapter discrepancy as provenance instead of discarding a valid,
    # permission-checked submission or leaving State terminal but Outcome failed.
    return AgentOutcome(
        status=expected_status,
        answer=expected_answer,
        usage=result.usage,
        metadata={
            **dict(result.metadata),
            "reported_outcome": {
                "status": result.status,
                "answer": result.answer,
                "error": result.error,
            },
        },
    )


async def _run_bound_agent(
    agent: Agent,
    interface: AgentSession,
) -> AgentOutcome:
    """Run one agent against an existing session through the shared protocol."""
    from corral.agents.hooks import AgentHooks, HookPoint

    raw_hooks = getattr(agent, "hooks", None)
    if raw_hooks is None:
        hooks = AgentHooks()
    elif isinstance(raw_hooks, AgentHooks):
        hooks = raw_hooks
    else:
        raise TypeError("agent hooks must be an AgentHooks instance")

    try:
        before_task = await interface.run_hooks(
            HookPoint.BEFORE_TASK,
            agent=agent,
            hooks=hooks,
        )
        if interface.submission is not None:
            result = AgentOutcome(
                status=(
                    "surrendered"
                    if interface.submission_status == "surrendered"
                    else "completed"
                ),
                answer=(
                    None
                    if interface.submission_status == "surrendered"
                    else interface.submission
                ),
            )
        elif not before_task.should_continue:
            result = AgentOutcome(
                status="cancelled",
                error="agent session cancelled by a before-task hook",
            )
        else:
            result = await agent.run_session(interface)
        if not isinstance(result, AgentOutcome):
            raise TypeError("Agent.run_session() must return AgentOutcome")
        result = _normalize_outcome(result)
        result = _enforce_tool_submission(result, interface)
    except BaseException as exc:
        await interface.run_hooks(
            HookPoint.AFTER_TASK,
            agent=agent,
            hooks=hooks,
            data={
                "status": "raised",
                "answer": None,
                "error": f"{type(exc).__name__}: {exc}",
            },
        )
        raise
    await interface.run_hooks(
        HookPoint.AFTER_TASK,
        agent=agent,
        hooks=hooks,
        data={
            "status": result.status,
            "answer": result.answer,
            "error": result.error,
        },
    )
    return result


async def run_agent_session(
    agent: Agent,
    environment: Environment,
    state: State,
    *,
    last_score: Mapping[str, Any] | None = None,
    previous_state: State | None = None,
    max_iterations: int,
    state_store: StateStore | None = None,
    durable_state: State | None = None,
    advance_head: bool = True,
    observer: Observer | None = None,
    observation_context: ObservationContext | None = None,
) -> AgentSessionOutcome:
    """Run one first-class session agent and collect its canonical effects."""
    environment.prepare_workspace(state.workspace)
    interface = AgentSession(
        environment,
        state,
        last_score=last_score,
        previous_state=previous_state,
        max_iterations=max_iterations,
        state_store=state_store,
        durable_state=durable_state,
        advance_head=advance_head,
        observer=observer,
        observation_context=observation_context,
        hooks=getattr(agent, "hooks", None),
        agent=agent,
    )
    try:
        result = await _run_bound_agent(agent, interface)
        usage = _session_usage(state, interface, result)
        completed_state = interface.state
        if completed_state.usage != usage:
            completed_state = completed_state.fork(usage=usage)
        return AgentSessionOutcome(
            outcome=result,
            state=completed_state,
            durable_state=interface.durable_state,
            messages=interface.final_messages(),
            environment=_json_value(interface.state.environment),
            workspace=_collapsed_workspace(state.workspace, interface.state.workspace),
            usage=usage,
            runtime=interface.state.runtime,
        )
    finally:
        interface.close()


__all__ = [
    "Agent",
    "AgentSession",
    "AgentSessionOutcome",
    "MCPConnection",
    "ToolResponse",
    "run_agent_session",
]
