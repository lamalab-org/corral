"""Agent sessions bound to trusted authors and filtered execution contexts."""

from __future__ import annotations

import asyncio
import json
import secrets
import socket
import threading
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import asynccontextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable
from uuid import NAMESPACE_URL, uuid4, uuid5

import anyio
import uvicorn
from mcp.server.fastmcp.server import StreamableHTTPASGIApp
from mcp.server.lowlevel import Server as MCPServer
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.types import CallToolResult, TextContent
from mcp.types import Tool as MCPTool
from starlette.applications import Starlette
from starlette.routing import Route

from corral.agents.hooks import AgentHooks, HookContext, HookPoint
from corral.agents.schema import AgentOutcome, AgentUsage
from corral.agents.usage import usage_from_mapping
from corral.core.action import Action
from corral.core.actors import ActorRef
from corral.core.commit import Commit, CommitRequest
from corral.core.context import AgentContext, AgentContextResolver
from corral.core.events import (
    AgentCompleted,
    AgentSpawned,
    AgentStarted,
    AgentStateUpdated,
    AgentTurnRecorded,
    ContextImported,
    ParallelGroupCompleted,
    SubmissionAccepted,
    ToolCompleted,
    ToolFailed,
    ToolStarted,
    UsageDelta,
)
from corral.core.state import ExecutionState, UsageState
from corral.core.transition import ToolEffects, execute_action, propose_action
from corral.observability import record_commit_safely
from corral.persistence import CommitConflictError

if TYPE_CHECKING:
    from pydantic import JsonValue

    from corral.core.environment import Environment
    from corral.observability import ObservationContext, Observer
    from corral.persistence import CommitStore

_HOOK_STATE_NAMESPACE = "hooks"


@runtime_checkable
class Agent(Protocol):
    async def run_session(self, session: AgentSession) -> AgentOutcome: ...


INSPECT_SUBAGENT_TOOL_NAME = "inspect_subagent"


@dataclass(frozen=True, slots=True)
class AgentSessionCapabilities:
    """Opt-in model-facing capabilities supplied by the session runtime."""

    inspect_subagents: bool = False

    def to_metadata(self) -> dict[str, bool]:
        return {"inspect_subagents": self.inspect_subagents}

    @classmethod
    def from_value(cls, value: object) -> AgentSessionCapabilities:
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if isinstance(value, Mapping):
            unknown = set(value) - {"inspect_subagents"}
            if unknown:
                raise ValueError(
                    f"unknown AgentSession capabilities: {sorted(unknown)}"
                )
            enabled = value.get("inspect_subagents", False)
            if not isinstance(enabled, bool):
                raise TypeError("inspect_subagents capability must be a boolean")
            return cls(inspect_subagents=enabled)
        raise TypeError(
            "session_capabilities must be AgentSessionCapabilities or a mapping"
        )


def agent_session_capabilities(agent: Agent | None) -> AgentSessionCapabilities:
    """Resolve the declarative capabilities requested by an agent class."""
    return AgentSessionCapabilities.from_value(
        None if agent is None else getattr(agent, "session_capabilities", None)
    )


def inspect_subagent_tool() -> dict[str, Any]:
    """Return the model-facing schema for authorized child-run inspection."""
    return {
        "type": "function",
        "function": {
            "name": INSPECT_SUBAGENT_TOOL_NAME,
            "description": (
                "Inspect the private context and execution state of a child or "
                "descendant subagent that this agent is authorized to inspect. "
                "Use this only when the normal subagent result needs investigation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "child_run_id": {
                        "type": "string",
                        "description": "The child run identifier returned by delegation.",
                    },
                    "include_commits": {
                        "type": "boolean",
                        "default": False,
                        "description": "Include recent authored commit records.",
                    },
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 100,
                        "default": 25,
                        "description": (
                            "Maximum recent messages, actions, invocations, and "
                            "optional commits to return."
                        ),
                    },
                },
                "required": ["child_run_id"],
                "additionalProperties": False,
            },
        },
    }


@dataclass(frozen=True, slots=True)
class ToolResponse:
    success: bool
    result: str | None
    error: str | None


@dataclass(frozen=True, slots=True)
class SubagentTrace:
    context: AgentContext
    commits: tuple[Commit, ...]


def _json_value(value: Any) -> JsonValue:
    return json.loads(json.dumps(value, allow_nan=False, default=str))


class _SessionMCPTransport:
    """Task-local MCP facade used by black-box harness agents."""

    def __init__(self, interface: AgentSession) -> None:
        self.interface = interface
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
                Action(name=name, arguments=arguments or {})
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
    url: str


class AgentSession:
    """Authorized view and append capability for one concrete agent run."""

    def __init__(
        self,
        environment: Environment,
        state: ExecutionState,
        *,
        actor: ActorRef,
        state_store: CommitStore,
        branch_id: str = "main",
        runtime_actor: ActorRef | None = None,
        last_score: Mapping[str, Any] | None = None,
        previous_state: ExecutionState | None = None,
        max_iterations: int | None = None,
        observer: Observer | None = None,
        observation_context: ObservationContext | None = None,
        hooks: AgentHooks | None = None,
        agent: Agent | None = None,
        require_submission: bool = True,
        capabilities: AgentSessionCapabilities | Mapping[str, Any] | None = None,
    ) -> None:
        if actor.kind != "agent":
            raise ValueError("AgentSession must be bound to an agent ActorRef")
        if state.execution_id != (state_store.execution_id or state.execution_id):
            raise ValueError("session projection and commit store execution differ")
        self.environment = environment
        self.execution_id = state.execution_id
        self.execution_workspace = environment.workspace_path
        self.actor = actor
        self.runtime_actor = runtime_actor or ActorRef(
            kind="runtime",
            actor_id="corral",
            run_id=f"runtime:{self.execution_id}",
        )
        self.state_store = state_store
        self.branch_id = branch_id
        self._bound_store = state_store.bind(
            actor, branch_id=branch_id, execution_id=self.execution_id
        )
        self.observer = observer
        self.observation_context = observation_context
        self._hooks = hooks
        self._hook_agent = agent
        self._require_submission = require_submission
        requested_capabilities = (
            agent_session_capabilities(agent)
            if capabilities is None
            else AgentSessionCapabilities.from_value(capabilities)
        )
        projected_run = state.agent_runs.get(actor.run_id)
        projected_capabilities = (
            None
            if projected_run is None
            else projected_run.metadata.get("session_capabilities")
        )
        if projected_capabilities is None:
            self.capabilities = requested_capabilities
        else:
            persisted_capabilities = AgentSessionCapabilities.from_value(
                projected_capabilities
            )
            if (
                capabilities is not None or agent is not None
            ) and persisted_capabilities != requested_capabilities:
                raise ValueError(
                    "session capabilities differ from the registered agent run"
                )
            self.capabilities = persisted_capabilities
        self.previous_state = previous_state
        self._last_score = _json_value(last_score) if last_score is not None else None
        self._max_iterations = max_iterations
        self._delegated_iteration_limit: ContextVar[int | None] = ContextVar(
            f"corral_delegate_budget_{id(self)}", default=None
        )
        self._action_lock = anyio.Lock()
        self._mcp_transport: _SessionMCPTransport | None = None
        self._resolver = AgentContextResolver()
        self.state = state
        self._last_observed_hash = state.through_commit_hash
        self._context = self._resolver.for_agent(state, actor.run_id)
        self._initial_message_count = len(self._context.messages)
        self._tool_catalog = environment.validate_state_tool_catalog(state)
        self._subagent_tasks: dict[str, asyncio.Task[AgentOutcome]] = {}
        self._subagent_outcomes: dict[str, AgentOutcome] = {}

    async def _notify(self, commit: Commit) -> None:
        if self.observer is None:
            return

        record_commit_safely(
            self.observer,
            commit,
            context=self.observation_context,
        )

    async def _refresh(self, *, observed_hash: str | None = None) -> ExecutionState:
        self.state = await self.state_store.materialize(self.branch_id)
        if observed_hash is not None:
            self._last_observed_hash = observed_hash
        self._context = self._resolver.for_agent(self.state, self.actor.run_id)
        return self.state

    async def _append_agent(self, event: Any, request_id: str) -> Commit:
        commit = await self._bound_store.append(
            CommitRequest(
                request_id=request_id,
                execution_id=self.execution_id,
                branch_id=self.branch_id,
                based_on_hash=self._last_observed_hash,
                author=self.actor,
                event=event,
            )
        )
        await self._notify(commit)
        await self._refresh(observed_hash=commit.hash)
        return commit

    async def _append_runtime(
        self,
        event: Any,
        request_id: str,
        *,
        based_on_hash: str | None = None,
    ) -> Commit:
        commit = await self.state_store.bind(
            self.runtime_actor,
            branch_id=self.branch_id,
            execution_id=self.execution_id,
        ).append(
            CommitRequest(
                request_id=request_id,
                execution_id=self.execution_id,
                branch_id=self.branch_id,
                based_on_hash=based_on_hash or self._last_observed_hash,
                author=self.runtime_actor,
                event=event,
            )
        )
        await self._notify(commit)
        return commit

    @property
    def context(self) -> AgentContext:
        return self._context

    @property
    def messages(self) -> tuple[Mapping[str, JsonValue], ...]:
        return self._context.messages

    @property
    def hook_manager(self) -> AgentHooks | None:
        return self._hooks

    @property
    def examples(self) -> tuple[Any, ...]:
        raw = self.state.task.scaffold.get("examples") or ()
        return tuple(raw) if isinstance(raw, list | tuple) else (raw,)

    @property
    def iteration_limit(self) -> int:
        delegated = self._delegated_iteration_limit.get()
        if delegated is not None:
            return delegated
        if self._max_iterations is None:
            raise RuntimeError("AgentSession has no max_iterations budget")
        if self._max_iterations < 1:
            raise ValueError("AgentSession max_iterations must be at least 1")
        return self._max_iterations

    @property
    def prompt(self) -> str | list[dict[str, Any]]:
        prompt = self.state.task.metadata.get("prompt")
        resolved = (
            prompt
            if isinstance(prompt, str | list)
            else self.environment.get_task_prompt(self.state)
        )
        handoff = self._context.agent_run.handoff
        if handoff is None:
            return resolved
        handoff_text = handoff if isinstance(handoff, str) else json.dumps(handoff)
        if isinstance(resolved, str):
            return f"{resolved}\n\nSubagent handoff:\n{handoff_text}"
        return [
            *resolved,
            {"role": "user", "content": f"Subagent handoff:\n{handoff_text}"},
        ]

    @property
    def task_id(self) -> str:
        value = self.state.task.metadata.get("id")
        return value if isinstance(value, str) and value else self.execution_id

    @property
    def tools(self) -> tuple[dict[str, Any], ...]:
        tools = list(self._tool_catalog.detached_tools())
        if self.capabilities.inspect_subagents:
            names = {str(tool.get("function", {}).get("name", "")) for tool in tools}
            if INSPECT_SUBAGENT_TOOL_NAME in names:
                raise ValueError(
                    f"{INSPECT_SUBAGENT_TOOL_NAME!r} is reserved for AgentSession"
                )
            tools.append(inspect_subagent_tool())
        return tuple(tools)

    @property
    def workspace(self) -> str | None:
        return self.execution_workspace

    @property
    def previous_evaluation(self) -> Mapping[str, Any] | None:
        return dict(self._last_score) if isinstance(self._last_score, Mapping) else None

    def _run_state(
        self, state: ExecutionState, *, previous: bool
    ) -> Mapping[str, Mapping[str, JsonValue]] | None:
        run = state.agent_runs.get(self.actor.run_id)
        if run is None and previous:
            run = next(
                (
                    candidate
                    for candidate in state.agent_runs.values()
                    if candidate.actor_id == self.actor.actor_id
                ),
                None,
            )
        return None if run is None else run.algorithm_state

    def get_agent_state(
        self, namespace: str, *, previous: bool = False
    ) -> Mapping[str, JsonValue] | None:
        if not namespace:
            raise ValueError("agent-state namespace cannot be empty")
        source = self.previous_state if previous else self.state
        if source is None:
            return None
        namespaces = self._run_state(source, previous=previous)
        value = None if namespaces is None else namespaces.get(namespace)
        return dict(value) if isinstance(value, Mapping) else None

    async def set_agent_state(self, namespace: str, value: Mapping[str, Any]) -> None:
        if not namespace:
            raise ValueError("agent-state namespace cannot be empty")
        detached = _json_value(value)
        if not isinstance(detached, Mapping):
            raise TypeError("agent-state namespace must be a JSON object")
        await self._append_agent(
            AgentStateUpdated(namespace=namespace, value=detached),
            f"agent:{self.actor.run_id}:state:{namespace}:{uuid4()}",
        )

    @property
    def surrender_allowed(self) -> bool:
        return bool(self.state.task.scaffold.get("enable_surrender", False))

    @property
    def submission(self) -> str | None:
        return self.state.submission

    @property
    def submission_status(self) -> str | None:
        return self.state.runtime.status if self.state.submission is not None else None

    async def record_messages(
        self,
        raw_messages: Sequence[Mapping[str, Any]],
        *,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        """Commit exactly the conversation messages added by one turn."""
        messages: list[Mapping[str, JsonValue]] = []
        for raw in raw_messages:
            message = dict(raw)
            if message.get("role") == "tool" or message.get("tool_calls"):
                continue
            if message.get("content") is None and not message.get("name"):
                continue
            detached = _json_value(message)
            if not isinstance(detached, Mapping):
                raise TypeError("a provider transcript message must be a JSON object")
            messages.append(detached)
        if not messages:
            return
        normalized = usage_from_mapping(
            usage,
            llm_calls=1 if usage is not None else 0,
        )
        await self._append_agent(
            AgentTurnRecorded(
                messages=tuple(messages),
                usage_delta=UsageDelta(
                    input_tokens=normalized.input_tokens,
                    output_tokens=normalized.output_tokens,
                    reasoning_tokens=normalized.reasoning_tokens,
                    llm_calls=normalized.llm_calls,
                ),
            ),
            f"agent:{self.actor.run_id}:turn:{uuid4()}",
        )

    async def record_message(
        self,
        raw: Mapping[str, Any],
        *,
        usage: Mapping[str, Any] | None = None,
    ) -> None:
        await self.record_messages((raw,), usage=usage)

    async def run_hooks(
        self,
        hook_point: HookPoint | str,
        *,
        agent: Agent | None = None,
        hooks: AgentHooks | None = None,
        data: Mapping[str, Any] | None = None,
    ) -> HookContext:
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
            await self.set_agent_state(
                _HOOK_STATE_NAMESPACE,
                {"schema_version": 1, "metadata": context.metadata},
            )
        return context

    @asynccontextmanager
    async def open_mcp(self) -> AsyncIterator[MCPConnection]:
        if self._mcp_transport is not None:
            raise RuntimeError("this agent session already has an open MCP endpoint")
        transport = _SessionMCPTransport(self)
        self._mcp_transport = transport
        try:
            await anyio.to_thread.run_sync(transport.start)
            yield MCPConnection(url=transport.url)
        finally:
            await anyio.to_thread.run_sync(transport.close)
            if self._mcp_transport is transport:
                self._mcp_transport = None

    async def _inspect_subagent_effects(
        self, arguments: Mapping[str, JsonValue]
    ) -> ToolEffects:
        if not self.capabilities.inspect_subagents:
            return ToolEffects(
                observation="inspect_subagent is not enabled for this agent",
                status="invalid_tool",
            )
        unknown = set(arguments) - {"child_run_id", "include_commits", "limit"}
        child_run_id = arguments.get("child_run_id")
        include_commits = arguments.get("include_commits", False)
        limit = arguments.get("limit", 25)
        if unknown:
            return ToolEffects(
                observation=f"unexpected argument(s): {', '.join(sorted(unknown))}",
                status="invalid_args",
            )
        if not isinstance(child_run_id, str) or not child_run_id:
            return ToolEffects(
                observation="child_run_id must be a non-empty string",
                status="invalid_args",
            )
        if not isinstance(include_commits, bool):
            return ToolEffects(
                observation="include_commits must be a boolean",
                status="invalid_args",
            )
        if (
            isinstance(limit, bool)
            or not isinstance(limit, int)
            or not 1 <= limit <= 100
        ):
            return ToolEffects(
                observation="limit must be an integer between 1 and 100",
                status="invalid_args",
            )
        try:
            trace = await self.inspect_subagent(child_run_id)
        except (PermissionError, ValueError) as exc:
            return ToolEffects(observation=str(exc), status="execution_error")

        context = trace.context
        messages = list(context.messages)
        actions = list(context.actions)
        invocations = list(context.tool_invocations)
        observation: dict[str, JsonValue] = {
            "child_run_id": child_run_id,
            "through_commit_hash": context.through_commit_hash,
            "agent_run": context.agent_run.model_dump(mode="json"),
            "messages": [dict(message) for message in messages[-limit:]],
            "actions": [action.model_dump(mode="json") for action in actions[-limit:]],
            "tool_invocations": [
                invocation.model_dump(mode="json")
                for invocation in invocations[-limit:]
            ],
            "available": {
                "messages": len(messages),
                "actions": len(actions),
                "tool_invocations": len(invocations),
                "commits": len(trace.commits),
            },
        }
        if include_commits:
            observation["commits"] = [
                commit.model_dump(mode="json") for commit in trace.commits[-limit:]
            ]
        return ToolEffects(observation=observation, status="success")

    async def _run_tool(
        self,
        action: Action,
        *,
        proposal_hash: str,
    ) -> ToolResponse:
        invocation_id = str(
            uuid5(NAMESPACE_URL, f"corral:tool:{self.execution_id}:{action.id}")
        )
        current = await self.state_store.materialize(self.branch_id)
        existing = current.tool_invocations.get(invocation_id)
        if existing is not None and existing.status in {"completed", "failed"}:
            content = (
                existing.observation
                if existing.status == "completed"
                else existing.error
            )
            rendered = content if isinstance(content, str) else json.dumps(content)
            return ToolResponse(
                success=existing.status == "completed",
                result=rendered if existing.status == "completed" else None,
                error=rendered if existing.status == "failed" else None,
            )
        if existing is None:
            started = await self._append_runtime(
                ToolStarted(
                    action_id=action.id,
                    invocation_id=invocation_id,
                    requested_by_run_id=self.actor.run_id,
                    tool_name=action.name,
                ),
                f"tool:{invocation_id}:started",
                based_on_hash=proposal_hash,
            )
            based_on = started.hash
        else:
            based_on = existing.started_commit_hash

        execution_state = await self.state_store.materialize(self.branch_id, based_on)
        trusted = execution_state.actions[action.id].action
        tool_actor = ActorRef(
            kind="tool",
            actor_id=trusted.name,
            run_id=invocation_id,
        )

        async def fail(error: Exception) -> ToolResponse:
            failed = await self.state_store.bind(
                tool_actor,
                branch_id=self.branch_id,
                execution_id=self.execution_id,
            ).append(
                CommitRequest(
                    request_id=f"tool:{invocation_id}:failed",
                    execution_id=self.execution_id,
                    branch_id=self.branch_id,
                    based_on_hash=based_on,
                    author=tool_actor,
                    event=ToolFailed(
                        action_id=trusted.id,
                        invocation_id=invocation_id,
                        requested_by_run_id=self.actor.run_id,
                        error=str(error),
                        error_type=type(error).__name__,
                    ),
                )
            )
            await self._notify(failed)
            await self._refresh(observed_hash=failed.hash)
            return ToolResponse(success=False, result=None, error=str(error))

        if trusted.name == INSPECT_SUBAGENT_TOOL_NAME:
            effects = await self._inspect_subagent_effects(trusted.arguments)
        elif trusted.is_submission and not self._require_submission:
            effects = ToolEffects(
                observation="subagents cannot submit the parent execution",
                status="execution_error",
            )
        else:
            try:
                effects = await anyio.to_thread.run_sync(
                    lambda: execute_action(self.environment, execution_state, trusted)
                )
            except Exception as exc:
                return await fail(exc)
        try:
            completed = await self.state_store.bind(
                tool_actor,
                branch_id=self.branch_id,
                execution_id=self.execution_id,
            ).append(
                CommitRequest(
                    request_id=f"tool:{invocation_id}:completed",
                    execution_id=self.execution_id,
                    branch_id=self.branch_id,
                    based_on_hash=based_on,
                    author=tool_actor,
                    event=ToolCompleted(
                        action_id=trusted.id,
                        invocation_id=invocation_id,
                        requested_by_run_id=self.actor.run_id,
                        observation=effects.observation,
                        status=effects.status,
                        environment_operations=effects.environment_operations,
                        workspace_delta=effects.workspace_delta,
                        usage_delta=effects.usage_delta,
                        runtime_update=effects.runtime_update,
                        expected_environment_revision=effects.expected_environment_revision,
                        expected_workspace_revision=effects.expected_workspace_revision,
                        duration_ms=effects.duration_ms,
                    ),
                ),
            )
        except CommitConflictError as exc:
            return await fail(exc)
        await self._notify(completed)

        if trusted.is_submission and effects.success:
            answer = trusted.arguments["answer"]
            assert isinstance(answer, str)
            surrendered = bool(
                self.state.runtime.metadata.get("surrender_sentinel")
            ) and answer == str(self.state.runtime.metadata["surrender_sentinel"])
            accepted = await self._append_runtime(
                SubmissionAccepted(
                    action_id=trusted.id,
                    requested_by_run_id=self.actor.run_id,
                    answer=answer,
                    surrendered=surrendered,
                ),
                f"submission:{trusted.id}:accepted",
                based_on_hash=completed.hash,
            )
            completion_hash = accepted.hash
        else:
            completion_hash = completed.hash
        await self._refresh(observed_hash=completion_hash)
        rendered = (
            effects.observation
            if isinstance(effects.observation, str)
            else json.dumps(effects.observation, ensure_ascii=False)
        )
        return ToolResponse(
            success=effects.success,
            result=rendered if effects.success else None,
            error=None if effects.success else rendered,
        )

    async def execute(
        self,
        action: Action,
        *,
        usage: Mapping[str, Any] | None = None,
    ) -> ToolResponse:
        results = await self.execute_many((action,), usage=usage)
        return results[0]

    async def execute_many(
        self,
        actions: Sequence[Action],
        *,
        usage: Mapping[str, Any] | None = None,
    ) -> tuple[ToolResponse, ...]:
        if not actions:
            return ()
        if len(actions) > 1 and any(action.is_submission for action in actions):
            raise ValueError("submit_answer cannot run in a parallel action group")
        async with self._action_lock:
            if self.state.is_terminal:
                failure = ToolResponse(
                    success=False,
                    result=None,
                    error="the session is terminal; no further actions can run",
                )
                return tuple(failure for _ in actions)
            group_id = str(uuid4()) if len(actions) > 1 else None
            normalized = usage_from_mapping(
                usage,
                llm_calls=1 if usage is not None else 0,
            )
            turn_usage = UsageDelta(
                input_tokens=normalized.input_tokens,
                output_tokens=normalized.output_tokens,
                reasoning_tokens=normalized.reasoning_tokens,
                llm_calls=normalized.llm_calls,
            )
            event = (
                propose_action(actions[0], usage_delta=turn_usage)
                if len(actions) == 1
                else AgentTurnRecorded(
                    messages=(
                        {
                            "role": "assistant",
                            "content": next(
                                (
                                    action.content
                                    for action in actions
                                    if action.content is not None
                                ),
                                None,
                            ),
                            "tool_calls": [action.to_tool_call() for action in actions],
                        },
                    ),
                    actions=tuple(actions),
                    parallel_group_id=group_id,
                    usage_delta=turn_usage,
                )
            )
            proposal = await self._append_agent(
                event,
                f"agent:{self.actor.run_id}:actions:{','.join(action.id for action in actions)}",
            )
            trusted_actions = tuple(
                self.state.actions[action.id].action for action in actions
            )
            if len(trusted_actions) == 1:
                results = (
                    await self._run_tool(
                        trusted_actions[0], proposal_hash=proposal.hash
                    ),
                )
            else:
                results = tuple(
                    await asyncio.gather(
                        *(
                            self._run_tool(action, proposal_hash=proposal.hash)
                            for action in trusted_actions
                        )
                    )
                )
                head = await self.state_store.head(self.branch_id)
                assert head is not None
                assert group_id is not None
                grouped = await self._append_runtime(
                    ParallelGroupCompleted(
                        group_id=group_id,
                        action_ids=tuple(action.id for action in trusted_actions),
                    ),
                    f"parallel:{group_id}:completed",
                    based_on_hash=head.hash,
                )
                await self._refresh(observed_hash=grouped.hash)
            return results

    async def resume_pending_actions(self) -> tuple[ToolResponse, ...]:
        await self._refresh(observed_hash=self._last_observed_hash)
        pending = tuple(
            action_state.action
            for action_state in self.state.actions.values()
            if action_state.requested_by_run_id == self.actor.run_id
            and action_state.status in {"pending", "running"}
        )
        if not pending:
            raise ValueError("the agent has no pending actions to resume")
        return tuple(
            await asyncio.gather(
                *(
                    self._run_tool(
                        action,
                        proposal_hash=self.state.through_commit_hash,
                    )
                    for action in pending
                )
            )
        )

    async def run_delegate(
        self, agent: Agent, *, max_iterations: int | None = None
    ) -> AgentOutcome:
        if not isinstance(agent, Agent):
            raise TypeError("a delegate must implement run_session(AgentSession)")
        limit = self.iteration_limit if max_iterations is None else max_iterations
        if limit < 1:
            raise ValueError("a delegate max_iterations budget must be at least 1")
        token = self._delegated_iteration_limit.set(limit)
        try:
            return await _run_bound_agent(agent, self)
        finally:
            self._delegated_iteration_limit.reset(token)

    async def spawn_subagent(
        self,
        agent: Agent,
        *,
        handoff: JsonValue,
        actor_id: str | None = None,
        max_iterations: int | None = None,
    ) -> str:
        if not isinstance(agent, Agent):
            raise TypeError("a subagent must implement run_session(AgentSession)")
        child_run_id = f"agent-run-{uuid4()}"
        child_actor_id = actor_id or type(agent).__name__
        child_capabilities = agent_session_capabilities(agent)
        child_metadata: Mapping[str, JsonValue] = {
            "session_capabilities": child_capabilities.to_metadata()
        }
        cutoff = self._last_observed_hash
        spawned = await self._append_agent(
            AgentSpawned(
                child_run_id=child_run_id,
                child_actor_id=child_actor_id,
                context_cutoff_hash=cutoff,
                handoff=handoff,
                metadata=child_metadata,
            ),
            f"agent:{self.actor.run_id}:spawn:{child_run_id}",
        )
        started = await self._append_runtime(
            AgentStarted(
                agent_run_id=child_run_id,
                agent_id=child_actor_id,
                parent_run_id=self.actor.run_id,
                context_cutoff_hash=cutoff,
                handoff=handoff,
                metadata=child_metadata,
            ),
            f"agent:{child_run_id}:started",
            based_on_hash=spawned.hash,
        )
        child_actor = ActorRef(
            kind="agent",
            actor_id=child_actor_id,
            run_id=child_run_id,
            parent_run_id=self.actor.run_id,
        )
        child_state = await self.state_store.materialize(self.branch_id)
        child = AgentSession(
            self.environment,
            child_state,
            actor=child_actor,
            state_store=self.state_store,
            branch_id=self.branch_id,
            runtime_actor=self.runtime_actor,
            max_iterations=max_iterations or self.iteration_limit,
            observer=self.observer,
            observation_context=self.observation_context,
            hooks=getattr(agent, "hooks", None),
            agent=agent,
            require_submission=False,
            capabilities=child_capabilities,
        )
        child._last_observed_hash = started.hash

        async def run_child() -> AgentOutcome:
            try:
                outcome = await _run_bound_agent(agent, child)
                terminal = AgentCompleted(
                    agent_run_id=child_run_id,
                    status=outcome.status,
                    result_summary={
                        "answer": outcome.answer,
                        "error": outcome.error,
                    },
                    trace_head=child._last_observed_hash,
                    usage_delta=_agent_usage_delta(
                        outcome,
                        child.state.usage_by_run.get(child_run_id),
                    ),
                    metadata=outcome.metadata,
                )
                await child._append_agent(terminal, f"agent:{child_run_id}:completed")
                self._subagent_outcomes[child_run_id] = outcome
                return outcome
            except BaseException as exc:
                terminal = AgentCompleted(
                    agent_run_id=child_run_id,
                    status="failed",
                    result_summary={"error": str(exc)},
                    trace_head=child._last_observed_hash,
                )
                await child._append_runtime(
                    terminal,
                    f"agent:{child_run_id}:failed",
                    based_on_hash=child._last_observed_hash,
                )
                raise
            finally:
                child.close()

        self._subagent_tasks[child_run_id] = asyncio.create_task(run_child())
        return child_run_id

    async def list_subagents(self) -> tuple[str, ...]:
        state = await self._refresh()
        self._last_observed_hash = state.through_commit_hash
        run = state.agent_runs.get(self.actor.run_id)
        return () if run is None else run.child_run_ids

    async def wait_for_subagent(self, child_run_id: str) -> AgentOutcome:
        if child_run_id not in await self.list_subagents():
            raise PermissionError("the selected run is not a direct child")
        task = self._subagent_tasks.get(child_run_id)
        if task is not None:
            try:
                return await task
            finally:
                state = await self._refresh()
                self._last_observed_hash = state.through_commit_hash
        outcome = self._subagent_outcomes.get(child_run_id)
        if outcome is not None:
            return outcome
        state = await self.state_store.materialize(self.branch_id)
        run = state.agent_runs[child_run_id]
        if run.status in {"created", "running"}:
            raise RuntimeError("subagent is still running in another worker")
        summary = run.result_summary if isinstance(run.result_summary, Mapping) else {}
        if run.status == "completed":
            answer = summary.get("answer")
            if not isinstance(answer, str):
                raise RuntimeError("completed subagent has no durable answer")
            return AgentOutcome(
                status="completed", answer=answer, metadata=run.metadata
            )
        if run.status == "surrendered":
            return AgentOutcome(status="surrendered", metadata=run.metadata)
        status = (
            run.status
            if run.status
            in {
                "iteration_limit",
                "timeout",
                "budget_exhausted",
                "cancelled",
                "tool_failure",
                "protocol_failure",
                "harness_failure",
                "agent_failure",
            }
            else "agent_failure"
        )
        error = summary.get("error")
        return AgentOutcome(
            status=status,
            error=error if isinstance(error, str) and error else "subagent failed",
            metadata=run.metadata,
        )

    async def inspect_subagent(self, child_run_id: str) -> SubagentTrace:
        if child_run_id == self.actor.run_id:
            raise PermissionError("an agent cannot inspect itself as a subagent")
        state = await self.state_store.materialize(self.branch_id)
        self._last_observed_hash = state.through_commit_hash
        context = self._resolver.inspect(
            state,
            requester_run_id=self.actor.run_id,
            target_run_id=child_run_id,
        )
        trace_method = getattr(self.state_store, "trace_commits", None)
        commits = (
            ()
            if trace_method is None
            else await trace_method(child_run_id, branch_id=self.branch_id)
        )
        return SubagentTrace(context=context, commits=commits)

    async def import_subagent_context(
        self,
        child_run_id: str,
        *,
        representation: JsonValue | None = None,
        source_commit_ids: Sequence[str] | None = None,
    ) -> Commit:
        trace = await self.inspect_subagent(child_run_id)
        selected_ids = tuple(
            source_commit_ids or (commit.hash for commit in trace.commits)
        )
        if representation is None:
            representation = trace.context.agent_run.result_summary
        return await self._append_agent(
            ContextImported(
                source_run_id=child_run_id,
                source_commit_ids=selected_ids,
                representation=representation,
            ),
            f"agent:{self.actor.run_id}:import:{child_run_id}:{uuid4()}",
        )

    async def fork_branch(self, *, branch_id: str | None = None) -> AgentSession:
        selected = branch_id or f"experiment-{uuid4()}"
        await self.state_store.create_branch(
            branch_id=selected,
            from_hash=self._last_observed_hash,
            execution_id=self.execution_id,
        )
        environment = self.environment.for_task(f"{self.execution_id}:{selected}")
        branch_state = await self.state_store.materialize(selected)
        environment.prepare_workspace(branch_state.workspace)
        return AgentSession(
            environment,
            branch_state,
            actor=self.actor,
            state_store=self.state_store,
            branch_id=selected,
            runtime_actor=self.runtime_actor,
            previous_state=self.previous_state,
            max_iterations=self._max_iterations,
            observer=self.observer,
            observation_context=self.observation_context,
            hooks=self._hooks,
            agent=self._hook_agent,
        )

    def close(self) -> None:
        transport = self._mcp_transport
        if transport is not None:
            transport.close()
            self._mcp_transport = None

    def final_messages(self) -> tuple[Mapping[str, JsonValue], ...]:
        return self.messages[self._initial_message_count :]


@dataclass(frozen=True, slots=True)
class AgentSessionOutcome:
    outcome: AgentOutcome
    state: ExecutionState
    final_commit: Commit
    messages: tuple[Mapping[str, JsonValue], ...]


def _normalize_outcome(result: AgentOutcome) -> AgentOutcome:
    metadata = _json_value(result.metadata)
    if not isinstance(metadata, Mapping):
        raise TypeError("session outcome metadata must be JSON objects")
    return AgentOutcome(
        status=result.status,
        answer=result.answer,
        error=result.error,
        usage=AgentUsage(
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            reasoning_tokens=result.usage.reasoning_tokens,
            llm_calls=result.usage.llm_calls,
        ),
        metadata=metadata,
    )


def _agent_usage_delta(
    result: AgentOutcome,
    already_recorded: UsageState | None = None,
) -> UsageDelta:
    recorded = already_recorded or UsageState()
    return UsageDelta(
        input_tokens=max(0, result.usage.input_tokens - recorded.input_tokens),
        output_tokens=max(0, result.usage.output_tokens - recorded.output_tokens),
        reasoning_tokens=max(
            0, result.usage.reasoning_tokens - recorded.reasoning_tokens
        ),
        llm_calls=max(0, result.usage.llm_calls - recorded.llm_calls),
        agent_steps=1,
    )


def _enforce_tool_submission(
    result: AgentOutcome, interface: AgentSession
) -> AgentOutcome:
    if not interface._require_submission:
        return result
    submission = interface.state.submission
    if submission is None:
        if result.status not in {"completed", "surrendered"}:
            return result
        return AgentOutcome(
            status="protocol_failure",
            error=f"agent returned {result.status!r} without calling submit_answer",
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


async def _run_bound_agent(agent: Agent, interface: AgentSession) -> AgentOutcome:
    raw_hooks = getattr(agent, "hooks", None)
    hooks = AgentHooks() if raw_hooks is None else raw_hooks
    if not isinstance(hooks, AgentHooks):
        raise TypeError("agent hooks must be an AgentHooks instance")
    try:
        before_task = await interface.run_hooks(
            HookPoint.BEFORE_TASK, agent=agent, hooks=hooks
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
        result = _enforce_tool_submission(_normalize_outcome(result), interface)
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
        data={"status": result.status, "answer": result.answer, "error": result.error},
    )
    return result


async def run_agent_session(
    agent: Agent,
    environment: Environment,
    state: ExecutionState,
    *,
    actor: ActorRef,
    state_store: CommitStore,
    branch_id: str = "main",
    runtime_actor: ActorRef | None = None,
    last_score: Mapping[str, Any] | None = None,
    previous_state: ExecutionState | None = None,
    max_iterations: int,
    observer: Observer | None = None,
    observation_context: ObservationContext | None = None,
) -> AgentSessionOutcome:
    environment.prepare_workspace(state.workspace)
    interface = AgentSession(
        environment,
        state,
        actor=actor,
        state_store=state_store,
        branch_id=branch_id,
        runtime_actor=runtime_actor,
        last_score=last_score,
        previous_state=previous_state,
        max_iterations=max_iterations,
        observer=observer,
        observation_context=observation_context,
        hooks=getattr(agent, "hooks", None),
        agent=agent,
    )
    try:
        result = await _run_bound_agent(agent, interface)
        if interface._subagent_tasks:
            await asyncio.gather(*tuple(interface._subagent_tasks.values()))
        completed = await interface._append_agent(
            AgentCompleted(
                agent_run_id=actor.run_id,
                status=result.status,
                result_summary={"answer": result.answer, "error": result.error},
                trace_head=interface._last_observed_hash,
                usage_delta=_agent_usage_delta(
                    result,
                    interface.state.usage_by_run.get(actor.run_id),
                ),
                metadata=result.metadata,
            ),
            f"agent:{actor.run_id}:completed",
        )
        return AgentSessionOutcome(
            outcome=result,
            state=interface.state,
            final_commit=completed,
            messages=interface.final_messages(),
        )
    finally:
        pending_children = tuple(
            task for task in interface._subagent_tasks.values() if not task.done()
        )
        for task in pending_children:
            task.cancel()
        if pending_children:
            await asyncio.gather(*pending_children, return_exceptions=True)
        interface.close()


__all__ = [
    "INSPECT_SUBAGENT_TOOL_NAME",
    "Agent",
    "AgentSession",
    "AgentSessionCapabilities",
    "AgentSessionOutcome",
    "MCPConnection",
    "SubagentTrace",
    "ToolResponse",
    "agent_session_capabilities",
    "inspect_subagent_tool",
    "run_agent_session",
]
