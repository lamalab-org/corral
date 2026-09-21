"""A narrow session broker for unprivileged native and SDK agent workers."""

from __future__ import annotations

import asyncio
import base64
import json
import secrets
import socket
import struct
import threading
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import anyio
from pydantic import TypeAdapter

from corral.agents.schema import AgentOutcome
from corral.agents.session import AgentSession, AgentSessionCapabilities, SubagentTrace
from corral.core.action import Action
from corral.core.actors import ActorRef
from corral.core.tool import ToolConnection, ToolResponse
from corral.runtime import permissions

_MAX_MESSAGE = 64 * 1024 * 1024
_METHODS = frozenset(
    {
        "execute",
        "execute_many",
        "record_message",
        "record_messages",
        "get_agent_state",
        "set_agent_state",
        "run_delegate",
        "spawn_subagent",
        "list_subagents",
        "wait_for_subagent",
        "inspect_subagent",
        "import_subagent_context",
        "fork_branch",
        "promote_artifacts",
        "shutdown_jobs",
    }
)
_RETURNS = {
    "execute": ToolResponse,
    "execute_many": tuple[ToolResponse, ...],
    "run_delegate": AgentOutcome,
    "wait_for_subagent": AgentOutcome,
    "inspect_subagent": SubagentTrace,
}
_ARGUMENTS = {
    "execute": ("action",),
    "execute_many": ("actions",),
    "record_message": ("raw",),
    "record_messages": ("raw_messages",),
    "set_agent_state": ("namespace", "value"),
    "run_delegate": ("agent",),
    "spawn_subagent": ("agent",),
    "wait_for_subagent": ("child_run_id",),
    "inspect_subagent": ("child_run_id",),
    "import_subagent_context": ("child_run_id",),
}


def _json(value: Any) -> Any:
    return TypeAdapter(type(value)).dump_python(value, mode="json")


def _snapshot(session: AgentSession) -> dict[str, Any]:
    fields = (
        "execution_id",
        "execution_workspace",
        "workspace",
        "branch_id",
        "actor",
        "messages",
        "examples",
        "iteration_limit",
        "prompt",
        "task_id",
        "tools",
        "tool_connection",
        "model_name",
        "previous_messages",
        "previous_commit_hash",
        "surrender_allowed",
        "submission",
        "submission_status",
        "capabilities",
        "_require_submission",
    )
    snapshot = {name: _json(getattr(session, name)) for name in fields}
    # Evaluation records can contain private scoring inputs and diagnostics.
    # Reflexion needs only the public score and attempt identifier.
    evaluation = session.previous_evaluation
    snapshot["previous_evaluation"] = (
        None
        if evaluation is None
        else {
            key: value
            for key, value in evaluation.items()
            if (key == "score" and isinstance(value, int | float))
            or (key == "trial_id" and isinstance(value, str))
        }
    )
    return snapshot


@dataclass
class _DelegatedAgent:
    """Opaque worker data; the trusted broker NEVER unpickles this payload."""

    blob: str
    tool_transport: str
    session_capabilities: dict[str, bool]

    async def run_session(self, session: AgentSession) -> AgentOutcome:
        return await run_agent(self, session)


class RemoteSession:
    """The public session API, with all durable operations brokered as JSON."""

    def __init__(
        self, host: str, port: int, token: str, handle: str, snapshot: dict[str, Any]
    ):
        self._endpoint = (host, port)
        self._token = token
        self._handle = handle
        self._update(snapshot)
        self._initial_message_count = len(self.messages)
        self.environment = SimpleNamespace(
            shutdown_jobs=lambda: self._sync_call("shutdown_jobs", {})
        )

    def _update(self, snapshot: dict[str, Any]) -> None:
        models = {
            "actor": ActorRef,
            "tool_connection": ToolConnection,
            "capabilities": AgentSessionCapabilities,
        }
        for name, value in snapshot.items():
            decoded = (
                TypeAdapter(models[name]).validate_python(value)
                if name in models and value is not None
                else value
            )
            setattr(self, name, decoded)

    def __getattr__(self, name: str) -> Any:
        if name not in _METHODS:
            raise AttributeError(name)

        async def call(*args: Any, **kwargs: Any) -> Any:
            names = _ARGUMENTS.get(name, ())
            if len(args) > len(names):
                raise TypeError(f"too many positional arguments to {name}")
            arguments = {**dict(zip(names[: len(args)], args, strict=True)), **kwargs}
            if "agent" in arguments:
                agent = arguments["agent"]
                arguments["agent"] = {
                    "blob": base64.b64encode(permissions.serialize(agent)).decode(),
                    "tool_transport": getattr(agent, "tool_transport", "python"),
                    "session_capabilities": AgentSessionCapabilities.from_value(
                        getattr(agent, "session_capabilities", None)
                    ).to_metadata(),
                }
                if name == "spawn_subagent":
                    arguments.setdefault("actor_id", type(agent).__name__)
            arguments = {key: _json(value) for key, value in arguments.items()}
            return await anyio.to_thread.run_sync(
                lambda: self._sync_call(name, arguments)
            )

        return call

    @staticmethod
    def _read(stream: socket.socket, count: int) -> bytes:
        chunks = bytearray()
        while len(chunks) < count:
            data = stream.recv(count - len(chunks))
            if not data:
                raise RuntimeError("session broker disconnected")
            chunks.extend(data)
        return bytes(chunks)

    def _sync_call(self, method: str, arguments: dict[str, Any]) -> Any:
        request = json.dumps(
            {
                "token": self._token,
                "handle": self._handle,
                "method": method,
                "arguments": arguments,
            }
        ).encode()
        if len(request) > _MAX_MESSAGE:
            raise ValueError("session request is too large")
        with socket.create_connection(self._endpoint) as stream:
            stream.sendall(struct.pack("!I", len(request)) + request)
            length = struct.unpack("!I", self._read(stream, 4))[0]
            if length > _MAX_MESSAGE:
                raise RuntimeError("session response is too large")
            response = json.loads(self._read(stream, length))
        if not response["ok"]:
            raise RuntimeError(response["error"])
        self._update(response["snapshot"])
        value = response["result"]
        if method == "fork_branch":
            return RemoteSession(
                *self._endpoint, self._token, value["handle"], value["snapshot"]
            )
        if method in _RETURNS:
            return TypeAdapter(_RETURNS[method]).validate_python(value)
        return value

    def get_agent_state(self, namespace: str, *, previous: bool = False) -> Any:
        """Read only this agent's bookkeeping, never an execution projection."""
        return self._sync_call(
            "get_agent_state", {"namespace": namespace, "previous": previous}
        )

    run_hooks = AgentSession.run_hooks

    def final_messages(self) -> tuple[dict[str, Any], ...]:
        return tuple(self.messages[self._initial_message_count :])


async def run_agent(agent: Any, session: AgentSession) -> AgentOutcome:
    """Keep implementations unchanged while moving their process off root."""
    workspace = session.workspace
    if not workspace:
        raise RuntimeError("Docker agents require an assigned workspace")
    snapshot = _snapshot(session)
    token = secrets.token_hex(32)
    handle = secrets.token_hex(16)
    sessions = {handle: session}
    tasks: set[asyncio.Task[Any]] = set()

    async def serve(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        tasks.add(task)
        try:
            length = struct.unpack("!I", await reader.readexactly(4))[0]
            if length > _MAX_MESSAGE:
                raise ValueError("session request is too large")
            request = json.loads(await reader.readexactly(length))
            if not secrets.compare_digest(request["token"], token):
                raise PermissionError("invalid session capability")
            selected = sessions[request["handle"]]
            method = request["method"]
            if method not in _METHODS:
                raise PermissionError("session operation is not permitted")
            arguments = request["arguments"]
            if method == "fork_branch":
                arguments.setdefault("workspace_parent", workspace)
                if arguments["workspace_parent"] != workspace:
                    raise PermissionError(
                        "node workspaces must belong to the calling agent"
                    )
            if method == "execute":
                arguments["action"] = TypeAdapter(Action).validate_python(
                    arguments["action"]
                )
            elif method == "execute_many":
                arguments["actions"] = TypeAdapter(tuple[Action, ...]).validate_python(
                    arguments["actions"]
                )
            elif method in {"run_delegate", "spawn_subagent"}:
                supplied = arguments["agent"]
                arguments["agent"] = _DelegatedAgent(
                    blob=supplied["blob"],
                    tool_transport=supplied["tool_transport"],
                    session_capabilities=AgentSessionCapabilities.from_value(
                        supplied["session_capabilities"]
                    ).to_metadata(),
                )
            if method == "get_agent_state":
                result = selected.get_agent_state(**arguments)
            elif method == "shutdown_jobs":
                await anyio.to_thread.run_sync(selected.environment.shutdown_jobs)
                result = None
            else:
                result = await getattr(selected, method)(**arguments)
            if method == "fork_branch":
                branch_handle = secrets.token_hex(16)
                sessions[branch_handle] = result
                result = {"handle": branch_handle, "snapshot": _snapshot(result)}
            elif method == "inspect_subagent":
                # Raw commits contain hidden environment updates and artifact
                # references. Only the child's public conversation may cross.
                result = _json(result)
                result["context"]["task"] = {}
                result["commits"] = []
            elif method == "import_subagent_context":
                result = result.hash
            response = {
                "ok": True,
                "result": _json(result),
                "snapshot": _snapshot(selected),
            }
        except Exception as exc:
            response = {"ok": False, "error": str(exc)}
        try:
            data = json.dumps(response).encode()
            if len(data) > _MAX_MESSAGE:
                data = b'{"ok":false,"error":"session response is too large"}'
            writer.write(struct.pack("!I", len(data)) + data)
            await writer.drain()
        finally:
            writer.close()
            await writer.wait_closed()
            tasks.discard(task)

    server = await asyncio.start_server(serve, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    cancelled = threading.Event()
    worker = asyncio.create_task(
        anyio.to_thread.run_sync(
            lambda: permissions.run_worker(
                "agent",
                (agent, ("127.0.0.1", port, token, handle, snapshot)),
                workspace,
                cancel=cancelled,
            ),
        )
    )
    try:
        result = await asyncio.shield(worker)
        return TypeAdapter(AgentOutcome).validate_python(result)
    finally:
        cancelled.set()
        # Complete identity cleanup before the controller releases this session
        # or returns checkpoint ownership to the host user.
        with anyio.CancelScope(shield=True):
            await asyncio.gather(worker, return_exceptions=True)
        server.close()
        await server.wait_closed()
        for task in tuple(tasks):
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for selected in sessions.values():
            if selected is not session:
                await anyio.to_thread.run_sync(selected.environment.shutdown_jobs)
