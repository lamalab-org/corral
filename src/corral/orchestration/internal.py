"""Container-only orchestration commands used by the Docker task launcher."""

from __future__ import annotations

import importlib
import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter

from corral.observability import LoggingObserver, ObservationContext
from corral.orchestration.activities import RuntimeRegistry
from corral.orchestration.launchers import LocalTaskLauncher
from corral.orchestration.models import RUNTIME_PROTOCOL_VERSION, RunTaskInput
from corral.persistence import SQLiteCommitStore, WorkspaceManager
from corral.workspace import WorkspaceFilesystem, build_terminal_tool


def _registry_from_module(specification: str, request: RunTaskInput) -> RuntimeRegistry:
    module_name, separator, attribute = specification.partition(":")
    if not separator or not module_name or not attribute:
        raise ValueError("registry_module must use the form 'module:function'")
    factory = getattr(importlib.import_module(module_name), attribute, None)
    if not callable(factory):
        raise TypeError(f"registry factory {specification!r} is not callable")
    registry = factory(request)
    if not isinstance(registry, RuntimeRegistry):
        raise TypeError("custom registry factory must return RuntimeRegistry")
    return registry


def _built_in_registry(
    request: RunTaskInput,
    *,
    workspace_manager: WorkspaceManager,
) -> RuntimeRegistry:
    agent_definition = request.agent_runtime
    environment_definition = request.environment_runtime
    if agent_definition is None or environment_definition is None:
        raise ValueError("built-in Docker runtime definitions are incomplete")

    # This is deliberately the same public factory used by the host CLI.
    from corral.cli import create_agent
    from corral.runtime.environment_loader import (
        load_environment_group,
    )

    agent = create_agent(
        agent_definition.name,
        model=agent_definition.model,
        api_endpoint=agent_definition.api_endpoint,
        temperature=agent_definition.temperature,
        agent_kwargs=agent_definition.options,
    )
    options = dict(environment_definition.options)
    options["work_dir"] = "/workspace"
    environments = load_environment_group(
        environment_definition.name,
        env_kwargs=options,
        repository_root=environment_definition.repository_root,
    )
    if request.environment_id not in environments:
        known = ", ".join(environments)
        raise KeyError(
            f"container environment has no task {request.environment_id!r}; "
            f"known tasks: {known}"
        )
    return RuntimeRegistry(
        agents={request.agent_id: agent},
        environments={request.environment_id: environments[request.environment_id]},
        workspace_manager_factory=lambda _execution_id: workspace_manager,
    )


def _write_result(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.chmod(0o640)
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _restore_host_ownership(root: Path) -> None:
    """Return bind-mounted checkpoint files to the invoking host user."""
    if os.geteuid() != 0:
        return
    uid = int(os.environ.get("CORRAL_HOST_UID", "0"))
    gid = int(os.environ.get("CORRAL_HOST_GID", "0"))
    for entry in (*root.rglob("*"), root):
        if entry.is_symlink():
            raise ValueError("checkpoint directories cannot contain symbolic links")
        os.chown(entry, uid, gid)


async def run_task_from_files(request_file: str | Path, result_file: str | Path) -> int:
    """Reconstruct one runtime, recover its last commit, and publish StateRef."""
    request = TypeAdapter(RunTaskInput).validate_json(
        Path(request_file).read_text(encoding="utf-8")
    )
    docker = request.sandbox.docker
    if docker is None:
        raise ValueError("internal run-task accepts Docker sandbox requests only")
    if docker.runtime_protocol_version != RUNTIME_PROTOCOL_VERSION:
        raise RuntimeError(
            "container/host runtime protocol mismatch: "
            f"{RUNTIME_PROTOCOL_VERSION} != {docker.runtime_protocol_version}"
        )

    checkpoint_root = Path(request_file).resolve().parent
    workspace_snapshots = checkpoint_root / "workspace-snapshots"
    legacy_snapshots = checkpoint_root / "snapshots"
    if (
        not (workspace_snapshots / "latest.json").is_file()
        and (legacy_snapshots / "latest.json").is_file()
    ):
        workspace_snapshots = legacy_snapshots
    workspace_manager = WorkspaceManager(
        artifact_root=checkpoint_root / "artifacts",
        snapshot_root=workspace_snapshots,
    )
    store = SQLiteCommitStore(
        checkpoint_root / "commits.sqlite3",
        execution_id=request.execution_id,
        state_snapshot_root=checkpoint_root / "state-snapshots",
    )
    registry: RuntimeRegistry | None = None
    try:
        if docker.registry_module is not None:
            registry = _registry_from_module(docker.registry_module, request)
        else:
            registry = _built_in_registry(request, workspace_manager=workspace_manager)
        environment = registry.environment(request.environment_id, request.execution_id)
        environment.workspace_manager = workspace_manager
        if environment.workspace_path:
            workspace_path = Path(environment.workspace_path).resolve()
            if not workspace_path.is_relative_to(Path("/workspace")):
                raise ValueError(
                    "Docker runtime workspaces must be located below /workspace"
                )
            environment.tools["terminal"] = build_terminal_tool(
                WorkspaceFilesystem(workspace_path)
            )
        head = await store.for_execution(request.execution_id).head("main")
        if head is not None:
            recovered_state = await store.for_execution(
                request.execution_id
            ).materialize("main", head.hash)
            environment.prepare_workspace(recovered_state.workspace)

        launcher = LocalTaskLauncher(
            store,
            registry,
            LoggingObserver(),
            scaffold_sandbox="docker",
        )
        result = await launcher.run(
            request,
            observation_context=ObservationContext(
                execution_id=request.execution_id,
                benchmark_run_id=request.benchmark_run_id,
                task_id=request.task_id,
            ),
        )
        _write_result(Path(result_file), asdict(result))
    finally:
        if registry is not None:
            registry.close()
        store.close()
        _restore_host_ownership(checkpoint_root)
    return 0


__all__ = ["run_task_from_files"]
