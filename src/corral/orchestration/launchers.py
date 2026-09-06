"""Task launchers for local and one-container-per-trial execution."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import tempfile
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import TypeAdapter

from corral.core.state import ExecutionState, TaskOutput
from corral.orchestration.models import (
    DockerSandboxSpec,
    RunTaskInput,
    SandboxRetention,
    StateRef,
)
from corral.persistence import CommitNotFoundError
from corral.runtime import TaskRuntime

if TYPE_CHECKING:
    from collections.abc import Mapping

    from corral.core.commit import Commit
    from corral.observability import ObservationContext, Observer
    from corral.orchestration.registry import RuntimeRegistry
    from corral.persistence import CommitStore


class TaskLauncher(Protocol):
    """Execution boundary selected for a task."""

    async def run(
        self,
        request: RunTaskInput,
        *,
        observation_context: ObservationContext | None = None,
    ) -> StateRef: ...


class DockerInfrastructureError(RuntimeError):
    """A container could not be created, supervised, or decoded safely."""


def state_ref(state: ExecutionState, commit: Commit) -> StateRef:
    """Project a materialized head into a task result reference."""
    output = (
        {"answer": state.submission}
        if state.submission is not None and state.runtime.status != "surrendered"
        else None
    )
    raw_error = state.runtime.metadata.get("error")
    return StateRef(
        commit_hash=commit.hash,
        execution_id=state.execution_id,
        branch_id=state.branch_id,
        sequence=commit.sequence,
        status=state.runtime.status,
        agent_steps=state.usage.agent_steps,
        submission=state.submission,
        output=output,
        error=str(raw_error) if raw_error is not None else None,
    )


class LocalTaskLauncher:
    """The existing in-process runtime, retained for `corral run` and debugging."""

    def __init__(
        self,
        state_store: CommitStore,
        registry: RuntimeRegistry,
        observer: Observer,
        *,
        scaffold_sandbox: str = "local",
    ) -> None:
        self.state_store = state_store
        self.registry = registry
        self.runtime = TaskRuntime(state_store, observer)
        self.scaffold_sandbox = scaffold_sandbox

    async def run(
        self,
        request: RunTaskInput,
        *,
        observation_context: ObservationContext | None = None,
    ) -> StateRef:
        agent = self.registry.agent(request.agent_id)
        environment = self.registry.environment(
            request.environment_id, request.execution_id
        )
        dependencies = {
            task_id: TaskOutput(output=output)
            for task_id, output in request.dependency_outputs.items()
        }
        last_evaluation = self.registry.last_evaluation(request.task_id)
        previous_state = None
        if last_evaluation is not None:
            previous_hash = last_evaluation.get("commit_hash")
            if isinstance(previous_hash, str):
                try:
                    execution_store = self.state_store.for_execution(
                        str(last_evaluation.get("trial_id"))
                    )
                    previous_state = await execution_store.materialize(
                        "main", previous_hash
                    )
                except CommitNotFoundError:
                    previous_state = None

        configured_model = request.model
        if configured_model is None:
            agent_model = getattr(agent, "model", None)
            if isinstance(agent_model, str) and agent_model:
                configured_model = agent_model
        scaffold_metadata: dict[str, Any] = {
            "name": type(agent).__name__,
            "enable_surrender": request.enable_surrender,
            "sandbox": self.scaffold_sandbox,
        }
        if request.sandbox.docker is not None:
            docker = request.sandbox.docker
            scaffold_metadata.update(
                sandbox_image_digest=docker.image_digest,
                sandbox_network=docker.network,
                sandbox_resources={
                    "cpus": docker.cpus,
                    "memory": docker.memory,
                    "pids_limit": docker.pids_limit,
                },
                sandbox_runtime_protocol=docker.runtime_protocol_version,
            )
        state = await self.runtime.run(
            agent,
            environment,
            execution_id=request.execution_id,
            started_at=datetime.fromisoformat(request.started_at),
            max_iterations=request.max_iterations,
            dependency_outputs=dependencies,
            model_metadata=(
                {"name": configured_model} if configured_model is not None else {}
            ),
            scaffold_metadata=scaffold_metadata,
            last_score=last_evaluation,
            previous_state=previous_state,
            observation_context=observation_context,
        )
        execution_store = self.state_store.for_execution(request.execution_id)
        head = await execution_store.head("main")
        if head is None:
            raise RuntimeError("task runtime returned without a branch head")
        return state_ref(state, head)


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
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
        temporary.replace(path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


async def _command(
    *arguments: str,
    output_limit: int = 256_000,
    check: bool = True,
) -> tuple[int, str]:
    try:
        process = await asyncio.create_subprocess_exec(
            *arguments,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
        )
    except FileNotFoundError as exc:
        raise DockerInfrastructureError(
            f"Docker executable {arguments[0]!r} was not found"
        ) from exc
    output = bytearray()

    async def consume_output() -> None:
        assert process.stdout is not None
        while chunk := await process.stdout.read(64 * 1024):
            output.extend(chunk)
            overflow = len(output) - output_limit
            if overflow > 0:
                del output[:overflow]

    try:
        await asyncio.gather(process.wait(), consume_output())
    except asyncio.CancelledError:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
        with contextlib.suppress(Exception):
            await asyncio.wait_for(process.wait(), timeout=5)
        raise
    text = bytes(output).decode("utf-8", errors="replace").strip()
    if check and process.returncode != 0:
        raise DockerInfrastructureError(
            f"command {arguments[0]} {arguments[1] if len(arguments) > 1 else ''} "
            f"failed with exit code {process.returncode}: {text}"
        )
    return int(process.returncode or 0), text


class DockerTaskLauncher:
    """Create and supervise one hardened Docker container for one trial."""

    def __init__(
        self,
        state_store: Any,
        *,
        docker_executable: str = "docker",
        output_limit: int = 256_000,
    ) -> None:
        if not hasattr(state_store, "execution_dir"):
            raise TypeError("DockerTaskLauncher requires an execution-sharded store")
        self.state_store = state_store
        self.docker_executable = docker_executable
        self.output_limit = output_limit

    @classmethod
    async def preflight(
        cls,
        spec: DockerSandboxSpec,
        *,
        docker_executable: str = "docker",
        build_context: str | Path | None = None,
        dockerfile: str | Path | None = None,
    ) -> DockerSandboxSpec:
        """Ensure an image exists before trials start and pin it to an image ID."""
        if spec.image_digest is not None:
            await _command(
                docker_executable,
                "image",
                "inspect",
                spec.image_digest,
                output_limit=32_000,
            )
            return spec
        code, digest = await _command(
            docker_executable,
            "image",
            "inspect",
            "--format",
            "{{.Id}}",
            spec.image,
            output_limit=32_000,
            check=False,
        )
        if code != 0:
            if build_context is not None and dockerfile is not None:
                await _command(
                    docker_executable,
                    "build",
                    "--file",
                    str(Path(dockerfile).resolve()),
                    "--tag",
                    spec.image,
                    str(Path(build_context).resolve()),
                    output_limit=1_000_000,
                )
            else:
                await _command(
                    docker_executable,
                    "pull",
                    spec.image,
                    output_limit=1_000_000,
                )
            _code, digest = await _command(
                docker_executable,
                "image",
                "inspect",
                "--format",
                "{{.Id}}",
                spec.image,
                output_limit=32_000,
            )
        digest = digest.splitlines()[-1].strip()
        if not digest.startswith("sha256:"):
            raise DockerInfrastructureError(
                f"Docker returned a non-immutable image identifier: {digest!r}"
            )
        return replace(spec, image_digest=digest)

    @staticmethod
    def _names(execution_id: str) -> tuple[str, str]:
        digest = hashlib.sha256(execution_id.encode("utf-8")).hexdigest()[:24]
        return f"corral-task-{digest}", f"corral-workspace-{digest}"

    @staticmethod
    def _write_sandbox_metadata(shard: Path, metadata: dict[str, Any]) -> None:
        _atomic_json(shard / "sandbox.json", metadata)
        execution_metadata_path = shard / "metadata.json"
        execution_metadata = json.loads(
            execution_metadata_path.read_text(encoding="utf-8")
        )
        execution_metadata.update(
            benchmark_run_id=metadata.get("benchmark_run_id"),
            task_id=metadata.get("task_id"),
            trial_index=metadata.get("trial_index"),
            sandbox=metadata,
        )
        _atomic_json(execution_metadata_path, execution_metadata)

    async def _discard_stale(self, container: str, volume: str) -> None:
        await _command(
            self.docker_executable,
            "rm",
            "--force",
            container,
            output_limit=32_000,
            check=False,
        )
        await _command(
            self.docker_executable,
            "volume",
            "rm",
            "--force",
            volume,
            output_limit=32_000,
            check=False,
        )

    async def _cleanup(
        self,
        container: str,
        volume: str,
        *,
        retention: str,
        failed: bool,
    ) -> None:
        keep = retention == SandboxRetention.ALWAYS.value or (
            retention == SandboxRetention.ON_FAILURE.value and failed
        )
        if keep:
            return
        await self._discard_stale(container, volume)

    async def run(
        self,
        request: RunTaskInput,
        *,
        observation_context: ObservationContext | None = None,
    ) -> StateRef:
        del observation_context
        docker = request.sandbox.docker
        if docker is None:
            raise ValueError("DockerTaskLauncher received a local sandbox request")
        if docker.image_digest is None:
            raise DockerInfrastructureError(
                "Docker sandbox image was not resolved during benchmark preflight"
            )
        shard = Path(self.state_store.execution_dir(request.execution_id)).resolve()
        close_execution = getattr(self.state_store, "close_execution", None)
        if close_execution is not None:
            await close_execution(request.execution_id)
        request_path = shard / "request.json"
        result_path = shard / "result.json"
        result_path.unlink(missing_ok=True)
        recovered = any(
            (shard / directory / "latest.json").is_file()
            for directory in ("workspace-snapshots", "snapshots")
        )
        _atomic_json(request_path, asdict(request))

        container_name, volume_name = self._names(request.execution_id)
        await self._discard_stale(container_name, volume_name)
        labels = {
            "corral.benchmark_run_id": request.benchmark_run_id or "",
            "corral.task_id": request.task_id,
            "corral.trial_index": str(request.trial_index),
            "corral.execution_id": request.execution_id,
            "corral.image_digest": docker.image_digest or "unresolved",
            "corral.runtime_protocol_version": docker.runtime_protocol_version,
        }
        volume_arguments = [self.docker_executable, "volume", "create"]
        for key, value in labels.items():
            volume_arguments.extend(("--label", f"{key}={value}"))
        volume_arguments.append(volume_name)
        await _command(*volume_arguments, output_limit=32_000)

        create_arguments = [
            self.docker_executable,
            "create",
            "--name",
            container_name,
            "--cpus",
            str(docker.cpus),
            "--memory",
            docker.memory,
            "--pids-limit",
            str(docker.pids_limit),
            "--network",
            docker.network,
            "--read-only",
            "--security-opt",
            "no-new-privileges",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "SETUID",
            "--cap-add",
            "SETGID",
            "--cap-add",
            "CHOWN",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,size=256m",
            "--mount",
            f"type=volume,src={volume_name},dst=/workspace",
            "--mount",
            f"type=bind,src={shard},dst=/corral-state",
            "--env",
            f"CORRAL_TERMINAL_UID={self._terminal_identity()[0]}",
            "--env",
            f"CORRAL_TERMINAL_GID={self._terminal_identity()[1]}",
            "--env",
            "HOME=/tmp",
            "--env",
            f"CORRAL_HOST_UID={os.getuid()}",
            "--env",
            f"CORRAL_HOST_GID={os.getgid()}",
        ]
        for key, value in labels.items():
            create_arguments.extend(("--label", f"{key}={value}"))
        for name in docker.environment_allowlist:
            if name in os.environ:
                create_arguments.extend(("--env", name))
        create_arguments.extend(
            (
                docker.immutable_image,
                "corral",
                "internal",
                "run-task",
                "--request",
                "/corral-state/request.json",
                "--result",
                "/corral-state/result.json",
            )
        )

        container_id = ""
        failed = True
        output = ""
        try:
            _code, container_id = await _command(*create_arguments, output_limit=32_000)
            container_id = container_id.splitlines()[-1].strip()
            self._write_sandbox_metadata(
                shard,
                {
                    "benchmark_run_id": request.benchmark_run_id,
                    "container_id": container_id,
                    "container_name": container_name,
                    "cpus": docker.cpus,
                    "execution_id": request.execution_id,
                    "image": docker.image,
                    "image_digest": docker.image_digest,
                    "memory": docker.memory,
                    "network": docker.network,
                    "pids_limit": docker.pids_limit,
                    "recovered": recovered,
                    "runtime_protocol_version": docker.runtime_protocol_version,
                    "task_id": request.task_id,
                    "trial_index": request.trial_index,
                    "volume_name": volume_name,
                },
            )
            _code, output = await _command(
                self.docker_executable,
                "start",
                "--attach",
                container_id,
                output_limit=self.output_limit,
            )
            if not result_path.is_file():
                raise DockerInfrastructureError(
                    "task container exited without publishing result.json; "
                    f"bounded output: {output}"
                )
            try:
                result = TypeAdapter(StateRef).validate_json(
                    result_path.read_text(encoding="utf-8")
                )
            except Exception as exc:
                raise DockerInfrastructureError(
                    "task container published an invalid StateRef"
                ) from exc
            failed = result.status == "failed"
            snapshot_revision = self._latest_revision(shard)
            sandbox_metadata_path = shard / "sandbox.json"
            sandbox_metadata = json.loads(
                sandbox_metadata_path.read_text(encoding="utf-8")
            )
            sandbox_metadata.update(
                final_commit_hash=result.commit_hash,
                snapshot_revision=snapshot_revision,
                status=result.status,
            )
            self._write_sandbox_metadata(shard, sandbox_metadata)
            return replace(
                result,
                metadata={
                    **result.metadata,
                    "container_id": container_id,
                    "image_digest": docker.image_digest,
                    "recovered": recovered,
                    "runtime_protocol_version": docker.runtime_protocol_version,
                    "snapshot_revision": snapshot_revision,
                },
            )
        except asyncio.CancelledError:
            if container_id:
                await _command(
                    self.docker_executable,
                    "stop",
                    "--time",
                    "10",
                    container_id,
                    output_limit=32_000,
                    check=False,
                )
                await _command(
                    self.docker_executable,
                    "kill",
                    container_id,
                    output_limit=32_000,
                    check=False,
                )
            raise
        finally:
            await asyncio.shield(
                self._cleanup(
                    container_name,
                    volume_name,
                    retention=docker.retention,
                    failed=failed,
                )
            )

    @staticmethod
    def _latest_revision(shard: Path) -> int | None:
        for directory in ("workspace-snapshots", "snapshots"):
            latest = shard / directory / "latest.json"
            if not latest.is_file():
                continue
            with contextlib.suppress(Exception):
                return int(json.loads(latest.read_text(encoding="utf-8"))["revision"])
        return None

    @staticmethod
    def _terminal_identity() -> tuple[int, int]:
        """Choose an unprivileged identity distinct from the host checkpoint owner."""
        candidate = 10001
        if candidate in {os.getuid(), os.getgid()}:
            candidate += 1
        return candidate, candidate


__all__ = [
    "DockerInfrastructureError",
    "DockerTaskLauncher",
    "LocalTaskLauncher",
    "TaskLauncher",
    "state_ref",
]
