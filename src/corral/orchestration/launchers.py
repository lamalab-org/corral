"""Task launchers for local and one-container-per-trial execution."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import shutil
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
from corral.orchestration.parameters import model_parameter_metadata
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


# Runs before the image's CLI, so retained containers also work with images
# built before resume support. A checkpoint import can remove SQLite sidecars
# on the host; docker cp alone would leave their old copies in the volume.
_DOCKER_BOOTSTRAP = """\
import json
import os
from pathlib import Path
root = Path('/corral-state')
missing = json.loads((root / 'checkpoint-import.json').read_text())['missing_sidecars']
for name in ('commits.sqlite3-wal', 'commits.sqlite3-shm'):
    if name in missing:
        (root / name).unlink(missing_ok=True)
(root / 'result.json').unlink(missing_ok=True)
os.execvp('corral', ['corral', 'internal', 'run-task', '--request',
                   '/corral-state/request.json', '--result', '/corral-state/result.json'])
"""


def state_ref(state: ExecutionState, commit: Commit) -> StateRef:
    """Project a materialized head into a task result reference."""
    output = (
        {"answer": state.submission}
        if state.submission is not None and state.runtime.status != "surrendered"
        else None
    )
    raw_error = state.runtime.metadata.get("error")
    model_metadata = state.task.model_dump(mode="json")["model"]
    parameters = {
        key: model_metadata[key]
        for key in ("temperature", "reasoning_effort", "parameter_sources")
        if key in model_metadata
    }
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
        metadata={"model_parameters": parameters} if parameters else {},
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
        last_evaluation = (
            self.registry.last_evaluation(request.task_id)
            if environment.current_task.allow_previous_attempt_context
            else None
        )
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
            model_metadata={
                **({"name": configured_model} if configured_model is not None else {}),
                **model_parameter_metadata(
                    agent, model=configured_model, definition=request.agent_runtime
                ),
            },
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


def _translate_private_paths(value: Any, source: str, target: str) -> Any:
    """Translate explicit environment option paths for controller-only mounts."""
    if isinstance(value, dict):
        return {
            key: _translate_private_paths(item, source, target)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_translate_private_paths(item, source, target) for item in value]
    if isinstance(value, str) and (
        value == source or value.startswith(source.rstrip("/") + "/")
    ):
        return target + value[len(source) :]
    return value


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
        build_args: Mapping[str, str] | None = None,
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
                    *(
                        argument
                        for name, value in (build_args or {}).items()
                        for argument in ("--build-arg", f"{name}={value}")
                    ),
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
            f"{volume}-state",
            output_limit=32_000,
            check=False,
        )

    async def _inspect_container(self, container: str) -> dict[str, Any] | None:
        code, output = await _command(
            self.docker_executable,
            "inspect",
            "--type",
            "container",
            container,
            output_limit=1_000_000,
            check=False,
        )
        if code:
            if (
                "no such object" in output.lower()
                or "no such container" in output.lower()
            ):
                return None
            raise DockerInfrastructureError(
                f"Could not inspect Docker container {container} (exit code {code})"
            )
        try:
            (details,) = json.loads(output)
            if not isinstance(details, dict) or not details.get("Id"):
                raise ValueError("missing container ID")
            return details
        except (ValueError, TypeError) as exc:
            # Docker inspect includes credentials; never include its raw output.
            raise DockerInfrastructureError(
                f"Docker returned invalid container metadata for {container}"
            ) from exc

    @staticmethod
    def _configuration_key(arguments: list[str]) -> str:
        """Identify immutable launch settings without storing environment values."""
        immutable = []
        iterator = iter(arguments)
        for argument in iterator:
            if argument in ("--cpus", "--memory", "--pids-limit"):
                next(iterator)
            else:
                immutable.append(argument)
        return hashlib.sha256(json.dumps(immutable).encode()).hexdigest()

    @staticmethod
    def _environment_matches(details: dict[str, Any], arguments: list[str]) -> bool:
        existing = dict(
            entry.split("=", 1) for entry in details.get("Config", {}).get("Env", [])
        )
        for index, argument in enumerate(arguments):
            if argument != "--env":
                continue
            name, separator, value = arguments[index + 1].partition("=")
            if existing.get(name) != (value if separator else os.environ[name]):
                return False
        return True

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

    async def _export_checkpoint(self, shard: Path, container: str) -> None:
        """Copy a stopped controller's checkpoint before releasing its volume."""
        try:
            # Also handles cancellation or a host process dying during a trial.
            # Copying a live database and its WAL separately is unsafe.
            await _command(
                self.docker_executable,
                "stop",
                "--time",
                "10",
                container,
                output_limit=32_000,
            )
            with tempfile.TemporaryDirectory(
                prefix=".corral-checkpoint-", dir=shard.parent
            ) as directory:
                staged = Path(directory)
                await _command(
                    self.docker_executable,
                    "cp",
                    f"{container}:/corral-state/.",
                    str(staged),
                    output_limit=32_000,
                )
                shutil.copytree(staged, shard, dirs_exist_ok=True)
                # A clean close can remove these in the container. Do not replay
                # stale host WAL contents over the newly exported database.
                for suffix in ("-wal", "-shm"):
                    name = f"commits.sqlite3{suffix}"
                    if not (staged / name).exists():
                        (shard / name).unlink(missing_ok=True)
            (shard / "checkpoint-pending.json").unlink()
        except Exception as exc:
            raise DockerInfrastructureError(
                f"Could not export checkpoint from {container}; its container and "
                "volumes are preserved. Retry to recover the checkpoint before "
                "starting another trial."
            ) from exc

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
        pending_checkpoint = shard / "checkpoint-pending.json"
        if pending_checkpoint.exists():
            pending = json.loads(pending_checkpoint.read_text(encoding="utf-8"))
            await self._export_checkpoint(shard, pending["container_id"])
        request_path = shard / "request.json"
        result_path = shard / "result.json"
        result_path.unlink(missing_ok=True)
        recovered = any(
            (shard / directory / "latest.json").is_file()
            for directory in ("workspace-snapshots", "snapshots")
        )
        container_request = asdict(request)
        private_mounts = []
        for index, source in enumerate(docker.private_directories):
            if not Path(source).is_dir():
                raise ValueError(
                    f"Private controller directory does not exist: {source}"
                )
            target = f"/corral-private/{index}"
            private_mounts.extend(
                ("--mount", f"type=bind,src={source},dst={target},readonly")
            )
            definition = container_request.get("environment_runtime")
            if definition:
                definition["options"] = _translate_private_paths(
                    definition["options"], source, target
                )
        _atomic_json(request_path, container_request)
        # Checkpoints are private to the trusted controller.
        shard.chmod(0o700)

        container_name, volume_name = self._names(request.execution_id)
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
            "--init",
            "--user",
            "0:0",
            "--security-opt",
            "no-new-privileges",
            # docker-default denies mounts even with SYS_ADMIN; the trusted
            # bootstrap must construct the worker filesystem before dropping UID.
            "--security-opt",
            "apparmor=unconfined",
            "--cap-drop",
            "ALL",
            "--cap-add",
            "SETUID",
            "--cap-add",
            "SETGID",
            "--cap-add",
            "CHOWN",
            "--cap-add",
            "DAC_OVERRIDE",
            "--cap-add",
            "KILL",
            "--cap-add",
            "SYS_ADMIN",
            "--cap-add",
            "SYS_CHROOT",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,noexec,mode=0700,size=256m",
            "--mount",
            f"type=volume,src={volume_name},dst=/workspace",
            "--mount",
            f"type=volume,src={volume_name}-state,dst=/corral-state",
            "--env",
            "HOME=/tmp",
            "--env",
            f"CORRAL_HOST_UID={os.getuid()}",
            "--env",
            f"CORRAL_HOST_GID={os.getgid()}",
        ]
        for key, value in labels.items():
            create_arguments.extend(("--label", f"{key}={value}"))
        create_arguments.extend(private_mounts)
        for name in docker.environment_allowlist:
            if name in os.environ:
                create_arguments.extend(("--env", name))
        create_arguments.extend(
            (
                docker.immutable_image,
                "python",
                "-c",
                _DOCKER_BOOTSTRAP,
            )
        )
        configuration_key = self._configuration_key(create_arguments)
        create_arguments[2:2] = [
            "--label",
            f"corral.launcher_config={configuration_key}",
        ]

        existing = await self._inspect_container(container_name)
        reused = False
        if existing is not None:
            existing_labels = existing.get("Config", {}).get("Labels") or {}
            for key in (
                "corral.execution_id",
                "corral.benchmark_run_id",
                "corral.task_id",
                "corral.trial_index",
            ):
                if existing_labels.get(key) != labels[key]:
                    raise DockerInfrastructureError(
                        f"Container {container_name} belongs to another execution; "
                        "it has been left untouched."
                    )
            if existing.get("State", {}).get("Running"):
                raise DockerInfrastructureError(
                    f"Container {container_name} is already running; "
                    "stop the active run before resuming it."
                )
            reused = (
                existing_labels.get("corral.launcher_config") == configuration_key
                and existing.get("Image") == docker.image_digest
                and not existing.get("State", {}).get("Dead")
                and self._environment_matches(existing, create_arguments)
            )
        if not reused:
            # Any pending checkpoint was exported above. The durable host shard
            # is authoritative when an image, mount, or credential has changed.
            await self._discard_stale(container_name, volume_name)

        container_id = ""
        failed = True
        output = ""
        try:
            if reused:
                assert existing is not None
                container_id = existing["Id"]
                # Resource limits can change without replacing the container.
                # Keep Docker's default swap allowance (twice the memory limit)
                # when increasing memory beyond the previous swap limit.
                unit = docker.memory[-1].lower()
                memory_bytes = (
                    int(docker.memory[:-1]) * 1024 ** "bkmg".index(unit)
                    if unit in "bkmg"
                    else int(docker.memory)
                )
                await _command(
                    self.docker_executable,
                    "update",
                    "--cpus",
                    str(docker.cpus),
                    "--memory",
                    docker.memory,
                    "--memory-swap",
                    str(2 * memory_bytes),
                    "--pids-limit",
                    str(docker.pids_limit),
                    container_id,
                    output_limit=32_000,
                )
            else:
                for name in (volume_name, f"{volume_name}-state"):
                    await _command(*volume_arguments, name, output_limit=32_000)
                _code, container_id = await _command(
                    *create_arguments, output_limit=32_000
                )
                container_id = container_id.splitlines()[-1].strip()
            sandbox_metadata = {
                "benchmark_run_id": request.benchmark_run_id,
                "container_id": container_id,
                "container_name": container_name,
                "permissions_policy": "workspace-root-v1",
                "cpus": docker.cpus,
                "execution_id": request.execution_id,
                "image": docker.image,
                "image_digest": docker.image_digest,
                "memory": docker.memory,
                "network": docker.network,
                "pids_limit": docker.pids_limit,
                "recovered": recovered,
                "reused": reused,
                "retention": docker.retention,
                "runtime_protocol_version": docker.runtime_protocol_version,
                "task_id": request.task_id,
                "trial_index": request.trial_index,
                "volume_name": volume_name,
                "state_volume_name": f"{volume_name}-state",
            }
            self._write_sandbox_metadata(shard, sandbox_metadata)
            _atomic_json(
                shard / "checkpoint-import.json",
                {
                    "missing_sidecars": [
                        name
                        for name in ("commits.sqlite3-wal", "commits.sqlite3-shm")
                        if not (shard / name).exists()
                    ]
                },
            )
            await _command(
                self.docker_executable,
                "cp",
                f"{shard}/.",
                f"{container_id}:/corral-state",
                output_limit=32_000,
            )
            # Survives host crashes and failed exports. A retry must recover this
            # checkpoint before discarding the previous container or volumes.
            _atomic_json(pending_checkpoint, {"container_id": container_id})
            try:
                _code, output = await _command(
                    self.docker_executable,
                    "start",
                    "--attach",
                    container_id,
                    output_limit=self.output_limit,
                    check=False,
                )
                details = await self._inspect_container(container_id)
                if details is None:
                    raise DockerInfrastructureError(
                        "Task container disappeared before checkpoint export"
                    )
                state = details.get("State", {})
                sandbox_metadata.update(
                    oom_killed=bool(state.get("OOMKilled")),
                    exit_code=state.get("ExitCode"),
                )
                if state.get("OOMKilled"):
                    raise DockerInfrastructureError(
                        f"Task {request.execution_id} was killed by an out-of-memory event "
                        f"(memory limit: {docker.memory}). Increase --sandbox-memory and the Docker VM's "
                        "available memory, or reduce --max-parallel."
                    )
                if _code or state.get("ExitCode", 0):
                    raise DockerInfrastructureError(
                        f"Task container exited with code {state.get('ExitCode') or _code}; "
                        f"bounded output: {output}"
                    )
            finally:
                await asyncio.shield(self._export_checkpoint(shard, container_id))
                self._write_sandbox_metadata(shard, sandbox_metadata)
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
            if result.execution_id != request.execution_id:
                raise DockerInfrastructureError(
                    "Task container returned a result for another execution"
                )
            snapshot_revision = self._latest_revision(shard)
            sandbox_metadata.update(
                final_commit_hash=result.commit_hash,
                snapshot_revision=snapshot_revision,
                status=result.status,
            )
            self._write_sandbox_metadata(shard, sandbox_metadata)
            failed = result.status == "failed"
            return replace(
                result,
                metadata={
                    **result.metadata,
                    "container_id": container_id,
                    "image_digest": docker.image_digest,
                    "recovered": recovered,
                    "reused": reused,
                    "runtime_protocol_version": docker.runtime_protocol_version,
                    "snapshot_revision": snapshot_revision,
                },
            )
        finally:
            if not pending_checkpoint.exists():
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


__all__ = [
    "DockerInfrastructureError",
    "DockerTaskLauncher",
    "LocalTaskLauncher",
    "TaskLauncher",
    "state_ref",
]
