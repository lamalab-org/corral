"""Import a historical checkpoint without opening the source for writing."""

from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from corral.core.actors import ActorRef
from corral.core.commit import CommitRequest
from corral.core.events import ExecutionForked
from corral.persistence.artifacts import LocalArtifactStore
from corral.persistence.base import CommitConflictError
from corral.persistence.sqlite import SQLiteCommitStore

if TYPE_CHECKING:
    from corral.orchestration.models import ContinuationSource, RunTaskInput
    from corral.persistence.base import CommitStore
    from corral.persistence.workspace import WorkspaceManager


def _copy_database(source: Path, target: Path) -> None:
    # The source is a finished trial. Copy its WAL too, since the final commit
    # need not have been checkpointed. Never open/migrate/chmod the original DB.
    files = [source]
    wal = source.with_name(source.name + "-wal")
    if wal.exists():
        files.append(wal)
    before = [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    for path in files:
        shutil.copyfile(
            path, target.with_name(target.name + path.name[len(source.name) :])
        )
    after = [(p.stat().st_size, p.stat().st_mtime_ns) for p in files]
    if before != after:
        raise ValueError("source trial changed while reading its checkpoint")


async def read_checkpoint(source: ContinuationSource) -> ExecutionForked:
    """Read and validate an exact point in a finished trial's branch history."""
    root = Path(source.trial).expanduser().resolve()
    database = root / "commits.sqlite3"
    if not database.is_file():
        raise FileNotFoundError(f"source trial has no commits.sqlite3: {root}")
    metadata = (
        json.loads((root / "metadata.json").read_text())
        if (root / "metadata.json").is_file()
        else {}
    )
    execution_id = source.execution_id or metadata.get("execution_id")
    if not execution_id:
        raise ValueError("source execution ID is required when metadata.json is absent")
    with tempfile.TemporaryDirectory(prefix="corral-source-") as temporary:
        copied = Path(temporary) / "commits.sqlite3"
        await asyncio.to_thread(_copy_database, database, copied)
        async with SQLiteCommitStore(copied, execution_id) as store:
            final = await store.materialize("main")
            if not final.is_terminal:
                raise ValueError("continuation source trial must be finished")
            state = await store.materialize(source.branch_id, source.commit_hash)
            configured = state.runtime.metadata.get("checkpoint_configured", False)
            async for commit in store.iter_commits(
                source.branch_id, through_hash=source.commit_hash
            ):
                configured = configured or commit.event.type == "task.configured"
    selected = source.agent_run_id
    if selected is None:
        roots = [
            run.run_id for run in state.agent_runs.values() if run.parent_run_id is None
        ]
        if len(roots) > 1:
            raise ValueError("checkpoint has several root agents; specify agent_run_id")
        selected = roots[0] if roots else None
    if selected is not None and selected not in state.agent_runs:
        raise ValueError("selected agent does not exist at this commit")
    if any(
        run.run_id != selected and run.status in {"created", "running"}
        for run in state.agent_runs.values()
    ):
        raise ValueError(
            "checkpoint has active subagents; live subagent continuation is unsupported"
        )
    jobs = state.environment.values.get("jobs", {})
    if isinstance(jobs, Mapping) and any(
        isinstance(job, Mapping)
        and job.get("status") in {"pending", "queued", "running"}
        for job in jobs.values()
    ):
        raise ValueError("checkpoint has live background jobs that cannot be cloned")
    return ExecutionForked(
        source=state,
        source_run_id=selected,
        source_trial=str(root),
        configured=bool(configured),
        message=source.message,
    )


def _blob_refs(value):
    if isinstance(value, Mapping):
        for key, child in value.items():
            if (
                key == "blob_ref"
                and isinstance(child, str)
                and child.startswith("sha256:")
            ):
                yield child
            else:
                yield from _blob_refs(child)
    elif isinstance(value, tuple | list):
        for child in value:
            yield from _blob_refs(child)


def validate_destination(request: RunTaskInput) -> None:
    """Reject reuse of the parent's identity before a destination is opened."""
    source = request.continuation
    assert source is not None
    metadata_path = Path(source.trial).expanduser() / "metadata.json"
    metadata = json.loads(metadata_path.read_text()) if metadata_path.is_file() else {}
    if request.execution_id in {source.execution_id, metadata.get("execution_id")}:
        raise ValueError("continuation must use a new execution ID")


async def import_checkpoint(
    request: RunTaskInput,
    store: CommitStore,
    manager: WorkspaceManager,
) -> None:
    """Copy referenced blobs, then atomically publish the child's initial commit."""
    source = request.continuation
    assert source is not None
    validate_destination(request)
    target = store.for_execution(request.execution_id)
    head = await target.head("main")
    if head is not None:
        # Retry/resume must work even after the source was moved or deleted.
        first = None
        async for next_commit in target.iter_commits("main"):
            first = next_commit
            break
        if first is None or not isinstance(first.event, ExecutionForked):
            raise CommitConflictError("destination is not a continuation trial")
        event = first.event
        if (
            event.source.task.metadata.get("id") != request.task_id
            or event.source.through_commit_hash != source.commit_hash
            or event.source.branch_id != source.branch_id
            or event.source_trial != str(Path(source.trial).expanduser().resolve())
            or event.message != source.message
            or (
                source.execution_id is not None
                and source.execution_id != event.source.execution_id
            )
            or (
                source.agent_run_id is not None
                and source.agent_run_id != event.source_run_id
            )
        ):
            raise CommitConflictError(
                "destination already continues a different checkpoint"
            )
        return
    event = await read_checkpoint(source)
    if event.source.execution_id == request.execution_id:
        raise ValueError("continuation must use a new execution ID")
    if event.source.task.metadata.get("id") != request.task_id:
        raise ValueError("continuation task does not match source task")
    model = dict(event.source.task.model)
    if request.agent_runtime is not None and request.agent_runtime.model is not None:
        model["name"] = request.agent_runtime.model
    if request.model is not None:
        model["name"] = request.model
    if request.agent_runtime is not None:
        if request.agent_runtime.temperature is not None:
            model["temperature"] = request.agent_runtime.temperature
        if request.agent_runtime.reasoning_effort is not None:
            model["reasoning_effort"] = request.agent_runtime.reasoning_effort
    scaffold = {
        "sandbox": request.sandbox.mode,
        **(
            {"max_iterations": request.max_iterations}
            if request.max_iterations is not None
            else {}
        ),
    }
    if request.sandbox.docker is not None:
        docker = request.sandbox.docker
        scaffold.update(
            sandbox_image_digest=docker.image_digest,
            sandbox_network=docker.network,
            sandbox_resources={
                "cpus": docker.cpus,
                "memory": docker.memory,
                "pids_limit": docker.pids_limit,
            },
            sandbox_runtime_protocol=docker.runtime_protocol_version,
        )
    event = ExecutionForked.model_validate(
        {
            **event.model_dump(mode="python"),
            "model": model,
            "scaffold": scaffold,
        }
    )
    root = Path(event.source_trial) / "artifacts"
    refs = set(_blob_refs(event.source.model_dump(mode="python")))
    if refs:
        if not root.is_dir():
            raise FileNotFoundError(f"source artifacts are missing: {root}")
        artifacts = LocalArtifactStore(root)
        with tempfile.TemporaryDirectory(prefix="corral-blobs-") as temporary:
            for ref in sorted(refs):
                path = Path(temporary) / "blob"
                await artifacts.materialize(ref, path)
                imported = await manager.artifact_store.put_file(path)
                if imported.blob_ref != ref:
                    raise ValueError("imported artifact hash mismatch")
    committed = await target.append(
        CommitRequest(
            request_id="execution:forked",
            execution_id=request.execution_id,
            branch_id="main",
            based_on_hash=None,
            author=ActorRef(
                kind="runtime",
                actor_id="corral",
                run_id=f"runtime:{request.execution_id}",
            ),
            occurred_at=datetime.fromisoformat(request.started_at),
            event=event,
        )
    )
    if committed.event != event:
        raise CommitConflictError(
            "destination was initialized with a different checkpoint"
        )
