"""Execution-sharded durable state for isolated benchmark trials."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Self

from corral.persistence.sqlite import SQLiteCommitStore
from corral.persistence.workspace import WorkspaceManager

if TYPE_CHECKING:
    from corral.core.actors import ActorRef
    from corral.core.commit import Commit, CommitRequest
    from corral.core.state import ExecutionState


def execution_shard_name(execution_id: str) -> str:
    """Map an opaque execution ID to a traversal-safe stable directory name."""
    if not execution_id:
        raise ValueError("execution_id cannot be empty")
    return hashlib.sha256(execution_id.encode("utf-8")).hexdigest()


def _descriptive_component(prefix: str, value: str) -> str:
    """Return a readable, traversal-safe component with collision protection."""
    component = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    ).strip("-")
    component = component[:120] or "unnamed"
    if component != value:
        digest = hashlib.sha256(value.encode("utf-8")).hexdigest()[:8]
        component = f"{component}--{digest}"
    return f"{prefix}-{component}"


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
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


def _ensure_metadata(
    path: Path,
    execution_id: str,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    metadata_path = path / "metadata.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("execution_id") != execution_id:
            raise RuntimeError("execution shard metadata does not match its ID")
        if extra:
            updated = {**metadata, **extra}
            if updated != metadata:
                _atomic_json(metadata_path, updated)
        return
    _atomic_json(
        metadata_path,
        {
            "execution_id": execution_id,
            "schema_version": 1,
            "created_at": datetime.now(timezone.utc).isoformat(),
            **dict(extra or {}),
        },
    )


class ExecutionShard:
    """One execution-bound CommitStore plus its artifact/snapshot directories."""

    def __init__(
        self,
        root: Path,
        execution_id: str,
        *,
        snapshot_interval: int,
        workspace_snapshots_name: str = "snapshots",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.root = root
        self.execution_id = execution_id
        self.root.mkdir(parents=True, exist_ok=True)
        self.artifacts_path = self.root / "artifacts"
        self.workspace_snapshots_path = self.root / workspace_snapshots_name
        # Compatibility alias for callers that used this attribute before
        # workspace and state snapshots were separated on disk.
        self.snapshots_path = self.workspace_snapshots_path
        self.state_snapshots_path = self.root / "state-snapshots"
        self.recovery_path = self.root / "recovery"
        self.artifacts_path.mkdir(exist_ok=True)
        self.workspace_snapshots_path.mkdir(exist_ok=True)
        self.state_snapshots_path.mkdir(exist_ok=True)
        self.recovery_path.mkdir(exist_ok=True)
        _ensure_metadata(self.root, execution_id, extra=metadata)
        self.workspace_manager = WorkspaceManager(
            artifact_root=self.artifacts_path,
            snapshot_root=self.workspace_snapshots_path,
        )
        self._store = SQLiteCommitStore(
            self.root / "commits.sqlite3",
            execution_id=execution_id,
            snapshot_interval=snapshot_interval,
            state_snapshot_root=self.state_snapshots_path,
        )
        self._view = self._store.for_execution(execution_id)

    def close(self) -> None:
        self._store.close()

    def bind(
        self,
        author: ActorRef,
        *,
        branch_id: str = "main",
        execution_id: str | None = None,
    ):
        return self._view.bind(author, branch_id=branch_id, execution_id=execution_id)

    async def append(self, request: CommitRequest) -> Commit:
        return await self._view.append(request)

    async def head(self, branch_id: str) -> Commit | None:
        return await self._view.head(branch_id)

    async def get_commit(self, commit_hash: str) -> Commit:
        return await self._view.get_commit(commit_hash)

    async def materialize(
        self, branch_id: str, at_hash: str | None = None
    ) -> ExecutionState:
        return await self._view.materialize(branch_id, at_hash)

    async def create_branch(
        self,
        *,
        branch_id: str,
        from_hash: str,
        execution_id: str | None = None,
    ) -> None:
        await self._view.create_branch(
            branch_id=branch_id,
            from_hash=from_hash,
            execution_id=execution_id,
        )

    def iter_commits(
        self,
        branch_id: str | None = None,
        *,
        after_sequence: int = -1,
        through_hash: str | None = None,
    ):
        return self._view.iter_commits(
            branch_id,
            after_sequence=after_sequence,
            through_hash=through_hash,
        )

    async def trace_commits(
        self, run_id: str, *, branch_id: str | None = None
    ) -> tuple[Commit, ...]:
        return await self._view.trace_commits(run_id, branch_id=branch_id)

    def for_execution(self, execution_id: str) -> ExecutionShard:
        if execution_id != self.execution_id:
            raise ValueError("execution-bound shard cannot change execution")
        return self


class ShardedCommitStore:
    """Route every execution to an independent host-owned SQLite shard.

    The top-level object intentionally has no unscoped append/read operations.
    Callers select a shard with :meth:`for_execution`, which prevents unrelated
    task containers from sharing a SQLite writer or artifact directory.
    """

    execution_id = None

    def __init__(
        self,
        root: str | Path = ".corral",
        *,
        snapshot_interval: int = 50,
        benchmark_run_id: str | None = None,
    ):
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be at least one")
        self.root = Path(root).expanduser().resolve()
        self.benchmark_run_id = benchmark_run_id
        self.executions_root = (
            self.root if benchmark_run_id is not None else self.root / "executions"
        )
        self.executions_root.mkdir(parents=True, exist_ok=True)
        self.snapshot_interval = snapshot_interval
        self._shards: dict[str, ExecutionShard] = {}
        self._lock = threading.RLock()
        self._closed = False

    def execution_dir(self, execution_id: str) -> Path:
        """Return and initialize the host checkpoint directory for an execution."""
        metadata: dict[str, Any] = {}
        if self.benchmark_run_id is None:
            path = self.executions_root / execution_shard_name(execution_id)
            workspace_snapshots_name = "snapshots"
        else:
            prefix = f"{self.benchmark_run_id}:"
            if not execution_id.startswith(prefix):
                raise ValueError(
                    "benchmark execution_id must start with its benchmark_run_id"
                )
            suffix = execution_id[len(prefix) :]
            task_id, separator, raw_trial_index = suffix.rpartition(":")
            if not separator or not task_id:
                raise ValueError(
                    "benchmark execution_id must end with :<task_id>:<trial_index>"
                )
            try:
                trial_index = int(raw_trial_index)
            except ValueError as exc:
                raise ValueError("benchmark trial index must be an integer") from exc
            if trial_index < 0:
                raise ValueError("benchmark trial index cannot be negative")
            path = (
                self.executions_root
                / _descriptive_component("task", task_id)
                / f"k-{trial_index + 1}"
            )
            workspace_snapshots_name = "workspace-snapshots"
            metadata = {
                "benchmark_run_id": self.benchmark_run_id,
                "task_id": task_id,
                "trial_index": trial_index,
                "k": trial_index + 1,
            }
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o2770)
        for directory_name in (
            "artifacts",
            "recovery",
            workspace_snapshots_name,
            "state-snapshots",
        ):
            directory = path / directory_name
            directory.mkdir(mode=0o2770, exist_ok=True)
            directory.chmod(0o2770)
        commits = path / "commits.sqlite3"
        commits.touch(mode=0o660, exist_ok=True)
        _ensure_metadata(path, execution_id, extra=metadata)
        return path

    def _shard_metadata(self, execution_id: str) -> dict[str, Any]:
        if self.benchmark_run_id is None:
            return {}
        suffix = execution_id[len(f"{self.benchmark_run_id}:") :]
        task_id, _, raw_trial_index = suffix.rpartition(":")
        trial_index = int(raw_trial_index)
        return {
            "benchmark_run_id": self.benchmark_run_id,
            "task_id": task_id,
            "trial_index": trial_index,
            "k": trial_index + 1,
        }

    def for_execution(self, execution_id: str) -> ExecutionShard:
        if not execution_id:
            raise ValueError("execution_id cannot be empty")
        with self._lock:
            if self._closed:
                raise RuntimeError("ShardedCommitStore is closed")
            shard = self._shards.get(execution_id)
            if shard is None:
                shard = ExecutionShard(
                    self.execution_dir(execution_id),
                    execution_id,
                    snapshot_interval=self.snapshot_interval,
                    workspace_snapshots_name=(
                        "workspace-snapshots"
                        if self.benchmark_run_id is not None
                        else "snapshots"
                    ),
                    metadata=self._shard_metadata(execution_id),
                )
                self._shards[execution_id] = shard
            return shard

    def workspace_manager(self, execution_id: str) -> WorkspaceManager:
        return self.for_execution(execution_id).workspace_manager

    def close_execution(self, execution_id: str) -> None:
        """Close a host connection before a container opens the same shard."""
        with self._lock:
            shard = self._shards.pop(execution_id, None)
        if shard is not None:
            shard.close()

    def close(self) -> None:
        with self._lock:
            shards = tuple(self._shards.values())
            self._shards.clear()
            self._closed = True
        for shard in shards:
            shard.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *args: object) -> None:
        del args
        self.close()


__all__ = ["ExecutionShard", "ShardedCommitStore", "execution_shard_name"]
