"""Transactional SQLite implementation of the authored commit ledger."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

import aiosqlite
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import create_async_engine
from tenacity import retry, retry_if_exception, stop_after_delay, wait_fixed

from corral.core.commit import Commit, CommitRequest, canonical_json
from corral.core.events import (
    AgentCompleted,
    AgentSpawned,
    AgentStarted,
    AgentStateUpdated,
    AgentTurnRecorded,
    ContextImported,
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionStarted,
    ParallelGroupCompleted,
    SubmissionAccepted,
    TaskConfigured,
    ToolCompleted,
    ToolFailed,
    ToolStarted,
)
from corral.core.reducer import EventReducer, ReducerError
from corral.core.state import ExecutionState
from corral.persistence._sqlite_schema import (
    branch_heads,
    commits,
    idempotency_keys,
    metadata,
    snapshots,
)
from corral.persistence.base import (
    AuthorPermissionError,
    CommitConflictError,
    CommitIntegrityError,
    CommitNotFoundError,
)

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from sqlalchemy.engine import Connection, RowMapping
    from sqlalchemy.ext.asyncio import AsyncConnection
    from sqlalchemy.sql.elements import ColumnElement

    from corral.core.actors import ActorRef


def _configure_connection(connection: Any, _record: Any) -> None:
    # Let SQLAlchemy control BEGIN on every supported Python version.
    connection.isolation_level = None
    cursor = connection.cursor()
    cursor.execute("PRAGMA foreign_keys = ON")
    cursor.execute("PRAGMA synchronous = FULL")
    cursor.close()
    connection.run_async(_enable_wal)


@retry(
    retry=retry_if_exception(
        lambda exc: isinstance(exc, aiosqlite.OperationalError)
        and str(exc) == "database is locked"
    ),
    wait=wait_fixed(0.05),
    stop=stop_after_delay(5),
    reraise=True,
)
async def _enable_wal(connection: aiosqlite.Connection) -> None:
    # Concurrent first connections can race the journal-mode change before
    # BEGIN IMMEDIATE can serialize them. Retry without blocking the event loop.
    async with connection.execute("PRAGMA journal_mode = WAL") as cursor:
        await cursor.fetchone()


def _begin_transaction(connection: Connection) -> None:
    connection.exec_driver_sql(
        "BEGIN IMMEDIATE"
        if connection.get_execution_options().get("sqlite_write")
        else "BEGIN"
    )


def _trace_filter(run_id: str) -> ColumnElement[bool]:
    return sa.or_(
        commits.c.author_run_id == run_id,
        commits.c.requested_by_run_id == run_id,
        sa.func.json_extract(commits.c.event_json, "$.agent_run_id") == run_id,
        sa.func.json_extract(commits.c.event_json, "$.child_run_id") == run_id,
    )


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


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


_AUTHOR_KINDS: dict[type[Any], frozenset[str]] = {
    ExecutionStarted: frozenset({"runtime"}),
    TaskConfigured: frozenset({"runtime"}),
    AgentStarted: frozenset({"runtime"}),
    AgentSpawned: frozenset({"agent"}),
    AgentTurnRecorded: frozenset({"agent"}),
    AgentStateUpdated: frozenset({"agent"}),
    ToolStarted: frozenset({"runtime"}),
    ToolCompleted: frozenset({"tool"}),
    ToolFailed: frozenset({"tool", "runtime"}),
    ParallelGroupCompleted: frozenset({"runtime"}),
    ContextImported: frozenset({"agent"}),
    AgentCompleted: frozenset({"agent", "runtime"}),
    SubmissionAccepted: frozenset({"runtime"}),
    ExecutionFailed: frozenset({"runtime"}),
    ExecutionCompleted: frozenset({"runtime"}),
}


class _BoundSQLiteCommitStore:
    def __init__(
        self,
        store: SQLiteCommitStore,
        author: ActorRef,
        branch_id: str,
        execution_id: str | None,
    ) -> None:
        self._store = store
        self._author = author
        self._branch_id = branch_id
        self._execution_id = execution_id

    @property
    def author(self) -> ActorRef:
        return self._author

    @property
    def branch_id(self) -> str:
        return self._branch_id

    async def append(self, request: CommitRequest) -> Commit:
        if request.author != self._author:
            raise AuthorPermissionError("a bound commit session cannot change author")
        if request.branch_id != self._branch_id:
            raise AuthorPermissionError("a bound commit session cannot change branch")
        if (
            self._execution_id is not None
            and request.execution_id is not None
            and request.execution_id != self._execution_id
        ):
            raise AuthorPermissionError(
                "a bound commit session cannot change execution"
            )
        return await self._store._append(
            request,
            authenticated_author=self._author,
            bound_execution_id=self._execution_id,
        )


class _ExecutionSQLiteCommitStore:
    """A lightweight view that binds shared database access to one execution."""

    def __init__(self, store: SQLiteCommitStore, execution_id: str) -> None:
        self._store = store
        self.execution_id = execution_id

    def bind(
        self,
        author: ActorRef,
        *,
        branch_id: str = "main",
        execution_id: str | None = None,
    ) -> _BoundSQLiteCommitStore:
        if execution_id is not None and execution_id != self.execution_id:
            raise ValueError("execution-bound store cannot change execution")
        return self._store.bind(
            author, branch_id=branch_id, execution_id=self.execution_id
        )

    async def append(self, request: CommitRequest) -> Commit:
        values = request.model_dump(mode="python")
        values["execution_id"] = self.execution_id
        return await self._store.append(CommitRequest.model_validate(values))

    async def head(self, branch_id: str) -> Commit | None:
        return await self._store.head(branch_id, execution_id=self.execution_id)

    async def get_commit(self, commit_hash: str) -> Commit:
        return await self._store.get_commit(commit_hash, execution_id=self.execution_id)

    async def materialize(
        self, branch_id: str, at_hash: str | None = None
    ) -> ExecutionState:
        return await self._store.materialize(
            branch_id, at_hash, execution_id=self.execution_id
        )

    async def create_branch(
        self,
        *,
        branch_id: str,
        from_hash: str,
        execution_id: str | None = None,
    ) -> None:
        if execution_id is not None and execution_id != self.execution_id:
            raise ValueError("execution-bound store cannot change execution")
        await self._store.create_branch(
            branch_id=branch_id,
            from_hash=from_hash,
            execution_id=self.execution_id,
        )

    async def iter_commits(
        self,
        branch_id: str | None = None,
        *,
        after_sequence: int = -1,
        through_hash: str | None = None,
    ) -> AsyncIterator[Commit]:
        async for commit in self._store.iter_commits(
            branch_id,
            after_sequence=after_sequence,
            through_hash=through_hash,
            execution_id=self.execution_id,
        ):
            yield commit

    async def trace_commits(
        self, run_id: str, *, branch_id: str | None = None
    ) -> tuple[Commit, ...]:
        return await self._store.trace_commits(
            run_id, branch_id=branch_id, execution_id=self.execution_id
        )

    def for_execution(self, execution_id: str) -> _ExecutionSQLiteCommitStore:
        if execution_id != self.execution_id:
            raise ValueError("execution-bound store cannot change execution")
        return self


class SQLiteCommitStore:
    """One SQLite database containing small commits and occasional snapshots."""

    def __init__(
        self,
        path: str | Path,
        execution_id: str | None = None,
        *,
        snapshot_interval: int = 50,
        state_snapshot_root: str | Path | None = None,
    ) -> None:
        if snapshot_interval < 1:
            raise ValueError("snapshot_interval must be at least 1")
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.execution_id = execution_id
        self.snapshot_interval = snapshot_interval
        self.state_snapshot_root = (
            None
            if state_snapshot_root is None
            else Path(state_snapshot_root).expanduser()
        )
        if self.state_snapshot_root is not None:
            self.state_snapshot_root.mkdir(parents=True, exist_ok=True)
        self._lock = asyncio.Lock()
        self._closed = False
        self._initialized = False
        self._engine = create_async_engine(
            sa.URL.create("sqlite+aiosqlite", database=str(self.path))
        )
        sa.event.listen(self._engine.sync_engine, "connect", _configure_connection)
        sa.event.listen(self._engine.sync_engine, "begin", _begin_transaction)
        self._reducer = EventReducer()
        self._projection_cache: dict[tuple[str, str], ExecutionState] = {}

    def _publish_state_snapshot(
        self,
        commit: Commit,
        state_json: str,
        *,
        terminal: bool,
    ) -> None:
        """Publish a human-readable projection beside the authoritative ledger."""
        if self.state_snapshot_root is None:
            return
        state_data = json.loads(state_json)
        filename = f"{commit.sequence:08d}-{commit.hash[:12]}.json"
        _atomic_json(self.state_snapshot_root / filename, state_data)
        _atomic_json(self.state_snapshot_root / "latest.json", state_data)
        if terminal:
            _atomic_json(self.state_snapshot_root / "final.json", state_data)

    @asynccontextmanager
    async def _transaction(
        self, *, write: bool = False
    ) -> AsyncIterator[AsyncConnection]:
        # Serialize this store's transactions and projection cache access. SQLite
        # also serializes independent writers via BEGIN IMMEDIATE below.
        async with self._lock:
            self._ensure_open()
            async with self._engine.connect() as connection:
                if not self._initialized:
                    await connection.execution_options(sqlite_write=True)
                    async with connection.begin():
                        await connection.run_sync(metadata.create_all)
                    self._initialized = True
                await connection.execution_options(sqlite_write=write)
                async with connection.begin():
                    yield connection

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("SQLiteCommitStore is closed")

    async def aclose(self) -> None:
        async with self._lock:
            await self._engine.dispose()
            self._projection_cache.clear()
            self._closed = True

    async def __aenter__(self) -> SQLiteCommitStore:  # noqa: PYI034
        async with self._transaction():
            pass
        return self

    async def __aexit__(self, *args: object) -> None:
        del args
        await self.aclose()

    def bind(
        self,
        author: ActorRef,
        *,
        branch_id: str = "main",
        execution_id: str | None = None,
    ) -> _BoundSQLiteCommitStore:
        return _BoundSQLiteCommitStore(
            self,
            author,
            branch_id,
            execution_id or self.execution_id,
        )

    def for_execution(self, execution_id: str) -> _ExecutionSQLiteCommitStore:
        if not execution_id:
            raise ValueError("execution_id cannot be empty")
        if self.execution_id is not None and self.execution_id != execution_id:
            raise ValueError("store is already bound to another execution")
        return _ExecutionSQLiteCommitStore(self, execution_id)

    def _execution(self, requested: str | None, bound: str | None = None) -> str:
        candidates = {value for value in (requested, bound, self.execution_id) if value}
        if len(candidates) > 1:
            raise ValueError("commit execution_id differs from the bound execution")
        if not candidates:
            raise ValueError(
                "execution_id is required on CommitRequest or SQLiteCommitStore"
            )
        return candidates.pop()

    @staticmethod
    def _correlation(event: Any) -> tuple[str | None, str | None, str | None]:
        return (
            getattr(event, "action_id", None),
            getattr(event, "invocation_id", None),
            getattr(event, "requested_by_run_id", None),
        )

    @staticmethod
    def _validate_author(request: CommitRequest, state: ExecutionState | None) -> None:
        allowed = _AUTHOR_KINDS.get(type(request.event))
        if allowed is None or request.author.kind not in allowed:
            raise AuthorPermissionError(
                f"{request.author.kind} actors cannot author {request.event.type}"
            )
        event = request.event
        author = request.author
        if (
            isinstance(
                event,
                AgentTurnRecorded
                | AgentStateUpdated
                | AgentSpawned
                | ContextImported
                | AgentCompleted,
            )
            and author.kind == "agent"
        ):
            run = None if state is None else state.agent_runs.get(author.run_id)
            if run is None or run.actor_id != author.actor_id:
                raise AuthorPermissionError("agent author is not a registered run")
            if run.parent_run_id != author.parent_run_id:
                raise AuthorPermissionError(
                    "agent parent identity does not match its run"
                )
            if (
                isinstance(event, AgentCompleted)
                and event.agent_run_id != author.run_id
            ):
                raise AuthorPermissionError("an agent cannot complete another run")
        if (
            isinstance(event, AgentSpawned)
            and event.context_cutoff_hash != request.based_on_hash
        ):
            raise CommitConflictError(
                "a child context cutoff must match the commit observed by its parent"
            )
        if isinstance(event, ToolCompleted):
            if event.invocation_id != author.run_id:
                raise AuthorPermissionError("tool author must match invocation_id")
            if state is not None:
                action = state.actions.get(event.action_id)
                if action is None or action.action.name != author.actor_id:
                    raise AuthorPermissionError("tool author does not match the action")
        if (
            isinstance(event, ToolFailed)
            and author.kind == "tool"
            and event.invocation_id != author.run_id
        ):
            raise AuthorPermissionError("tool author must match invocation_id")
        if (
            isinstance(event, ToolFailed)
            and author.kind == "tool"
            and state is not None
        ):
            action = state.actions.get(event.action_id)
            if action is None or action.action.name != author.actor_id:
                raise AuthorPermissionError("tool author does not match the action")

    @staticmethod
    def _row_to_commit(row: RowMapping) -> Commit:
        try:
            return Commit.model_validate(
                {
                    **{
                        name: row[name]
                        for name in Commit.model_fields
                        if name not in {"author", "event"}
                    },
                    "author": json.loads(row["author_json"]),
                    "event": json.loads(row["event_json"]),
                }
            )
        except Exception as exc:
            raise CommitIntegrityError(
                f"commit {row['hash']!r} failed schema or hash validation"
            ) from exc

    @staticmethod
    async def _history_hashes(
        connection: AsyncConnection, head_hash: str | None
    ) -> list[str]:
        hashes: list[str] = []
        current = head_hash
        seen: set[str] = set()
        while current is not None:
            if current in seen:
                raise CommitIntegrityError("commit parent history contains a cycle")
            seen.add(current)
            result = await connection.execute(
                sa.select(commits.c.hash, commits.c.parent_hash).where(
                    commits.c.hash == current
                )
            )
            row = result.mappings().first()
            if row is None:
                raise CommitIntegrityError(
                    f"history references missing commit {current!r}"
                )
            hashes.append(row["hash"])
            current = row["parent_hash"]
        return hashes

    async def _materialize(
        self,
        connection: AsyncConnection,
        execution_id: str,
        branch_id: str,
        at_hash: str | None,
    ) -> ExecutionState:
        result = await connection.execute(
            sa.select(branch_heads.c.head_hash).where(
                branch_heads.c.execution_id == execution_id,
                branch_heads.c.branch_id == branch_id,
            )
        )
        branch = result.mappings().first()
        if branch is None:
            raise CommitNotFoundError(f"branch {branch_id!r} was not found")
        selected = at_hash or branch["head_hash"]
        if selected is None:
            raise CommitNotFoundError(f"branch {branch_id!r} has no commits")
        history = await self._history_hashes(connection, branch["head_hash"])
        if selected not in history:
            raise CommitNotFoundError(
                f"commit {selected!r} does not belong to branch {branch_id!r}"
            )

        chain: list[Commit] = []
        state: ExecutionState | None = None
        current: str | None = selected
        while current is not None:
            snapshot = await connection.scalar(
                sa.select(snapshots.c.state_json).where(
                    snapshots.c.execution_id == execution_id,
                    snapshots.c.commit_hash == current,
                )
            )
            if snapshot is not None:
                try:
                    state = ExecutionState.model_validate_json(snapshot)
                except Exception as exc:
                    raise CommitIntegrityError(
                        f"snapshot at {current!r} is invalid"
                    ) from exc
                if state.through_commit_hash != current:
                    raise CommitIntegrityError(
                        "snapshot cutoff hash does not match its key"
                    )
                break
            result = await connection.execute(
                sa.select(commits).where(commits.c.hash == current)
            )
            row = result.mappings().first()
            if row is None:
                raise CommitIntegrityError(f"commit {current!r} was not found")
            commit = self._row_to_commit(row)
            if commit.execution_id != execution_id:
                raise CommitIntegrityError(
                    "branch history crosses execution identities"
                )
            chain.append(commit)
            current = commit.parent_hash

        for commit in reversed(chain):
            state = self._reducer.apply(state, commit)
        if state is None:  # pragma: no cover - selected guarantees data
            raise CommitIntegrityError("materialization produced no projection")
        if state.branch_id != branch_id:
            values = state.model_dump(mode="python")
            values["branch_id"] = branch_id
            state = ExecutionState.model_validate(values)
        return state

    async def append(self, request: CommitRequest) -> Commit:
        if request.author.kind != "runtime":
            raise AuthorPermissionError(
                "agent and tool commits require an author-bound store capability"
            )
        return await self._append(request, authenticated_author=request.author)

    async def _append(
        self,
        request: CommitRequest,
        *,
        authenticated_author: ActorRef,
        bound_execution_id: str | None = None,
    ) -> Commit:
        self._ensure_open()
        if request.author != authenticated_author:
            raise AuthorPermissionError(
                "request author does not match authenticated actor"
            )
        execution_id = self._execution(request.execution_id, bound_execution_id)
        async with self._transaction(write=True) as connection:
            commit, projected, state_json = await self._append_commit(
                connection, request, execution_id
            )
        # Publish only after the transaction has committed successfully.
        if state_json is not None:
            self._publish_state_snapshot(
                commit,
                state_json,
                terminal=isinstance(commit.event, ExecutionCompleted | ExecutionFailed),
            )
        if projected is not None:
            self._projection_cache[(execution_id, request.branch_id)] = projected
        return commit

    async def _append_commit(
        self,
        connection: AsyncConnection,
        request: CommitRequest,
        execution_id: str,
    ) -> tuple[Commit, ExecutionState | None, str | None]:
        result = await connection.execute(
            sa.select(commits)
            .join(idempotency_keys)
            .where(
                idempotency_keys.c.execution_id == execution_id,
                idempotency_keys.c.request_id == request.request_id,
            )
        )
        existing = result.mappings().first()
        if existing is not None:
            persisted = self._row_to_commit(existing)
            if persisted.author != request.author:
                raise AuthorPermissionError(
                    "idempotency key belongs to another authenticated author"
                )
            if persisted.branch_id != request.branch_id:
                raise CommitConflictError("idempotency key belongs to another branch")
            state_json = await connection.scalar(
                sa.select(snapshots.c.state_json).where(
                    snapshots.c.execution_id == execution_id,
                    snapshots.c.commit_hash == persisted.hash,
                )
            )
            return persisted, None, state_json

        result = await connection.execute(
            sa.select(branch_heads.c.head_hash).where(
                branch_heads.c.execution_id == execution_id,
                branch_heads.c.branch_id == request.branch_id,
            )
        )
        branch = result.mappings().first()
        if branch is None:
            any_commit = await connection.scalar(
                sa.select(commits.c.hash)
                .where(commits.c.execution_id == execution_id)
                .limit(1)
            )
            if any_commit is not None or not isinstance(
                request.event, ExecutionStarted
            ):
                raise CommitNotFoundError(
                    f"branch {request.branch_id!r} was not explicitly created"
                )
            await connection.execute(
                branch_heads.insert().values(
                    execution_id=execution_id, branch_id=request.branch_id
                )
            )
            head_hash = None
        else:
            head_hash = branch["head_hash"]

        history = set(await self._history_hashes(connection, head_hash))
        if request.based_on_hash is not None and request.based_on_hash not in history:
            raise CommitConflictError(
                "based_on_hash is not part of the selected branch history"
            )
        if head_hash is None and request.based_on_hash is not None:
            raise CommitConflictError("the first commit cannot be based on a hash")
        if head_hash is not None and request.based_on_hash is None:
            raise CommitConflictError("non-initial commits must record based_on_hash")
        if isinstance(request.event, ContextImported) and any(
            commit_hash not in history
            for commit_hash in request.event.source_commit_ids
        ):
            raise CommitConflictError(
                "imported source commits must belong to branch history"
            )
        if isinstance(request.event, ContextImported):
            source_hashes = set(
                await connection.scalars(
                    sa.select(commits.c.hash).where(
                        commits.c.execution_id == execution_id,
                        _trace_filter(request.event.source_run_id),
                    )
                )
            )
            if any(
                commit_hash not in source_hashes
                for commit_hash in request.event.source_commit_ids
            ):
                raise CommitConflictError(
                    "imported commits must belong to the selected source trace"
                )
        if (
            isinstance(request.event, AgentCompleted)
            and request.event.trace_head is not None
            and request.event.trace_head not in history
        ):
            raise CommitConflictError(
                "an agent trace head must belong to branch history"
            )

        state = (
            None
            if head_hash is None
            else await self._materialize(
                connection, execution_id, request.branch_id, head_hash
            )
        )
        self._validate_author(request, state)
        sequence = await connection.scalar(
            sa.select(sa.func.coalesce(sa.func.max(commits.c.sequence), -1) + 1).where(
                commits.c.execution_id == execution_id
            )
        )
        branch_sequence = await connection.scalar(
            sa.select(
                sa.func.coalesce(sa.func.max(commits.c.branch_sequence), -1) + 1
            ).where(
                commits.c.execution_id == execution_id,
                commits.c.branch_id == request.branch_id,
            )
        )
        recorded_at = _utc_now()
        occurred_at = request.occurred_at or recorded_at
        commit = Commit.create(
            execution_id=execution_id,
            branch_id=request.branch_id,
            sequence=sequence,
            branch_sequence=branch_sequence,
            parent_hash=head_hash,
            based_on_hash=request.based_on_hash,
            author=request.author,
            event=request.event,
            occurred_at=occurred_at,
            recorded_at=recorded_at,
        )
        try:
            projected = self._reducer.apply(state, commit)
        except ReducerError as exc:
            raise CommitConflictError(str(exc)) from exc

        action_id, invocation_id, requested_by_run_id = self._correlation(request.event)
        await connection.execute(
            commits.insert().values(
                **commit.model_dump(
                    exclude={"author", "event", "occurred_at", "recorded_at"}
                ),
                author_json=canonical_json(commit.author.model_dump(mode="json")),
                author_kind=commit.author.kind,
                author_id=commit.author.actor_id,
                author_run_id=commit.author.run_id,
                parent_run_id=commit.author.parent_run_id,
                event_type=commit.event.type,
                event_json=canonical_json(commit.event.model_dump(mode="json")),
                action_id=action_id,
                invocation_id=invocation_id,
                requested_by_run_id=requested_by_run_id,
                occurred_at=commit.occurred_at.isoformat(),
                recorded_at=commit.recorded_at.isoformat(),
            )
        )
        await connection.execute(
            idempotency_keys.insert().values(
                execution_id=execution_id,
                request_id=request.request_id,
                commit_hash=commit.hash,
            )
        )
        await connection.execute(
            branch_heads.update()
            .where(
                branch_heads.c.execution_id == execution_id,
                branch_heads.c.branch_id == request.branch_id,
            )
            .values(head_hash=commit.hash)
        )
        terminal = isinstance(request.event, ExecutionCompleted | ExecutionFailed)
        state_json = None
        if commit.sequence % self.snapshot_interval == 0 or terminal:
            state_json = projected.model_dump_json()
            # A new commit has a new hash; idempotent appends returned above.
            await connection.execute(
                snapshots.insert().values(
                    execution_id=execution_id,
                    commit_hash=commit.hash,
                    state_json=state_json,
                    created_at=recorded_at.isoformat(),
                )
            )
        return commit, projected, state_json

    async def head(
        self, branch_id: str, *, execution_id: str | None = None
    ) -> Commit | None:
        execution_id = self._execution(execution_id)
        async with self._transaction() as connection:
            result = await connection.execute(
                sa.select(commits)
                .join(branch_heads, commits.c.hash == branch_heads.c.head_hash)
                .where(
                    branch_heads.c.execution_id == execution_id,
                    branch_heads.c.branch_id == branch_id,
                )
            )
            row = result.mappings().first()
        return None if row is None else self._row_to_commit(row)

    async def get_commit(
        self, commit_hash: str, *, execution_id: str | None = None
    ) -> Commit:
        execution_id = self._execution(execution_id)
        async with self._transaction() as connection:
            result = await connection.execute(
                sa.select(commits).where(
                    commits.c.execution_id == execution_id,
                    commits.c.hash == commit_hash,
                )
            )
            row = result.mappings().first()
        if row is None:
            raise CommitNotFoundError(f"commit {commit_hash!r} was not found")
        return self._row_to_commit(row)

    async def materialize(
        self,
        branch_id: str,
        at_hash: str | None = None,
        *,
        execution_id: str | None = None,
    ) -> ExecutionState:
        execution_id = self._execution(execution_id)
        async with self._transaction() as connection:
            if at_hash is None:
                cached = self._projection_cache.get((execution_id, branch_id))
                head_hash = await connection.scalar(
                    sa.select(branch_heads.c.head_hash).where(
                        branch_heads.c.execution_id == execution_id,
                        branch_heads.c.branch_id == branch_id,
                    )
                )
                if cached is not None and cached.through_commit_hash == head_hash:
                    return cached
            state = await self._materialize(
                connection, execution_id, branch_id, at_hash
            )
            if at_hash is None:
                self._projection_cache[(execution_id, branch_id)] = state
            return state

    async def create_branch(
        self,
        *,
        branch_id: str,
        from_hash: str,
        execution_id: str | None = None,
    ) -> None:
        selected_execution = self._execution(execution_id)
        if not branch_id:
            raise ValueError("branch_id cannot be empty")
        async with self._transaction(write=True) as connection:
            source = await connection.scalar(
                sa.select(commits.c.execution_id).where(commits.c.hash == from_hash)
            )
            if source is None:
                raise CommitNotFoundError(f"commit {from_hash!r} was not found")
            if source != selected_execution:
                raise CommitConflictError("branch source belongs to another execution")
            existing = await connection.scalar(
                sa.select(branch_heads.c.branch_id).where(
                    branch_heads.c.execution_id == selected_execution,
                    branch_heads.c.branch_id == branch_id,
                )
            )
            if existing is not None:
                raise CommitConflictError(f"branch {branch_id!r} already exists")
            await connection.execute(
                branch_heads.insert().values(
                    execution_id=selected_execution,
                    branch_id=branch_id,
                    origin_hash=from_hash,
                    head_hash=from_hash,
                )
            )

    async def iter_commits(
        self,
        branch_id: str | None = None,
        *,
        after_sequence: int = -1,
        through_hash: str | None = None,
        execution_id: str | None = None,
    ) -> AsyncIterator[Commit]:
        execution_id = self._execution(execution_id)
        query = (
            sa.select(commits)
            .where(
                commits.c.execution_id == execution_id,
                commits.c.sequence > after_sequence,
            )
            .order_by(commits.c.sequence)
        )
        async with self._transaction() as connection:
            if branch_id is None:
                if through_hash is not None:
                    maximum = await connection.scalar(
                        sa.select(commits.c.sequence).where(
                            commits.c.execution_id == execution_id,
                            commits.c.hash == through_hash,
                        )
                    )
                    if maximum is None:
                        raise CommitNotFoundError(
                            f"commit {through_hash!r} was not found"
                        )
                    query = query.where(commits.c.sequence <= maximum)
            else:
                result = await connection.execute(
                    sa.select(branch_heads.c.head_hash).where(
                        branch_heads.c.execution_id == execution_id,
                        branch_heads.c.branch_id == branch_id,
                    )
                )
                branch = result.mappings().first()
                if branch is None:
                    raise CommitNotFoundError(f"branch {branch_id!r} was not found")
                head_hash = through_hash or branch["head_hash"]
                history = await self._history_hashes(connection, branch["head_hash"])
                if head_hash not in history:
                    raise CommitNotFoundError(
                        f"commit {head_hash!r} does not belong to branch {branch_id!r}"
                    )
                # History is newest first, so the suffix includes the selected
                # commit and its ancestors without a second database traversal.
                query = query.where(
                    commits.c.hash.in_(history[history.index(head_hash) :])
                )
            rows = (await connection.execute(query)).mappings().all()
        for row in rows:
            yield self._row_to_commit(row)

    async def trace_commits(
        self,
        run_id: str,
        *,
        branch_id: str | None = None,
        execution_id: str | None = None,
    ) -> tuple[Commit, ...]:
        """Return commits authored by or directly delivered to one run."""
        execution_id = self._execution(execution_id)
        if branch_id is not None:
            branch_commits = [
                commit
                async for commit in self.iter_commits(
                    branch_id, execution_id=execution_id
                )
            ]
            return tuple(
                commit
                for commit in branch_commits
                if commit.author.run_id == run_id
                or getattr(commit.event, "requested_by_run_id", None) == run_id
                or getattr(commit.event, "agent_run_id", None) == run_id
                or getattr(commit.event, "child_run_id", None) == run_id
            )
        async with self._transaction() as connection:
            result = await connection.execute(
                sa.select(commits)
                .where(
                    commits.c.execution_id == execution_id,
                    _trace_filter(run_id),
                )
                .order_by(commits.c.sequence)
            )
            rows = result.mappings().all()
        return tuple(self._row_to_commit(row) for row in rows)

    async def snapshot_count(self) -> int:
        """Return the number of projections persisted as replay accelerators."""
        execution_id = self._execution(None)
        async with self._transaction() as connection:
            return await connection.scalar(
                sa.select(sa.func.count())
                .select_from(snapshots)
                .where(snapshots.c.execution_id == execution_id)
            )


__all__ = ["SQLiteCommitStore"]
